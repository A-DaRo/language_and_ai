"""Phase D training loop (baseline vs constrained).

Optimized for high-throughput training on modern GPUs:
- torch.compile with mode="reduce-overhead" for CUDA Graph caching
- FP8 precision via TransformerEngine (Blackwell/Hopper architecture)
- Fused AdamW optimizer to eliminate Python loop overhead
- Async prefetching via DevicePrefetcher
- Quantized bucketing for stable tensor shapes
"""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import logging
import math
import json
from pathlib import Path
from typing import Dict, Optional, Any, Union

import numpy as np
import torch
import torch.nn.functional as F
from torch.utils.data import DataLoader
from tqdm.auto import tqdm

from ..hardware_ops.oom_guard import execute_with_oom_protection
from ..hardware_ops.runtime import RuntimeController
from ..hardware_ops.telemetry import CUDATimer, TelemetryCollector, create_runtime_metrics
from ..data_engine.schemas import get_demographic_columns
from ..data_engine.bucketing import (
    QuantizedBucketSampler,
    DEFAULT_QUANTIZE_STEP,
    create_quantized_sampler,
)
from ..stylometry_net.classification_head import MultiTaskHead
from ..stylometry_net.phase_d_dataset import (
    DevicePrefetcher,
    FastCollator,
    PhaseDBudgetedBatchSampler,
    PhaseDCollator,
    PhaseDDataset,
    PhaseDLabelMaps,
    load_label_maps,
)
from ..stylometry_net.transformer import AffineGuardTransformer
from ..stylometry_net.tokenizer import PhaseDTokenizer
from .checkpointing import config_to_metadata, load_checkpoint, save_checkpoint, save_metadata
from .metrics import compute_task_metrics
from .optimizer import build_optimizer, build_param_groups, build_scheduler

logger = logging.getLogger(__name__)


# ==============================================================================
# FP8 Support Detection (TransformerEngine for Blackwell/Hopper)
# ==============================================================================

_FP8_AVAILABLE = False
_TE_MODULE = None

try:
    import transformer_engine.pytorch as te
    from transformer_engine.common.recipe import Format, DelayedScaling
    _FP8_AVAILABLE = True
    _TE_MODULE = te
    logger.info("TransformerEngine FP8 support available")
except ImportError:
    logger.debug("TransformerEngine not available, FP8 disabled")


def is_fp8_available() -> bool:
    """Check if FP8 precision is available (requires TransformerEngine)."""
    return _FP8_AVAILABLE


def get_fp8_recipe() -> Optional[Any]:
    """Get FP8 recipe for TransformerEngine autocast."""
    if not _FP8_AVAILABLE:
        return None
    return DelayedScaling(
        fp8_format=Format.HYBRID,
        amax_history_len=16,
        amax_compute_algo="max",
    )


@dataclass
class PhaseDTrainConfig:
    """Configuration for Phase D training with AOT optimizations."""
    
    dataset_path: Path
    artifacts_dir: Path
    output_dir: Path
    model_name: str = "roberta-base"
    taxonomy_path: Path = Path("conf/base/gliner_taxonomy.yaml")
    max_length: int = 512
    batch_size: int = 8
    num_epochs: int = 1
    max_steps: Optional[int] = None
    learning_rate: float = 2e-5
    layerwise_lr_decay: float = 1.0
    gradient_accumulation_steps: int = 1
    precision: str = "fp32"  # "fp32", "bf16", "fp8"
    resume_from: Optional[Path] = None
    scheduler_name: str = "linear"
    num_warmup_steps: int = 0
    early_stopping_enabled: bool = False
    early_stopping_patience: int = 2
    early_stopping_min_delta: float = 0.0
    early_stopping_metric: str = "f1_macro"
    save_every_steps: Optional[int] = None
    save_every_epochs: Optional[int] = None
    split_ratios: Dict[str, float] = None
    execution_config: Dict[str, Any] = field(default_factory=dict)
    
    # AOT Pipeline Optimization Settings
    use_aot_mode: bool = True  # Use pre-tokenized data + FastCollator
    use_torch_compile: bool = True  # Apply torch.compile to model
    torch_compile_mode: str = "reduce-overhead"  # CUDA Graph optimization
    compile_train_step: bool = False  # Compile full train step (forward+loss+backward)
    use_cuda_graph_training: bool = False  # Manual CUDA-graph training path
    cuda_graph_training: Dict[str, Any] = field(default_factory=dict)
    use_fused_optimizer: bool = True  # Fused AdamW kernel
    use_device_prefetch: bool = True  # Async H2D transfers
    quantize_step: int = DEFAULT_QUANTIZE_STEP  # Snap-to-Grid step (16)
    token_budget: int = 65536  # Default token budget for quantized sampler


class PhaseDTrainer:
    """
    Train baseline vs constrained models on Phase A outputs.
    
    Supports two execution modes:
    1. Legacy mode: JIT tokenization, standard DataLoader
    2. AOT mode: Pre-tokenized data, quantized bucketing, async prefetch
    
    AOT mode is automatically enabled when:
    - Dataset has 'input_ids' column (from preprocess_tokens.py)
    - config.use_aot_mode is True
    """

    def __init__(self, config: PhaseDTrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        
        # Enable TF32 tensor cores for faster float32 matmuls on Ampere+ GPUs
        if self.device.type == "cuda":
            torch.set_float32_matmul_precision("high")
        self._execution_config = config.execution_config or {}
        autotuning_cfg = self._execution_config.get("autotuning", {})
        self._autotuning_enabled = bool(autotuning_cfg.get("enabled", False))
        self._runtime_controller = (
            RuntimeController.from_config({"execution": {"autotuning": autotuning_cfg}})
            if self._autotuning_enabled
            else None
        )
        self._dynamic_batching_cfg = self._execution_config.get("dynamic_batching", {})
        self._dynamic_batching_enabled = bool(
            self._dynamic_batching_cfg.get("enabled", self._autotuning_enabled)
        )
        
        # Telemetry configuration with strided sampling support
        telemetry_cfg = self._execution_config.get("telemetry", {})
        self._telemetry_enabled = bool(
            telemetry_cfg.get("enabled", False) or self._autotuning_enabled
        )
        self._telemetry_stride = int(telemetry_cfg.get("stride", 1))  # Measure every Nth batch
        self._telemetry = TelemetryCollector.get_instance() if self._telemetry_enabled else None
        self._telemetry_step_counter = 0
        
        self._static_token_budget = int(self.config.batch_size * self.config.max_length)
        self._precision = str(self.config.precision or "fp32").lower()
        
        # AOT mode settings
        self._aot_mode_requested = bool(config.use_aot_mode)
        self._use_device_prefetch = bool(config.use_device_prefetch)
        self._use_torch_compile = bool(config.use_torch_compile)
        self._torch_compile_mode = str(config.torch_compile_mode)
        self._compile_train_step = bool(config.compile_train_step)
        self._use_cuda_graph_training = bool(config.use_cuda_graph_training)
        self._cuda_graph_training_cfg = config.cuda_graph_training or {}
        self._use_fused_optimizer = bool(config.use_fused_optimizer)
        self._quantize_step = int(config.quantize_step)
        self._token_budget = int(config.token_budget)
        self._model_compiled = False
        self._graph_training = None
        
        # FP8 support check
        if self._precision == "fp8":
            if not is_fp8_available():
                logger.warning(
                    "FP8 precision requested but TransformerEngine not available. "
                    "Falling back to BF16."
                )
                self._precision = "bf16"
            else:
                logger.info("FP8 precision enabled via TransformerEngine")
        
        logger.info(
            f"PhaseDTrainer initialized: device={self.device}, precision={self._precision}, "
            f"aot_mode={self._aot_mode_requested}, torch_compile={self._use_torch_compile}, "
            f"compile_train_step={self._compile_train_step}, "
            f"cuda_graph_training={self._use_cuda_graph_training}"
        )

        if self._use_cuda_graph_training:
            if self.device.type != "cuda":
                logger.warning("CUDA graph training requested but CUDA unavailable - disabling.")
                self._use_cuda_graph_training = False
            else:
                from ..hardware_ops.cuda_graphs import create_graph_aware_training_from_config

                if self._use_torch_compile:
                    logger.info(
                        "Disabling torch.compile because CUDA graph training is enabled."
                    )
                    self._use_torch_compile = False
                    self._compile_train_step = False

                graph_cfg = {"cuda_graph_training": self._cuda_graph_training_cfg}
                self._graph_training = create_graph_aware_training_from_config(
                    graph_cfg,
                    default_max_seq_len=self.config.max_length,
                )
                logger.info(
                    "CUDA graph training enabled: ensure bucketed static shapes; "
                    "dropout masks may repeat across graph replays."
                )

    def _build_loader(
        self,
        *,
        text_field: str,
        label_maps: Optional[PhaseDLabelMaps] = None,
        split: Optional[str] = None,
        shuffle: bool = False,
        enable_dynamic_batching: bool = False,
    ) -> tuple[DataLoader, PhaseDLabelMaps]:
        dataset = PhaseDDataset(
            self.config.dataset_path,
            text_field=text_field,
            label_fields=get_demographic_columns(),
            label_maps=label_maps,
            split=split,
            split_ratios=self.config.split_ratios,
            use_aot_tokens=self._aot_mode_requested,
        )
        
        # Determine collator based on dataset mode
        aot_mode_active = dataset.is_aot_mode
        
        if aot_mode_active:
            logger.info("AOT mode active: using FastCollator (zero tokenization)")
            collator = FastCollator(
                max_length=self.config.max_length,
                pad_token_id=1,  # RoBERTa pad token
                quantize_step=self._quantize_step,
                use_pinned_memory=self.device.type == "cuda",
            )
        else:
            logger.info("JIT mode: using PhaseDCollator (runtime tokenization)")
            tokenizer = PhaseDTokenizer(
                model_name=self.config.model_name,
                max_length=self.config.max_length,
                taxonomy_path=self.config.taxonomy_path,
            )
            collator = PhaseDCollator(tokenizer)

        loader_cfg = self._execution_config.get("data_loader", {})
        num_workers = int(loader_cfg.get("num_workers", 0))
        pin_memory = bool(loader_cfg.get("pin_memory", self.device.type == "cuda"))
        persistent_workers = bool(loader_cfg.get("persistent_workers", False))
        prefetch_factor = int(loader_cfg.get("prefetch_factor", 2))
        if num_workers <= 0:
            persistent_workers = False
            prefetch_factor = 2
        loader_kwargs = {
            "num_workers": num_workers,
            "pin_memory": pin_memory,
        }
        if num_workers > 0:
            loader_kwargs["persistent_workers"] = persistent_workers
            loader_kwargs["prefetch_factor"] = prefetch_factor

        # Use QuantizedBucketSampler in AOT mode for CUDA Graph stability
        if aot_mode_active and enable_dynamic_batching:
            token_counts = dataset.get_all_token_counts()
            
            if token_counts is not None:
                logger.info(
                    f"Using QuantizedBucketSampler: {len(token_counts)} samples, "
                    f"token_budget={self._token_budget}, step={self._quantize_step}"
                )
                batch_sampler = create_quantized_sampler(
                    lengths=token_counts,
                    token_budget=self._token_budget,
                    max_length=self.config.max_length,
                    quantize_step=self._quantize_step,
                    shuffle=shuffle,
                    drop_last=False,
                    seed=42,
                )
                loader = DataLoader(
                    dataset,
                    batch_sampler=batch_sampler,
                    collate_fn=collator,
                    **loader_kwargs,
                )
                return loader, dataset.label_maps
            else:
                logger.warning(
                    "AOT mode but token_count column not found. "
                    "Falling back to budgeted sampler."
                )

        if enable_dynamic_batching and self._dynamic_batching_enabled:
            # Legacy budgeted batch sampler
            max_batch_size = int(
                self._dynamic_batching_cfg.get("max_batch_size", self.config.batch_size)
            )
            min_batch_size = int(self._dynamic_batching_cfg.get("min_batch_size", 1))
            drop_last = bool(self._dynamic_batching_cfg.get("drop_last", False))
            shuffle_batches = bool(self._dynamic_batching_cfg.get("shuffle", shuffle))
            seed = int(self._dynamic_batching_cfg.get("seed", 42))
            length_column = str(self._dynamic_batching_cfg.get("length_column", "text_length"))
            length_scale = float(self._dynamic_batching_cfg.get("length_scale", 1.0))

            # Create tokenizer only if not in AOT mode
            if not aot_mode_active:
                tokenizer = PhaseDTokenizer(
                    model_name=self.config.model_name,
                    max_length=self.config.max_length,
                    taxonomy_path=self.config.taxonomy_path,
                )

            def length_fn(index: int) -> int:
                # Prefer token_count in AOT mode
                if aot_mode_active:
                    tc = dataset.get_token_count(index)
                    if tc is not None:
                        return min(tc, int(self.config.max_length))
                cached = dataset.get_length(index, length_column)
                if cached is not None:
                    scaled = max(1, int(cached * length_scale))
                    return min(scaled, int(self.config.max_length))
                text = dataset.get_text(index)
                length = tokenizer.estimate_length(text)
                return min(int(length), int(self.config.max_length))

            budget_provider = (
                self._runtime_controller.get_next_budget
                if self._runtime_controller
                else lambda: self._static_token_budget
            )
            batch_sampler = PhaseDBudgetedBatchSampler(
                dataset,
                length_fn=length_fn,
                budget_provider=budget_provider,
                max_batch_size=max_batch_size,
                min_batch_size=min_batch_size,
                shuffle=shuffle_batches,
                drop_last=drop_last,
                seed=seed,
            )
            loader = DataLoader(
                dataset,
                batch_sampler=batch_sampler,
                collate_fn=collator,
                **loader_kwargs,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=collator,
                **loader_kwargs,
            )
        return loader, dataset.label_maps

    def _build_model(self, *, use_affine_guard: bool) -> tuple[AffineGuardTransformer, MultiTaskHead]:
        if use_affine_guard:
            model = AffineGuardTransformer.from_phase_a(
                self.config.artifacts_dir,
                model_name=self.config.model_name,
                taxonomy_path=self.config.taxonomy_path,
                max_length=self.config.max_length,
            )
        else:
            model = AffineGuardTransformer(
                model_name=self.config.model_name,
                projection_matrix_path=None,
                taxonomy_path=self.config.taxonomy_path,
                max_length=self.config.max_length,
            )

        model = model.to(self.device)
        head = MultiTaskHead(
            hidden_dim=model.config.hidden_size,
            num_labels_per_task=self.label_maps.num_classes(),
        ).to(self.device)
        
        # Apply torch.compile for kernel optimization (CUDA Graphs)
        self._model_compiled = False
        if (
            self._use_torch_compile
            and self.device.type == "cuda"
            and not self._compile_train_step
        ):
            logger.info(
                f"Applying torch.compile to transformer (mode={self._torch_compile_mode})"
            )
            try:
                # Compile the transformer backbone for CUDA Graph caching
                # mode="reduce-overhead" enables automatic CUDA Graph capture
                model = torch.compile(
                    model,
                    mode=self._torch_compile_mode,
                    fullgraph=False,  # Allow graph breaks for flexibility
                )
                # Keep the classification head eager to avoid multi-graph
                # CUDA tensor overwrite issues at the graph boundary.
                self._model_compiled = True
                logger.info("torch.compile applied to transformer (head excluded)")
            except Exception as e:
                logger.warning(f"torch.compile failed, continuing without: {e}")
        elif self._use_torch_compile and self.device.type == "cuda":
            logger.info(
                "Skipping torch.compile on transformer because compile_train_step is enabled"
            )
        
        return model, head

    def _log_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")
    
    def _get_autocast_context(self):
        """Get appropriate autocast context based on precision setting."""
        if self.device.type != "cuda":
            return nullcontext()
        
        if self._precision == "fp8" and is_fp8_available():
            # FP8 via TransformerEngine
            return _TE_MODULE.fp8_autocast(
                enabled=True,
                fp8_recipe=get_fp8_recipe(),
            )
        elif self._precision in {"bf16", "bfloat16"}:
            return torch.autocast(device_type="cuda", dtype=torch.bfloat16)
        elif self._precision in {"fp16", "float16"}:
            return torch.autocast(device_type="cuda", dtype=torch.float16)
        else:
            return nullcontext()

    @torch.no_grad()
    def _evaluate(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        num_classes: Dict[str, int],
    ) -> Dict[str, Dict[str, float]]:
        model.eval()
        head.eval()
        metrics: Dict[str, Dict[str, float]] = {}
        use_bf16 = self.device.type == "cuda" and self._precision in {"bf16", "bfloat16"}

        task_logits: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}
        task_labels: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}

        use_prefetch = self._use_device_prefetch and self.device.type == "cuda"
        batch_iter = (
            DevicePrefetcher(loader, self.device, non_blocking=True)
            if use_prefetch
            else loader
        )
        non_blocking = self.device.type == "cuda"

        eval_pbar = tqdm(
            batch_iter,
            desc="Evaluating",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        
        for batch in eval_pbar:
            if use_prefetch:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
            else:
                input_ids = batch["input_ids"].to(self.device, non_blocking=non_blocking)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=non_blocking
                )
                labels = {
                    k: v.to(self.device, non_blocking=non_blocking)
                    for k, v in batch["labels"].items()
                }

            # Mark CUDA Graph step boundary for compiled models
            if self._model_compiled and self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs["cls_embedding"]
                if self._model_compiled and self.device.type == "cuda":
                    # Break CUDAGraph output aliasing before eager head usage.
                    cls_embedding = cls_embedding.clone()
                logits = head(cls_embedding)

            for task, task_logits_batch in logits.items():
                task_logits[task].append(task_logits_batch.detach().cpu())
                task_labels[task].append(labels[task].detach().cpu())
        
        eval_pbar.close()

        for task in num_classes:
            if not task_logits[task]:
                continue
            logits_cat = torch.cat(task_logits[task], dim=0)
            labels_cat = torch.cat(task_labels[task], dim=0)
            metrics[task] = compute_task_metrics(
                logits_cat,
                labels_cat,
                num_classes=num_classes[task],
            )

        model.train()
        head.train()
        return metrics

    @torch.no_grad()
    def _evaluate_detailed(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        num_classes: Dict[str, int],
    ) -> Dict[str, Dict[str, Any]]:
        model.eval()
        head.eval()
        details: Dict[str, Dict[str, Any]] = {}
        use_bf16 = self.device.type == "cuda" and self._precision in {"bf16", "bfloat16"}

        task_logits: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}
        task_labels: Dict[str, list[torch.Tensor]] = {k: [] for k in num_classes}

        use_prefetch = self._use_device_prefetch and self.device.type == "cuda"
        batch_iter = (
            DevicePrefetcher(loader, self.device, non_blocking=True)
            if use_prefetch
            else loader
        )
        non_blocking = self.device.type == "cuda"

        eval_pbar = tqdm(
            batch_iter,
            desc="Detailed Eval",
            unit="batch",
            leave=False,
            dynamic_ncols=True,
        )
        
        for batch in eval_pbar:
            if use_prefetch:
                input_ids = batch["input_ids"]
                attention_mask = batch["attention_mask"]
                labels = batch["labels"]
            else:
                input_ids = batch["input_ids"].to(self.device, non_blocking=non_blocking)
                attention_mask = batch["attention_mask"].to(
                    self.device, non_blocking=non_blocking
                )
                labels = {
                    k: v.to(self.device, non_blocking=non_blocking)
                    for k, v in batch["labels"].items()
                }

            # Mark CUDA Graph step boundary for compiled models
            if self._model_compiled and self.device.type == "cuda":
                torch.compiler.cudagraph_mark_step_begin()

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                cls_embedding = outputs["cls_embedding"]
                if self._model_compiled and self.device.type == "cuda":
                    # Break CUDAGraph output aliasing before eager head usage.
                    cls_embedding = cls_embedding.clone()
                logits = head(cls_embedding)

            for task, task_logits_batch in logits.items():
                task_logits[task].append(task_logits_batch.detach().cpu())
                task_labels[task].append(labels[task].detach().cpu())
        
        eval_pbar.close()

        for task in num_classes:
            if not task_logits[task]:
                continue
            logits_cat = torch.cat(task_logits[task], dim=0)
            labels_cat = torch.cat(task_labels[task], dim=0)
            details[task] = _compute_detailed_metrics(
                logits_cat,
                labels_cat,
                num_classes=num_classes[task],
            )

        model.train()
        head.train()
        return details

    def _train(
        self,
        *,
        model: AffineGuardTransformer,
        head: MultiTaskHead,
        loader: DataLoader,
        eval_loader: Optional[DataLoader],
        run_dir: Path,
    ) -> None:
        accum_steps = max(1, int(self.config.gradient_accumulation_steps))

        param_groups = build_param_groups(
            model=model,
            head=head,
            base_lr=self.config.learning_rate,
            layerwise_lr_decay=self.config.layerwise_lr_decay,
        )
        optimizer = build_optimizer(
            param_groups=param_groups,
            base_lr=self.config.learning_rate,
            use_fused=self._use_fused_optimizer,
        )

        run_dir.mkdir(parents=True, exist_ok=True)
        checkpoint_dir = run_dir / "checkpoints"
        log_path = run_dir / "training_log.jsonl"
        metrics_path = run_dir / "phase_d_metrics.json"
        checkpoint_path = run_dir / "checkpoint.pt"
        model.train()
        head.train()

        step = 0
        optimizer_step = 0
        start_epoch = 0
        resume_payload = None
        if self.config.resume_from and self.config.resume_from.exists():
            resume_payload = load_checkpoint(self.config.resume_from)
            model.load_state_dict(resume_payload["model_state"])
            head.load_state_dict(resume_payload["head_state"])
            optimizer.load_state_dict(resume_payload["optimizer_state"])
            meta = resume_payload.get("metadata", {})
            checkpoint_meta = meta.get("checkpoint", {})
            start_epoch = int(checkpoint_meta.get("epoch", 0))
            step = int(checkpoint_meta.get("step", 0))
            optimizer_step = int(checkpoint_meta.get("optimizer_step", 0))

        steps_per_epoch = len(loader)
        total_batch_steps = (
            int(self.config.max_steps)
            if self.config.max_steps
            else steps_per_epoch * self.config.num_epochs
        )
        total_optimizer_steps = math.ceil(total_batch_steps / accum_steps)
        scheduler = build_scheduler(
            optimizer=optimizer,
            scheduler_name=self.config.scheduler_name,
            num_warmup_steps=self.config.num_warmup_steps,
            total_training_steps=total_optimizer_steps,
        )
        if scheduler is not None and resume_payload is not None:
            scheduler_state = resume_payload.get("scheduler_state")
            if scheduler_state:
                scheduler.load_state_dict(scheduler_state)

        checkpoint_dir.mkdir(parents=True, exist_ok=True)
        best_score = None
        bad_epochs = 0
        
        # Overall training info
        num_epochs = self.config.num_epochs
        total_batches = len(loader)
        logger.info(
            f"Starting training: {num_epochs} epochs, ~{total_batches} batches/epoch, "
            f"{total_batches * num_epochs} total batches"
        )
        
        compiled_step = None
        if (
            self._use_torch_compile
            and self.device.type == "cuda"
            and self._compile_train_step
        ):
            logger.info(
                f"Compiling full train step (forward+loss+backward) "
                f"(mode={self._torch_compile_mode})"
            )

            def _compiled_step(
                input_ids: torch.Tensor,
                attention_mask: torch.Tensor,
                labels: tuple[torch.Tensor, ...],
            ) -> tuple[torch.Tensor, torch.Tensor]:
                cls_embedding = model.forward_cls(
                    input_ids=input_ids,
                    attention_mask=attention_mask,
                )
                logits = head.forward_compiled(cls_embedding)
                loss, valid_flag = head.compute_loss_compiled(logits, labels)
                (loss / accum_steps).backward()
                return loss, valid_flag

            compiled_step = torch.compile(
                _compiled_step,
                mode=self._torch_compile_mode,
                fullgraph=True,
            )
            logger.info("torch.compile applied to full train step")

        use_prefetch = self._use_device_prefetch and self.device.type == "cuda"
        non_blocking = self.device.type == "cuda"

        for epoch in range(start_epoch, num_epochs):
            optimizer.zero_grad(set_to_none=not self._use_cuda_graph_training)
            accum_counter = 0
            epoch_loss_sum = 0.0
            epoch_loss_count = 0
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            
            # Progress bar for batches within epoch - single persistent bar
            batch_iter = (
                DevicePrefetcher(loader, self.device, non_blocking=True)
                if use_prefetch
                else loader
            )
            batch_pbar = tqdm(
                batch_iter,
                desc=f"Epoch {epoch + 1}/{num_epochs}",
                unit="batch",
                leave=True,  # Keep bar visible after epoch completes
                ncols=100,  # Fixed width for consistency
            )
            
            for batch in batch_pbar:
                if use_prefetch:
                    input_ids = batch["input_ids"]
                    attention_mask = batch["attention_mask"]
                    labels = batch["labels"]
                else:
                    input_ids = batch["input_ids"].to(
                        self.device, non_blocking=non_blocking
                    )
                    attention_mask = batch["attention_mask"].to(
                        self.device, non_blocking=non_blocking
                    )
                    labels = {
                        k: v.to(self.device, non_blocking=non_blocking)
                        for k, v in batch["labels"].items()
                    }

                tokens = int(attention_mask.sum().item())
                batch_size = int(attention_mask.size(0))
                
                # Strided telemetry: only measure every Nth batch to reduce overhead
                self._telemetry_step_counter += 1
                should_measure = (
                    self._telemetry_enabled 
                    and self._telemetry_step_counter % self._telemetry_stride == 0
                )

                def handle_oom(_event) -> None:
                    nonlocal accum_counter
                    optimizer.zero_grad(set_to_none=not self._use_cuda_graph_training)
                    accum_counter = 0

                def run_step() -> tuple[Optional[float], bool]:
                    nonlocal accum_counter, optimizer_step
                    
                    # Mark CUDA Graph step boundary BEFORE any computation to prevent
                    # tensor overwrite errors when using torch.compile with mode="reduce-overhead"
                    # This must be called before both forward AND backward passes
                    if (
                        self._use_torch_compile
                        and self.device.type == "cuda"
                        and not self._use_cuda_graph_training
                    ):
                        torch.compiler.cudagraph_mark_step_begin()
                    
                    # Use strided telemetry to avoid synchronization overhead
                    timer_ctx = CUDATimer(synchronize=False) if should_measure else nullcontext()
                    with timer_ctx as timer:
                        # Use precision-aware autocast context
                        autocast_ctx = self._get_autocast_context()
                        with autocast_ctx:
                            if compiled_step is not None:
                                labels_tuple = tuple(
                                    labels[task] for task in head.task_order
                                )
                                loss_tensor, did_backward = compiled_step(
                                    input_ids,
                                    attention_mask,
                                    labels_tuple,
                                )
                                did_backward = bool(did_backward.item())
                            elif self._graph_training is not None:
                                labels_tuple = tuple(
                                    labels[task] for task in head.task_order
                                )
                                pad_token_id = getattr(
                                    getattr(model, "tokenizer", None),
                                    "tokenizer",
                                    None,
                                )
                                pad_token_id = getattr(pad_token_id, "pad_token_id", None)
                                loss_tensor, valid_flag, _ = self._graph_training.run(
                                    model=model,
                                    head=head,
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                    labels=labels_tuple,
                                    accum_steps=accum_steps,
                                    pad_token_id=pad_token_id,
                                )
                                did_backward = bool(valid_flag.item())
                            else:
                                outputs = model(
                                    input_ids=input_ids,
                                    attention_mask=attention_mask,
                                )
                                cls_embedding = outputs["cls_embedding"]
                                if self._model_compiled and self.device.type == "cuda":
                                    # Break CUDAGraph output aliasing before eager head usage.
                                    cls_embedding = cls_embedding.clone()
                                logits = head(cls_embedding)
                                loss_tensor = head.compute_loss(logits, labels)
                                if loss_tensor is None:
                                    return None, False
                                (loss_tensor / accum_steps).backward()
                                did_backward = True

                        if not did_backward:
                            return None, False

                        loss_value = float(loss_tensor.item())

                        did_step = False
                        accum_counter += 1
                        if accum_counter >= accum_steps:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=not self._use_cuda_graph_training)
                            if scheduler is not None:
                                scheduler.step()
                            optimizer_step += 1
                            accum_counter = 0
                            did_step = True

                    # Record telemetry only on strided steps (no synchronize in CUDATimer)
                    if should_measure and self._telemetry is not None and timer is not None:
                        self._telemetry.record_batch(
                            batch_size=batch_size,
                            tokens=tokens,
                            elapsed_ms=timer.elapsed_ms,
                        )
                        if self._runtime_controller is not None:
                            runtime_metrics = create_runtime_metrics(
                                timer,
                                tokens=tokens,
                                batch_size=batch_size,
                            )
                            self._runtime_controller.report_metrics(runtime_metrics)

                    return loss_value, did_step

                loss_value, did_step = execute_with_oom_protection(
                    run_step,
                    controller=self._runtime_controller,
                    on_oom=handle_oom,
                )

                if loss_value is None:
                    continue

                # Track epoch loss for averaging
                epoch_loss_sum += loss_value
                epoch_loss_count += 1
                
                # Update batch progress bar with current loss
                batch_pbar.set_postfix({
                    "loss": f"{loss_value:.4f}",
                    "avg": f"{epoch_loss_sum / epoch_loss_count:.4f}",
                })

                if did_step:
                    step_metrics = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "epoch": epoch + 1,
                        "step": step,
                        "optimizer_step": optimizer_step,
                        "loss": loss_value,
                    }
                    self._log_jsonl(log_path, step_metrics)

                step += 1
                if self.config.save_every_steps and optimizer_step > 0:
                    if optimizer_step % self.config.save_every_steps == 0:
                        save_checkpoint(
                            checkpoint_dir / f"checkpoint_step_{optimizer_step}.pt",
                            model_state=model.state_dict(),
                            head_state=head.state_dict(),
                            optimizer_state=optimizer.state_dict(),
                            scheduler_state=scheduler.state_dict() if scheduler else None,
                            metadata={
                                "config": config_to_metadata(self.config),
                                "label_maps": self.label_maps.maps,
                                "checkpoint": {
                                    "epoch": epoch + 1,
                                    "step": step,
                                    "optimizer_step": optimizer_step,
                                },
                            },
                        )
                if self.config.max_steps and step >= self.config.max_steps:
                    break
            
            # Close batch progress bar after epoch
            batch_pbar.close()
            
            # Calculate epoch average loss
            epoch_avg_loss = epoch_loss_sum / max(1, epoch_loss_count)
            
            if accum_counter > 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                optimizer_step += 1
            
            # Log epoch summary
            logger.info(
                f"Epoch {epoch + 1}/{num_epochs} complete: "
                f"avg_loss={epoch_avg_loss:.4f}, optimizer_steps={optimizer_step}"
            )
            
            if self.config.save_every_epochs and (epoch + 1) % self.config.save_every_epochs == 0:
                save_checkpoint(
                    checkpoint_dir / f"checkpoint_epoch_{epoch + 1}.pt",
                    model_state=model.state_dict(),
                    head_state=head.state_dict(),
                    optimizer_state=optimizer.state_dict(),
                    scheduler_state=scheduler.state_dict() if scheduler else None,
                    metadata={
                        "config": config_to_metadata(self.config),
                        "label_maps": self.label_maps.maps,
                        "checkpoint": {
                            "epoch": epoch + 1,
                            "step": step,
                            "optimizer_step": optimizer_step,
                        },
                    },
                )
            if self.config.early_stopping_enabled and eval_loader is not None:
                eval_metrics = self._evaluate(
                    model=model,
                    head=head,
                    loader=eval_loader,
                    num_classes=self.label_maps.num_classes(),
                )
                score = _aggregate_metric(
                    eval_metrics,
                    metric=self.config.early_stopping_metric,
                )
                if best_score is None or score > best_score + self.config.early_stopping_min_delta:
                    best_score = score
                    bad_epochs = 0
                else:
                    bad_epochs += 1
                    if bad_epochs >= self.config.early_stopping_patience:
                        logger.info(f"Early stopping at epoch {epoch + 1}")
                        break
            if self.config.max_steps and step >= self.config.max_steps:
                logger.info(f"Reached max_steps={self.config.max_steps}")
                break
        
        # Final training summary
        logger.info(
            f"Training complete: {optimizer_step} optimizer steps, "
            f"{step} total batches"
        )
        
        model.train()
        head.train()

        torch.save(model.state_dict(), run_dir / "model.pt")
        torch.save(head.state_dict(), run_dir / "head.pt")
        metrics = self._evaluate(
            model=model,
            head=head,
            loader=eval_loader or loader,
            num_classes=self.label_maps.num_classes(),
        )
        save_metadata(metrics_path, metrics)
        if eval_loader is not None:
            details = self._evaluate_detailed(
                model=model,
                head=head,
                loader=eval_loader,
                num_classes=self.label_maps.num_classes(),
            )
            save_metadata(run_dir / "phase_d_evaluation_details.json", details)
        save_checkpoint(
            checkpoint_path,
            model_state=model.state_dict(),
            head_state=head.state_dict(),
            optimizer_state=optimizer.state_dict(),
            scheduler_state=scheduler.state_dict() if scheduler else None,
            metadata={
                "config": config_to_metadata(self.config),
                "label_maps": self.label_maps.maps,
                "metrics": metrics,
                "checkpoint": {
                    "epoch": self.config.num_epochs,
                    "step": step,
                    "optimizer_step": optimizer_step,
                },
            },
        )


def _aggregate_metric(metrics: Dict[str, Dict[str, float]], *, metric: str) -> float:
    values = []
    for task_metrics in metrics.values():
        if metric in task_metrics:
            values.append(float(task_metrics[metric]))
    if not values:
        return 0.0
    return float(sum(values) / len(values))


def _compute_detailed_metrics(
    logits: torch.Tensor,
    labels: torch.Tensor,
    num_classes: int,
    ignore_index: int = -1,
) -> Dict[str, Any]:
    mask = labels != ignore_index
    if mask.sum().item() == 0:
        return {
            "accuracy": 0.0,
            "f1_macro": 0.0,
            "confusion_matrix": [[0 for _ in range(num_classes)] for _ in range(num_classes)],
            "per_class": {},
        }

    logits = logits[mask]
    labels = labels[mask]
    preds = torch.argmax(logits, dim=-1)

    conf = torch.zeros((num_classes, num_classes), dtype=torch.long)
    for true, pred in zip(labels, preds):
        conf[int(true), int(pred)] += 1

    per_class: Dict[str, Dict[str, float]] = {}
    for cls in range(num_classes):
        tp = conf[cls, cls].item()
        fp = conf[:, cls].sum().item() - tp
        fn = conf[cls, :].sum().item() - tp
        precision = tp / (tp + fp) if (tp + fp) > 0 else 0.0
        recall = tp / (tp + fn) if (tp + fn) > 0 else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        per_class[str(cls)] = {
            "precision": float(precision),
            "recall": float(recall),
            "f1": float(f1),
            "support": int(conf[cls, :].sum().item()),
        }

    overall = compute_task_metrics(logits, labels, num_classes=num_classes, ignore_index=ignore_index)
    return {
        "accuracy": overall["accuracy"],
        "f1_macro": overall["f1_macro"],
        "confusion_matrix": conf.tolist(),
        "per_class": per_class,
    }

    def train_baseline_and_constrained(self, *, use_affine_guard: bool = True) -> None:
        label_maps = load_label_maps(self.config.dataset_path, get_demographic_columns())
        self.label_maps = label_maps
        baseline_loader, _ = self._build_loader(
            text_field="post",
            label_maps=label_maps,
            split="train",
        )
        baseline_eval_loader, _ = self._build_loader(
            text_field="post",
            label_maps=label_maps,
            split="val",
        )
        baseline_model, baseline_head = self._build_model(use_affine_guard=False)

        self._train(
            model=baseline_model,
            head=baseline_head,
            loader=baseline_loader,
            eval_loader=baseline_eval_loader,
            run_dir=self.config.output_dir / "baseline",
        )
        if use_affine_guard:
            constrained_loader, _ = self._build_loader(
                text_field="post_masked",
                label_maps=label_maps,
                split="train",
            )
            constrained_eval_loader, _ = self._build_loader(
                text_field="post_masked",
                label_maps=label_maps,
                split="val",
            )
            constrained_model, constrained_head = self._build_model(use_affine_guard=True)
            self._train(
                model=constrained_model,
                head=constrained_head,
                loader=constrained_loader,
                eval_loader=constrained_eval_loader,
                run_dir=self.config.output_dir / "constrained",
            )
