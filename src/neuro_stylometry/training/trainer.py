"""Phase D training loop (baseline vs constrained)."""

from __future__ import annotations

from contextlib import nullcontext
from dataclasses import dataclass, field
from datetime import datetime, timezone
import math
import json
from pathlib import Path
from typing import Dict, Optional, Any

import torch
from torch.utils.data import DataLoader
from tqdm import tqdm

from ..hardware_ops.oom_guard import execute_with_oom_protection
from ..hardware_ops.runtime import RuntimeController
from ..hardware_ops.telemetry import CUDATimer, TelemetryCollector, create_runtime_metrics
from ..data_engine.schemas import get_demographic_columns
from ..stylometry_net.classification_head import MultiTaskHead
from ..stylometry_net.phase_d_dataset import (
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


@dataclass
class PhaseDTrainConfig:
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
    precision: str = "fp32"
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


class PhaseDTrainer:
    """Train baseline vs constrained models on Phase A outputs."""

    def __init__(self, config: PhaseDTrainConfig) -> None:
        self.config = config
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
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
        self._telemetry_enabled = bool(
            self._execution_config.get("telemetry", {}).get("enabled", False)
            or self._autotuning_enabled
        )
        self._telemetry = TelemetryCollector.get_instance() if self._telemetry_enabled else None
        self._static_token_budget = int(self.config.batch_size * self.config.max_length)
        self._precision = str(self.config.precision or "fp32").lower()

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
        )
        tokenizer = PhaseDTokenizer(
            model_name=self.config.model_name,
            max_length=self.config.max_length,
            taxonomy_path=self.config.taxonomy_path,
        )
        collator = PhaseDCollator(tokenizer)

        if enable_dynamic_batching and self._dynamic_batching_enabled:
            max_batch_size = int(
                self._dynamic_batching_cfg.get("max_batch_size", self.config.batch_size)
            )
            min_batch_size = int(self._dynamic_batching_cfg.get("min_batch_size", 1))
            drop_last = bool(self._dynamic_batching_cfg.get("drop_last", False))
            shuffle_batches = bool(self._dynamic_batching_cfg.get("shuffle", shuffle))
            seed = int(self._dynamic_batching_cfg.get("seed", 42))

            def length_fn(index: int) -> int:
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
                num_workers=0,
            )
        else:
            loader = DataLoader(
                dataset,
                batch_size=self.config.batch_size,
                shuffle=shuffle,
                collate_fn=collator,
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
        return model, head

    def _log_jsonl(self, path: Path, payload: Dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        with path.open("a", encoding="utf-8") as handle:
            handle.write(json.dumps(payload) + "\n")

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

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = head(outputs["cls_embedding"])

            for task, task_logits_batch in logits.items():
                task_logits[task].append(task_logits_batch.detach().cpu())
                task_labels[task].append(labels[task].detach().cpu())

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

        for batch in loader:
            input_ids = batch["input_ids"].to(self.device)
            attention_mask = batch["attention_mask"].to(self.device)
            labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

            autocast_ctx = (
                torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                if use_bf16
                else nullcontext()
            )
            with autocast_ctx:
                outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                logits = head(outputs["cls_embedding"])

            for task, task_logits_batch in logits.items():
                task_logits[task].append(task_logits_batch.detach().cpu())
                task_labels[task].append(labels[task].detach().cpu())

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
        use_bf16 = self.device.type == "cuda" and self._precision in {"bf16", "bfloat16"}

        param_groups = build_param_groups(
            model=model,
            head=head,
            base_lr=self.config.learning_rate,
            layerwise_lr_decay=self.config.layerwise_lr_decay,
        )
        optimizer = build_optimizer(
            param_groups=param_groups,
            base_lr=self.config.learning_rate,
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
        for epoch in range(start_epoch, self.config.num_epochs):
            optimizer.zero_grad(set_to_none=True)
            accum_counter = 0
            batch_sampler = getattr(loader, "batch_sampler", None)
            if hasattr(batch_sampler, "set_epoch"):
                batch_sampler.set_epoch(epoch)
            progress = tqdm(loader, desc=f"Epoch {epoch + 1}", leave=False)
            for batch in progress:
                input_ids = batch["input_ids"].to(self.device)
                attention_mask = batch["attention_mask"].to(self.device)
                labels = {k: v.to(self.device) for k, v in batch["labels"].items()}

                tokens = int(attention_mask.sum().item())
                batch_size = int(attention_mask.size(0))

                def handle_oom(_event) -> None:
                    nonlocal accum_counter
                    optimizer.zero_grad(set_to_none=True)
                    accum_counter = 0

                def run_step() -> tuple[Optional[float], bool]:
                    nonlocal accum_counter, optimizer_step
                    timer_ctx = CUDATimer() if self._telemetry_enabled else nullcontext()
                    with timer_ctx as timer:
                        autocast_ctx = (
                            torch.autocast(device_type="cuda", dtype=torch.bfloat16)
                            if use_bf16
                            else nullcontext()
                        )
                        with autocast_ctx:
                            outputs = model(input_ids=input_ids, attention_mask=attention_mask)
                            logits = head(outputs["cls_embedding"])
                            loss = head.compute_loss(logits, labels)

                        if loss is None:
                            return None, False

                        loss_value = float(loss.item())
                        loss = loss / accum_steps
                        loss.backward()

                        did_step = False
                        accum_counter += 1
                        if accum_counter >= accum_steps:
                            optimizer.step()
                            optimizer.zero_grad(set_to_none=True)
                            if scheduler is not None:
                                scheduler.step()
                            optimizer_step += 1
                            accum_counter = 0
                            did_step = True

                    if self._telemetry_enabled and self._telemetry is not None and timer is not None:
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

                if did_step:
                    step_metrics = {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "epoch": epoch + 1,
                        "step": step,
                        "optimizer_step": optimizer_step,
                        "loss": loss_value,
                    }
                    self._log_jsonl(log_path, step_metrics)
                    progress.set_postfix({"loss": f"{loss_value:.4f}"})

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
            if accum_counter > 0:
                optimizer.step()
                optimizer.zero_grad(set_to_none=True)
                if scheduler is not None:
                    scheduler.step()
                optimizer_step += 1
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
                        break
            if self.config.max_steps and step >= self.config.max_steps:
                break
        # Ensure eval mode after early stop break
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
