"""
HPC Strategy for Phase A Pollution Filtering (Staged Execution Architecture).

Implements two-stage decoupled execution for maximum throughput:
- Stage 1 (CPU Saturation): Parallel chunking with streaming writes to post_chunked.
- Stage 2 (GPU Saturation): Global-sorted inference with near-zero padding.

Key Optimizations:
- Chunks persisted to Arrow column enable crash recovery.
- Global argsort by token_count minimizes padding waste.
- Label embeddings hoisted to strategy level for bi-encoder efficiency.
- pin_memory + non_blocking for overlapped PCIe transfers.

Reference: Technical Reports on Staged Execution Architecture.
Implements: phaseA-D_implementation_plan.md Section 7.4 (HPC Mode)
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from tqdm import tqdm

from .base import PollutionFilterStrategy
from ..gliner_detector import GLiNERDetector, BatchInferenceConfig
from ..masker import SpanMasker
from ..semantic_chunker import BudgetConfig
from ..explicit_recall import compute_explicit_recall
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer
from ..probe import compute_amnesic_drop
from ..concept_encoding import DemographicEncoder, extract_probe_labels
from ...data_engine.dataset import SOBRDataset
from ...data_engine.schemas import (
    SOBR_SCHEMA,
    POLLUTION_LOG_SCHEMA,
    get_demographic_columns,
    has_post_chunked_column,
    validate_schema_flexible,
)
from ...data_engine.chunking_writer import ChunkingArtifactWriter

logger = logging.getLogger(__name__)


class HPCFilterStrategy(PollutionFilterStrategy):
    """
    HPC-optimized pollution filtering strategy with staged execution.
    
    Architecture (Staged Execution):
    - Stage 1: CPU-saturated parallel chunking → persist to post_chunked column.
    - Stage 2: GPU-saturated global-sorted inference → near-zero padding.
    
    Key Features:
    - Full GPU utilization (A100/H100) via global length sorting.
    - Crash recovery: if post_chunked exists, skip Stage 1.
    - Bi-encoder label embedding caching hoisted to strategy level.
    - BF16 mixed precision (if available).
    
    Implements: FR-12 (HPC Mode), Staged Execution Architecture.
    """
    
    def __init__(self):
        """Initialize HPC strategy with staged execution support."""
        self._last_device: str = "cuda"
        self._last_batch_size: int = 0
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("BF16 mixed precision available")
        logger.info("HPCFilterStrategy initialized (staged execution mode)")
    
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self._last_device)
    
    def get_batch_size(self) -> int:
        """Get batch size."""
        return self._last_batch_size

    def _cfg_get(self, config: Dict[str, Any], path: str) -> Any:
        """Safely get nested config value."""
        cur: Any = config
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Missing required config key: '{path}'")
            cur = cur[part]
        return cur

    def _cfg_get_optional(self, config: Dict[str, Any], path: str, default: Any = None) -> Any:
        """Get nested config value with default."""
        try:
            return self._cfg_get(config, path)
        except KeyError:
            return default

    def _resolve_device(self, device_spec: str) -> str:
        """Resolve device specification to actual device string."""
        spec = str(device_spec).lower()
        if spec == "auto":
            return "cuda" if torch.cuda.is_available() else "cpu"
        if spec in {"cpu", "cuda"}:
            if spec == "cuda" and not torch.cuda.is_available():
                raise RuntimeError("Config requests CUDA but torch.cuda.is_available() is False")
            return spec
        raise ValueError(f"Unsupported device spec '{device_spec}'")
    
    def execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute Phase A with staged execution architecture.
        
        Pipeline:
        1. Load dataset (check for existing post_chunked).
        2. Stage 1: Parallel chunking → persist post_chunked (skip if exists).
        3. Stage 2: Global-sorted inference from post_chunked.
        4. Masking & LEACE computation.
        5. Save outputs.
        """
        logger.info("=" * 80)
        logger.info("Phase A: HPC Strategy (Staged Execution)")
        logger.info("=" * 80)

        # Validate device
        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
        if device != "cuda":
            raise RuntimeError("HPC strategy requires config gliner.device='cuda'")

        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))

        self._last_device = device
        self._last_batch_size = gliner_batch_size
        
        # =======================================================================
        # Step 1: Load Dataset
        # =======================================================================
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(
            arrow_path=input_dataset_path,
            seed=int(self._cfg_get(config, "seed")),
        )
        table = dataset.table

        # Optional subset (strictly config-driven)
        if bool(self._cfg_get(config, "subset.enabled")):
            size = self._cfg_get(config, "subset.size")
            if size is not None and len(table) > int(size):
                rng = np.random.default_rng(int(self._cfg_get(config, "seed")))
                indices = rng.choice(len(table), size=int(size), replace=False)
                logger.info(f"Using subset (seeded): {int(size)}/{len(table)} samples")
                table = table.take(pa.array(indices, type=pa.int64()))
        
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        
        logger.info(f"Loaded {len(posts)} posts")
        
        # =======================================================================
        # Initialize GLiNER Detector & Components
        # =======================================================================
        logger.info("Initializing GLiNER detector and components")
        taxonomy_config, constraints_config = self._load_taxonomy_config(config)
        gliner_cfg = config.get("gliner", {})
        batch_inference_cfg = gliner_cfg.get("batch_inference", {})
        
        batch_config = BatchInferenceConfig(
            enable_batching=batch_inference_cfg.get("enable_batching", True),
            batch_size=gliner_batch_size,
            num_buckets=batch_inference_cfg.get("num_buckets"),
            min_bucket_size=batch_inference_cfg.get("min_bucket_size", 4),
            enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
        )
        
        chunking_cfg = gliner_cfg.get("chunking", {})
        budget_config = BudgetConfig(
            model_max_length=int(self._cfg_get(config, "encoder.max_length")),
            mode=chunking_cfg.get("mode", "single_sentence"),
            legacy_sequential_mode=False,  # Always use staged mode
            parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 16)),
            parallel_chunking_min_texts=int(chunking_cfg.get("parallel_chunking_min_texts", 512)),
            gliner_max_words=int(gliner_cfg.get("gliner_max_words", 512)),
            tokens_per_word_ratio=float(gliner_cfg.get("tokens_per_word_ratio", 1.3)),
        )
        
        require_bi_encoder = bool(gliner_cfg.get("require_bi_encoder", False))
        
        gliner = GLiNERDetector(
            model_name=str(self._cfg_get(config, "gliner.model")),
            device=device,
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            confidence_threshold=float(self._cfg_get(config, "gliner.confidence_threshold")),
            taxonomy_config=taxonomy_config,
            constraints_config=constraints_config,
            budget_config=budget_config,
            batch_inference_config=batch_config,
            require_bi_encoder=require_bi_encoder,
        )
        
        # Hoist label embeddings to strategy level (bi-encoder optimization)
        inference_labels = gliner.taxonomy.get_inference_labels()
        cached_label_embeddings = gliner.get_cached_label_embeddings(inference_labels)
        if cached_label_embeddings is not None:
            logger.info(f"Hoisted label embeddings for {len(inference_labels)} labels")
        
        # =======================================================================
        # Step 2: Stage 1 - CPU Saturated Parallel Chunking
        # =======================================================================
        # Check for resume: if post_chunked exists, skip Stage 1
        has_chunks = has_post_chunked_column(table)
        
        if has_chunks:
            logger.info("Resume detected: post_chunked column exists, skipping Stage 1")
            table_with_chunks = table
        else:
            logger.info("Stage 1/2: CPU-Saturated Parallel Chunking")
            
            # Prepare labels list (full taxonomy for all documents)
            labels_list: List[Optional[List[str]]] = [inference_labels for _ in posts]
            
            # Get tokenizer name for worker initialization
            tokenizer_name = gliner.tokenizer.name_or_path
            words_splitter_type = gliner.chunker._words_splitter_type
            
            num_workers = int(chunking_cfg.get("parallel_chunking_workers", 16))
            micro_batch_size = int(chunking_cfg.get("micro_batch_size", 1000))
            
            chunk_pbar = tqdm(
                total=len(posts),
                desc="Stage 1: Chunking",
                unit="doc",
                dynamic_ncols=True,
            )
            
            def _chunk_progress(count: int) -> None:
                chunk_pbar.n = count
                chunk_pbar.refresh()
            
            try:
                writer = ChunkingArtifactWriter(
                    tokenizer_name=tokenizer_name,
                    budget_config=budget_config,
                    words_splitter_type=words_splitter_type,
                    language="en",
                    num_workers=num_workers,
                    micro_batch_size=micro_batch_size,
                    progress_callback=_chunk_progress,
                )
                
                table_with_chunks = writer.process_and_save(
                    posts=posts,
                    labels_list=labels_list,
                    table=table,
                )
            finally:
                chunk_pbar.close()
            
            logger.info("Stage 1 complete: post_chunked column populated")
            
            # Force garbage collection between stages
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # =======================================================================
        # Step 3: Stage 2 - GPU Saturated Global-Sorted Inference
        # =======================================================================
        logger.info("Stage 2/2: GPU-Saturated Global-Sorted Inference")
        
        post_chunked_column = table_with_chunks["post_chunked"]
        
        # Configure pin_memory for HPC
        pin_memory = bool(self._cfg_get_optional(
            config, "execution.async_prefetch.pin_memory", True
        ))
        
        entities_batch = gliner.inference_from_manifest(
            post_chunked_column=post_chunked_column,
            labels=inference_labels,
            batch_size=gliner_batch_size,
            cached_label_embeddings=cached_label_embeddings,
            pin_memory=pin_memory,
            show_progress=True,
        )
        
        logger.info("Stage 2 complete: inference finished")
        
        # =======================================================================
        # Step 4: Masking & Pollution Logs
        # =======================================================================
        logger.info("Applying typed masks")
        
        masker = SpanMasker(
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
            entity_to_mask=gliner.get_mask_tokens(),
        )
        
        masked_texts, pollution_logs = masker.mask_batch(posts, entities_batch, post_ids)
        total_spans = sum(len(entities) for entities in entities_batch)
        logger.info(f"Masked {total_spans} pollution spans")
        
        # =======================================================================
        # Step 5: Embedding & LEACE
        # =======================================================================
        logger.info("Embedding masked texts and computing LEACE projection")
        
        # Extract mask tokens for tokenizer alignment
        mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            special_tokens=mask_tokens,
        )
        
        encoder = DemographicEncoder(get_demographic_columns()).fit(table_with_chunks)
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            force_cpu=bool(self._cfg_get(config, "leace.force_cpu")),
        )
        
        # Sharded embedding + LEACE accumulation
        shard_size = int(chunking_cfg.get("shard_size", 0) or len(masked_texts))
        if shard_size <= 0:
            shard_size = len(masked_texts)
        
        total_shards = max(1, (len(masked_texts) + shard_size - 1) // shard_size)
        
        probe_enabled = bool(self._cfg_get(config, "probe.compute_amnesic_drop"))
        max_probe_samples = int(self._cfg_get(config, "probe.max_samples")) if probe_enabled else 0
        probe_embeddings: List[torch.Tensor] = []
        
        concept_stats = None
        
        embed_pbar = tqdm(
            total=total_shards,
            desc="Embedding & LEACE",
            unit="shard",
            dynamic_ncols=True,
        )
        
        try:
            for shard_idx in range(total_shards):
                start = shard_idx * shard_size
                end = min(len(masked_texts), start + shard_size)
                
                shard_masked = masked_texts[start:end]
                shard_table = table_with_chunks.slice(start, end - start)
                
                embeddings_shard = embedder.embed_texts(
                    shard_masked,
                    batch_size=encoder_batch_size,
                    show_progress=False,
                )
                
                if probe_enabled and len(probe_embeddings) < max_probe_samples:
                    remaining = max_probe_samples - sum(e.shape[0] for e in probe_embeddings)
                    if remaining > 0:
                        probe_embeddings.append(
                            embeddings_shard[:remaining].detach().to("cpu", dtype=torch.float32)
                        )
                
                concepts_shard = torch.from_numpy(encoder.transform(shard_table))
                concept_stats = leace.accumulate_batch_concepts(
                    embeddings_shard,
                    concepts_shard,
                    concept_stats,
                )
                
                del embeddings_shard
                embed_pbar.update(1)
        finally:
            embed_pbar.close()
        
        if concept_stats is None:
            raise RuntimeError("LEACE accumulation failed: no concept statistics computed")
        
        # Compute projection matrix
        logger.info("Computing LEACE projection matrix")
        projection_matrix = leace.compute_projection_from_concept_stats(concept_stats)
        
        # =======================================================================
        # Step 6: Save Outputs
        # =======================================================================
        logger.info("Saving outputs")
        
        # Save cleaned dataset (without post_chunked to match expected schema)
        self._save_cleaned_dataset(table_with_chunks, masked_texts, output_dataset_path)
        
        # Save projection matrix
        torch.save(projection_matrix.cpu(), projection_matrix_path)
        logger.info(f"Saved projection matrix: {projection_matrix_path}")
        
        # Save pollution logs
        self._save_pollution_logs(pollution_logs, pollution_logs_path)
        
        # =======================================================================
        # Step 7: Compute Metrics & Return
        # =======================================================================
        metadata: Dict[str, Any] = {
            "num_samples": len(table_with_chunks),
            "num_pollution_spans": total_spans,
            "projection_matrix_shape": list(projection_matrix.shape),
            "device": device,
            "use_bf16": self.use_bf16,
            "staged_execution": True,
            "stage1_skipped": has_chunks,
        }
        
        # Explicit recall
        if bool(self._cfg_get(config, "gliner.compute_explicit_recall")):
            recall = compute_explicit_recall(
                posts,
                entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall
        
        # Amnesic drop probe
        if probe_enabled and probe_embeddings:
            embeddings_before = torch.cat(probe_embeddings, dim=0)[:max_probe_samples]
            probe_table = table_with_chunks.slice(0, len(embeddings_before))
            
            P_cpu = projection_matrix.detach().to("cpu", dtype=torch.float32)
            embeddings_after = embeddings_before @ P_cpu.T
            
            by_column: Dict[str, Any] = {}
            drops: List[float] = []
            
            for col in get_demographic_columns():
                labels_np = extract_probe_labels(probe_table, col)
                labels_t = torch.tensor(labels_np, dtype=torch.long)
                acc_before, acc_after, amnesic_drop = compute_amnesic_drop(
                    embeddings_before,
                    embeddings_after,
                    labels_t,
                    train_split=float(self._cfg_get(config, "probe.train_split")),
                    random_state=int(self._cfg_get(config, "seed")),
                )
                by_column[col] = {
                    "accuracy_before": acc_before,
                    "accuracy_after": acc_after,
                    "amnesic_drop": amnesic_drop,
                }
                drops.append(float(amnesic_drop))
            
            min_drop = float(min(drops)) if drops else 0.0
            metadata["probe"] = {
                "by_column": by_column,
                "min_amnesic_drop": min_drop,
                "threshold": float(self._cfg_get(config, "probe.amnesic_drop_threshold")),
                "max_samples": len(embeddings_before),
            }
            
            if bool(self._cfg_get(config, "quality.enforce_thresholds")):
                threshold = float(self._cfg_get(config, "probe.amnesic_drop_threshold"))
                if min_drop < threshold:
                    raise ValueError(
                        f"Amnesic drop gate failed: min={min_drop:.3f} < threshold={threshold:.3f}"
                    )
        
        logger.info("Phase A complete!")
        return metadata
    
    def _save_cleaned_dataset(
        self,
        original_table: pa.Table,
        masked_texts: List[str],
        output_path: Path,
    ) -> None:
        """Save cleaned dataset with post_masked column populated."""
        import pandas as pd
        
        # Remove post_chunked if present (not part of base schema)
        columns_to_keep = [
            name for name in original_table.column_names
            if name != "post_chunked"
        ]
        clean_table = original_table.select(columns_to_keep)
        
        # Convert to pandas, update post_masked, convert back
        df = clean_table.to_pandas()
        df["post_masked"] = masked_texts
        
        cleaned_table = pa.Table.from_pandas(df, schema=SOBR_SCHEMA)
        feather.write_feather(cleaned_table, output_path)
        
        logger.info(f"Saved cleaned dataset: {output_path}")
    
    def _save_pollution_logs(
        self,
        pollution_logs: List[Dict[str, Any]],
        output_path: Path,
    ) -> None:
        """Save pollution logs as Arrow table."""
        if not pollution_logs:
            logger.warning("No pollution logs to save")
            empty_table = pa.Table.from_pylist([], schema=POLLUTION_LOG_SCHEMA)
            feather.write_feather(empty_table, output_path)
            return
        
        logs_table = pa.Table.from_pylist(pollution_logs, schema=POLLUTION_LOG_SCHEMA)
        feather.write_feather(logs_table, output_path)
        
        logger.info(f"Saved {len(pollution_logs)} pollution log entries: {output_path}")

    def _load_taxonomy_config(self, config: Dict[str, Any]) -> tuple:
        """Load taxonomy config (labels, masks, constraints) if provided."""
        taxonomy_path = self._cfg_get(config, "gliner.taxonomy_path")
        if not taxonomy_path:
            raise KeyError("Missing required config key: 'gliner.taxonomy_path'")

        from omegaconf import OmegaConf

        path = Path(str(taxonomy_path))
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[4]
            path = repo_root / path

        if not path.exists():
            raise FileNotFoundError(f"GLiNER taxonomy config not found: {path}")

        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        taxonomy_cfg = cfg.get("taxonomy", {})
        constraints_cfg = taxonomy_cfg.get("width_constraints", {})
        return taxonomy_cfg, constraints_cfg
