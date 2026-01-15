"""
Laptop Strategy for Phase A Pollution Filtering (Staged Execution Architecture).

Implements four-stage decoupled execution optimized for limited hardware:
- Stage 1 (Chunking): Parallel chunking with smaller micro-batches → post_chunked.
- Stage 2 (Inference): Manifest-based inference with retry logic.
- Stage 3 (LEACE): Masking + embedding + batch-accumulated LEACE projection.
- Stage 4 (Probing): Amnesic drop metrics and visualizations.

Key Optimizations:
- Smaller micro-batches (100-200 docs) for low RAM.
- Reduced parallel workers (4 default) for laptop cores.
- CPU fallback path for systems without CUDA.
- Batch accumulation for LEACE covariance statistics.
- RuntimeController autotuning for adaptive batch sizing (OOM resilience).
- Retry rounds for completeness guard on OOM recovery.

Skip Flags:
- --skip-chunking: Start at inference (assumes post_chunked populated)
- --skip-inference: Start at LEACE (assumes inference_results.arrow exists)
- --skip-leace: Start at probing (assumes projection_matrix.pt exists)
- --skip-probing: Exit after LEACE (skip metrics/visualizations)

Reference: Technical Reports on Staged Execution Architecture.
Implements: phaseA-D_implementation_plan.md Section 7.3 (Laptop Mode)
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from tqdm import tqdm


def _convert_numpy_types(obj: Any) -> Any:
    """Recursively convert numpy types to Python native types for JSON serialization."""
    if isinstance(obj, dict):
        return {k: _convert_numpy_types(v) for k, v in obj.items()}
    elif isinstance(obj, list):
        return [_convert_numpy_types(v) for v in obj]
    elif isinstance(obj, (np.bool_, np.bool8)):
        return bool(obj)
    elif isinstance(obj, (np.integer, np.int64, np.int32)):
        return int(obj)
    elif isinstance(obj, (np.floating, np.float64, np.float32)):
        return float(obj)
    elif isinstance(obj, np.ndarray):
        return obj.tolist()
    return obj


def _cast_strings_to_large(table: pa.Table) -> pa.Table:
    """Cast string columns to large_string to avoid offset overflow on take/filter ops."""
    new_fields = []
    for field in table.schema:
        if pa.types.is_string(field.type):
            new_fields.append(pa.field(field.name, pa.large_string(), nullable=field.nullable))
        elif pa.types.is_large_string(field.type):
            new_fields.append(field)
        else:
            new_fields.append(field)
    
    # Only cast if there are actual string columns to convert
    has_regular_strings = any(pa.types.is_string(f.type) for f in table.schema)
    if not has_regular_strings:
        return table
    
    new_schema = pa.schema(new_fields)
    return table.cast(new_schema)


def _safe_take_subset(table: pa.Table, indices: np.ndarray) -> pa.Table:
    """Take a subset of rows using pandas to avoid Arrow offset overflow issues.
    
    When taking random indices from a very large table, PyArrow's take() can fail
    with offset overflow even with large_string types. This function converts
    to pandas for the subsetting operation which handles large strings better.
    """
    # Use pandas iloc for safe row subsetting
    df = table.to_pandas()
    df_subset = df.iloc[indices].reset_index(drop=True)
    # Convert back to Arrow with string inference disabled to preserve types
    return pa.Table.from_pandas(df_subset, preserve_index=False)

from .base import (
    PollutionFilterStrategy,
    ChunkingContext,
    InferenceContext,
    LEACEContext,
    ProbingContext,
    SkipStagesConfig,
)
from ..gliner_detector import GLiNERDetector, BatchInferenceConfig, InferenceResult
from ..masker import SpanMasker
from ..semantic_chunker import BudgetConfig
from ..explicit_recall import compute_explicit_recall
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer
from ..probe import (
    compute_amnesic_drop,
    compute_amnesic_drop_extended,
    ProbeConfig,
)
from ..concept_encoding import DemographicEncoder, extract_probe_labels
from ...data_engine.dataset import SOBRDataset
from ...data_engine.schemas import (
    SOBR_SCHEMA,
    POLLUTION_LOG_SCHEMA,
    get_demographic_columns,
    has_post_chunked_column,
    count_missing_post_chunked,
    is_post_chunked_fully_populated,
    validate_schema_flexible,
)
from ...data_engine.chunking_writer import ChunkingArtifactWriter, save_chunked_table_atomic
from ...data_engine.chunking_validation import find_invalid_post_chunked_indices
from ...hardware_ops.runtime import RuntimeController, RuntimeConfig
from ...evaluation.metrics import (
    compute_embedding_separability,
    compute_class_imbalance_metrics,
)

logger = logging.getLogger(__name__)


class LaptopFilterStrategy(PollutionFilterStrategy):
    """
    Laptop-optimized pollution filtering strategy with staged execution.
    
    Architecture (Four-Stage Execution):
    - Stage 1 (Chunking): CPU chunking with smaller micro-batches → post_chunked.
    - Stage 2 (Inference): Manifest-based inference with retry rounds.
    - Stage 3 (LEACE): Masking + embedding + batch-accumulated projection.
    - Stage 4 (Probing): Amnesic drop metrics and visualizations.
    
    Key Features:
    - Works on CPU or low-VRAM GPU (4-8GB).
    - Smaller micro-batches (100-200 docs vs 1000 for HPC).
    - Reduced parallel workers (4 vs 16 for HPC).
    - Crash recovery: if artifacts exist, skip stages.
    - Batch accumulation for LEACE (memory-efficient).
    - Retry rounds for OOM recovery (completeness guard).
    
    Skip Flags (entry-point / early-exit):
    - skip_chunking: Start at inference (post_chunked must exist).
    - skip_inference: Start at LEACE (inference_results.arrow must exist).
    - skip_leace: Start at probing (projection_matrix.pt must exist).
    - skip_probing: Exit after LEACE (skip metrics/visualizations).
    
    Implements: FR-11 (Laptop Mode), Staged Execution Architecture.
    """
    
    def __init__(self):
        """Initialize laptop strategy with staged execution support."""
        self._last_device: str = "cpu"
        self._last_batch_size: int = 0
        logger.info("LaptopFilterStrategy initialized (staged execution mode)")
    
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
                logger.warning("Config requests CUDA but not available, falling back to CPU")
                return "cpu"
            return spec
        raise ValueError(f"Unsupported device spec '{device_spec}'")
    
    def _cleanup_memory(self) -> None:
        """
        Force garbage collection and clear CUDA cache.
        
        Should be called after each stage to ensure memory is freed
        before the next stage begins. Critical for laptop mode with limited RAM.
        """
        gc.collect()
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            torch.cuda.synchronize()
        logger.debug("Memory cleanup completed")
    
    def _create_runtime_controller(self, config: Dict[str, Any]) -> RuntimeController:
        """
        Create RuntimeController from config for autotuning.
        
        Reads from execution.autotuning section in config.
        Falls back to laptop-optimized defaults if not specified.
        
        Args:
            config: Pipeline configuration dict.
            
        Returns:
            RuntimeController configured for laptop inference.
        """
        autotuning_cfg = self._cfg_get_optional(config, "execution.autotuning", {})
        
        # Check if autotuning is explicitly disabled
        if not autotuning_cfg.get("enabled", True):
            logger.info("Autotuning disabled in config; using minimal controller")
            # Return a minimal controller that won't change batch sizes
            return RuntimeController(RuntimeConfig(
                warmup_batches=0,
                initial_token_budget=2**30,  # Very large, effectively no limit
                min_token_budget=2**30,
                max_token_budget=2**30,
            ))
        
        # Laptop defaults (conservative for 8-16GB systems)
        runtime_config = RuntimeConfig(
            warmup_batches=int(autotuning_cfg.get("warmup_batches", 5)),  # Shorter warmup
            initial_token_budget=int(autotuning_cfg.get("initial_token_budget", 8192)),  # Conservative
            min_token_budget=int(autotuning_cfg.get("min_token_budget", 2048)),
            max_token_budget=int(autotuning_cfg.get("max_token_budget", 65536)),  # Lower ceiling
            memory_headroom_mb=float(autotuning_cfg.get("memory_headroom_mb", 1024)),  # 1GB headroom
            scale_up_factor=float(autotuning_cfg.get("scale_up_factor", 1.15)),  # More cautious
            scale_down_factor=float(autotuning_cfg.get("scale_down_factor", 0.8)),
            oom_slash_factor=float(autotuning_cfg.get("oom_slash_factor", 0.4)),  # More aggressive slash
            stability_threshold=float(autotuning_cfg.get("stability_threshold", 0.15)),
            history_window=int(autotuning_cfg.get("history_window", 8)),
            recovery_patience=int(autotuning_cfg.get("recovery_patience", 3)),  # Shorter patience
        )
        
        return RuntimeController(runtime_config)
    
    # =======================================================================
    # Stage 1: Chunking Stage
    # =======================================================================
    def _run_chunking_stage(
        self,
        input_dataset_path: Path,
        table: pa.Table,
        posts: List[str],
        post_ids: List[str],
        gliner: GLiNERDetector,
        inference_labels: List[str],
        config: Dict[str, Any],
    ) -> ChunkingContext:
        """
        Execute Stage 1: Parallel chunking to populate post_chunked column.
        
        Args:
            input_dataset_path: Path to input Arrow dataset (for atomic save).
            table: PyArrow table to process.
            posts: List of post texts.
            post_ids: List of post IDs.
            gliner: Initialized GLiNER detector (for tokenizer).
            inference_labels: Labels for taxonomy.
            config: Pipeline configuration.
            
        Returns:
            ChunkingContext with table containing post_chunked column.
        """
        logger.info("=" * 60)
        logger.info("Stage 1/4: Chunking (Laptop Micro-Batches)")
        logger.info("=" * 60)
        
        chunking_cfg = config.get("gliner", {}).get("chunking", {})
        
        # Check for resume: if post_chunked exists and is fully populated, skip
        invalid_chunk_rows: List[int] = []
        
        if is_post_chunked_fully_populated(table):
            logger.info("Resume candidate detected: validating post_chunked against raw posts")
            invalid_chunk_rows = find_invalid_post_chunked_indices(
                posts=posts,
                post_chunked_column=table["post_chunked"],
            )
            if not invalid_chunk_rows:
                logger.info("Resume validated: post_chunked fully populated; skipping Stage 1")
                return ChunkingContext(
                    table_with_chunks=table,
                    posts=posts,
                    post_ids=post_ids,
                    stage_skipped=True,
                )
            else:
                logger.warning(f"Resume validation failed for {len(invalid_chunk_rows)} rows")
        
        # Reload without mmap for atomic save (Windows compatibility)
        dataset_stage1 = SOBRDataset(
            arrow_path=input_dataset_path,
            seed=int(self._cfg_get(config, "seed")),
            memory_map=False,
        )
        table = dataset_stage1.table
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        
        # Prepare labels list
        labels_list: List[Optional[List[str]]] = [inference_labels for _ in posts]
        
        # Get tokenizer config
        tokenizer_name = gliner.tokenizer.name_or_path
        words_splitter_type = gliner.chunker._words_splitter_type
        budget_config = gliner.chunker.budget_config
        
        # Laptop-optimized: fewer workers, smaller micro-batches
        num_workers = int(chunking_cfg.get("parallel_chunking_workers", 4))
        micro_batch_size = int(chunking_cfg.get("micro_batch_size", 200))
        
        missing = (
            len(invalid_chunk_rows)
            if invalid_chunk_rows
            else (count_missing_post_chunked(table) if has_post_chunked_column(table) else len(posts))
        )
        
        chunk_pbar = tqdm(
            total=missing,
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
                indices_to_recompute=invalid_chunk_rows if invalid_chunk_rows else None,
            )
        finally:
            chunk_pbar.close()
        
        logger.info("Stage 1 complete: post_chunked column populated")
        
        # Atomic save for crash recovery
        save_chunked_table_atomic(table_with_chunks, input_dataset_path)
        
        # Safety check
        force_skip = bool(self._cfg_get_optional(config, "execution.force_skip", False))
        if not is_post_chunked_fully_populated(table_with_chunks):
            if not force_skip:
                raise RuntimeError("Stage 1 finished but post_chunked still contains NULLs")
            else:
                logger.warning("Stage 1 finished but post_chunked contains NULLs; continuing due to execution.force_skip=True")
        
        # Cleanup between stages (critical for low RAM)
        self._cleanup_memory()
        
        return ChunkingContext(
            table_with_chunks=table_with_chunks,
            posts=posts,
            post_ids=post_ids,
            stage_skipped=False,
            invalid_rows_recomputed=len(invalid_chunk_rows),
        )
    
    # =======================================================================
    # Stage 2: Inference Stage
    # =======================================================================
    def _run_inference_stage(
        self,
        chunking_ctx: ChunkingContext,
        output_dir: Path,
        gliner: GLiNERDetector,
        inference_labels: List[str],
        cached_label_embeddings: Optional[torch.Tensor],
        config: Dict[str, Any],
    ) -> InferenceContext:
        """
        Execute Stage 2: Manifest-based inference with retry rounds.
        
        Args:
            chunking_ctx: Context from chunking stage.
            output_dir: Directory for inference artifacts.
            gliner: Initialized GLiNER detector.
            inference_labels: Labels for detection.
            cached_label_embeddings: Hoisted label embeddings (bi-encoder).
            config: Pipeline configuration.
            
        Returns:
            InferenceContext with detected entities per document.
        """
        logger.info("=" * 60)
        logger.info("Stage 2/4: Inference (Laptop-Optimized with Autotuning)")
        logger.info("=" * 60)
        
        table = chunking_ctx.table_with_chunks
        posts = chunking_ctx.posts
        post_ids = chunking_ctx.post_ids
        
        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        
        # Check for cached inference results
        inference_results_path = output_dir / "inference_results.arrow"
        if inference_results_path.exists():
            try:
                logger.info(f"Found cached inference results: {inference_results_path}")
                cached_table = feather.read_table(inference_results_path)
                # Reconstruct entities_batch from cached results
                entities_batch = self._load_cached_inference(cached_table, len(posts))
                logger.info("Stage 2 skipped: using cached inference results")
                return InferenceContext(
                    entities_batch=entities_batch,
                    skipped_doc_indices=set(),
                    inference_results_path=inference_results_path,
                    stage_skipped=True,
                )
            except Exception as e:
                logger.warning(f"Failed to load cached inference: {e}; re-running inference")
        
        post_chunked_column = table["post_chunked"]
        
        # Laptop: don't pin memory (may not have dedicated GPU)
        pin_memory = bool(self._cfg_get_optional(
            config, "execution.async_prefetch.pin_memory", False
        ))
        
        # Create RuntimeController from config (autotuning)
        runtime_controller = self._create_runtime_controller(config)
        logger.info(
            f"Autotuning enabled: initial_budget={runtime_controller.current_budget:,}, "
            f"warmup_batches={runtime_controller.config.warmup_batches}"
        )
        
        inference_result = gliner.inference_from_manifest(
            post_chunked_column=post_chunked_column,
            labels=inference_labels,
            batch_size=gliner_batch_size,
            cached_label_embeddings=cached_label_embeddings,
            pin_memory=pin_memory,
            show_progress=True,
            runtime_controller=runtime_controller,
        )
        
        entities_batch = inference_result.entities
        
        # Completeness Guard - Retry skipped documents
        max_retry_rounds = int(self._cfg_get_optional(config, "execution.autotuning.max_retry_rounds", 3))
        retry_round = 0
        all_skipped_docs: Set[int] = set(inference_result.skipped_doc_indices)
        
        while inference_result.skipped_doc_indices and retry_round < max_retry_rounds:
            retry_round += 1
            skipped_count = len(inference_result.skipped_doc_indices)
            logger.warning(
                f"Completeness guard: {skipped_count} documents skipped due to OOM. "
                f"Starting retry round {retry_round}/{max_retry_rounds}"
            )
            
            # Create conservative recovery controller
            oom_ceiling = runtime_controller.oom_ceiling
            recovery_budget = int(oom_ceiling * 0.8)
            recovery_budget = max(recovery_budget, runtime_controller.config.min_token_budget)
            
            recovery_config = RuntimeConfig(
                warmup_batches=2,
                initial_token_budget=recovery_budget,
                min_token_budget=runtime_controller.config.min_token_budget,
                max_token_budget=recovery_budget,
                memory_headroom_mb=runtime_controller.config.memory_headroom_mb * 2.0,
                scale_up_factor=1.03,
                scale_down_factor=0.6,
                oom_slash_factor=0.3,
                stability_threshold=0.25,
                history_window=4,
                recovery_patience=2,
            )
            recovery_controller = RuntimeController(recovery_config)
            
            logger.info(f"Retry controller: recovery_budget={recovery_budget:,}")
            
            # Re-run inference on skipped documents
            skipped_indices = sorted(inference_result.skipped_doc_indices)
            retry_result = gliner.inference_from_manifest(
                post_chunked_column=post_chunked_column.take(pa.array(skipped_indices)),
                labels=inference_labels,
                batch_size=max(1, gliner_batch_size // 4),
                cached_label_embeddings=cached_label_embeddings,
                pin_memory=False,
                show_progress=True,
                runtime_controller=recovery_controller,
            )
            
            # Merge retry results
            for local_idx, doc_idx in enumerate(skipped_indices):
                if local_idx not in retry_result.skipped_doc_indices:
                    entities_batch[doc_idx] = retry_result.entities[local_idx]
            
            new_skipped = {
                skipped_indices[local_idx]
                for local_idx in retry_result.skipped_doc_indices
            }
            
            inference_result = InferenceResult(
                entities=entities_batch,
                skipped_doc_indices=new_skipped,
                oom_count=inference_result.oom_count + retry_result.oom_count,
                final_budget=retry_result.final_budget,
            )
            
            self._cleanup_memory()
            
            if not new_skipped:
                logger.info(f"All documents processed after {retry_round} retry round(s)")
                break
        
        # Final completeness check
        final_skipped = inference_result.skipped_doc_indices
        if final_skipped:
            logger.error(
                f"INCOMPLETE INFERENCE: {len(final_skipped)} documents could not be processed"
            )
            # Save manifest of skipped documents
            skipped_manifest = {
                "skipped_doc_indices": sorted(final_skipped),
                "skipped_post_ids": [post_ids[i] for i in sorted(final_skipped)],
                "total_oom_count": inference_result.oom_count,
                "final_budget": inference_result.final_budget,
            }
            skipped_manifest_path = output_dir / "skipped_documents_manifest.json"
            with open(skipped_manifest_path, "w") as f:
                json.dump(skipped_manifest, f, indent=2)
            logger.warning(f"Saved skipped documents manifest: {skipped_manifest_path}")
        
        # Cache inference results for crash recovery
        self._save_inference_results(
            entities_batch, post_ids, inference_results_path, inference_result
        )
        
        logger.info("Stage 2 complete: inference finished")
        
        # Cleanup after inference stage
        self._cleanup_memory()
        
        return InferenceContext(
            entities_batch=entities_batch,
            skipped_doc_indices=final_skipped,
            inference_results_path=inference_results_path,
            stage_skipped=False,
            oom_count=inference_result.oom_count,
            final_budget=inference_result.final_budget,
            retry_rounds=retry_round,
        )
    
    def _save_inference_results(
        self,
        entities_batch: List[List[Dict[str, Any]]],
        post_ids: List[str],
        output_path: Path,
        inference_result: InferenceResult,
    ) -> None:
        """Save inference results for crash recovery."""
        records = []
        for doc_idx, entities in enumerate(entities_batch):
            for entity in entities:
                records.append({
                    "doc_idx": doc_idx,
                    "post_id": post_ids[doc_idx],
                    "text": entity.get("text", ""),
                    "label": entity.get("label", ""),
                    "start": entity.get("start", 0),
                    "end": entity.get("end", 0),
                    "score": entity.get("score", 0.0),
                })
        
        if records:
            df = pd.DataFrame(records)
            table = pa.Table.from_pandas(df)
            feather.write_feather(table, output_path)
            logger.info(f"Cached {len(records)} inference results: {output_path}")
    
    def _load_cached_inference(
        self,
        cached_table: pa.Table,
        num_docs: int,
    ) -> List[List[Dict[str, Any]]]:
        """Load cached inference results into entities_batch format."""
        entities_batch: List[List[Dict[str, Any]]] = [[] for _ in range(num_docs)]
        
        df = cached_table.to_pandas()
        for _, row in df.iterrows():
            doc_idx = int(row["doc_idx"])
            if 0 <= doc_idx < num_docs:
                entities_batch[doc_idx].append({
                    "text": row["text"],
                    "label": row["label"],
                    "start": int(row["start"]),
                    "end": int(row["end"]),
                    "score": float(row["score"]),
                })
        
        return entities_batch
    
    # =======================================================================
    # Stage 3: LEACE Stage
    # =======================================================================
    def _run_leace_stage(
        self,
        chunking_ctx: ChunkingContext,
        inference_ctx: InferenceContext,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        gliner: GLiNERDetector,
        config: Dict[str, Any],
        demographic_columns: Optional[List[str]] = None,
    ) -> LEACEContext:
        """
        Execute Stage 3: Masking + embedding + LEACE projection computation.
        
        Args:
            chunking_ctx: Context from chunking stage.
            inference_ctx: Context from inference stage.
            output_dataset_path: Path for clean dataset output.
            projection_matrix_path: Path for projection matrix output.
            pollution_logs_path: Path for pollution logs output.
            gliner: Initialized GLiNER detector (for masker).
            config: Pipeline configuration.
            demographic_columns: Optional list of demographic columns for LEACE.
                If provided, only these columns are used. Supports single-label mode.
            
        Returns:
            LEACEContext with projection matrix and masked texts.
        """
        logger.info("=" * 60)
        logger.info("Stage 3/4: LEACE (Batch-Accumulated Projection)")
        logger.info("=" * 60)
        
        table = chunking_ctx.table_with_chunks
        posts = chunking_ctx.posts
        post_ids = chunking_ctx.post_ids
        entities_batch = inference_ctx.entities_batch
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))
        
        # Check for cached projection matrix
        if projection_matrix_path.exists():
            try:
                logger.info(f"Found cached projection matrix: {projection_matrix_path}")
                projection_matrix = torch.load(projection_matrix_path, map_location="cpu")
                
                # Need to reconstruct masked_texts and pollution_logs
                # Load from pollution_logs if available
                if pollution_logs_path.exists():
                    # We still need to regenerate masked_texts for probe embeddings
                    logger.info("Regenerating masked texts from entities...")
                    taxonomy_config, _ = self._load_taxonomy_config(config)
                    masker = SpanMasker.from_taxonomy(
                        taxonomy_cfg=taxonomy_config,
                        tokenizer=gliner.model.data_processor.transformer_tokenizer,
                    )
                    masked_texts, pollution_logs = masker.mask_batch(posts, entities_batch, post_ids)
                    
                    return LEACEContext(
                        projection_matrix=projection_matrix,
                        masked_texts=masked_texts,
                        pollution_logs=pollution_logs,
                        probe_embeddings=None,  # Will be computed in probing stage
                        stage_skipped=True,
                    )
            except Exception as e:
                logger.warning(f"Failed to load cached projection: {e}; re-running LEACE")
        
        # Apply masks
        logger.info("Applying typed masks")
        taxonomy_config, _ = self._load_taxonomy_config(config)
        masker = SpanMasker.from_taxonomy(
            taxonomy_cfg=taxonomy_config,
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
        )
        
        masked_texts, pollution_logs = masker.mask_batch(posts, entities_batch, post_ids)
        total_spans = sum(len(entities) for entities in entities_batch)
        logger.info(f"Masked {total_spans} pollution spans")
        
        # Embedding & LEACE computation
        logger.info("Embedding masked texts and computing LEACE projection")
        
        mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))
        
        # Laptop: embeddings to CPU for LEACE accumulation
        encoder_output_device = str(self._cfg_get_optional(config, "encoder.output_device", "cpu"))
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            output_device=encoder_output_device,
            special_tokens=mask_tokens,
        )
        
        # Use filtered demographic columns if provided, else all columns
        demo_cols_for_leace = demographic_columns if demographic_columns else get_demographic_columns()
        logger.info(f"LEACE using demographic columns: {demo_cols_for_leace}")
        encoder = DemographicEncoder(demo_cols_for_leace).fit(table)
        
        # Force CPU for LEACE in laptop mode
        leace_force_cpu = bool(self._cfg_get(config, "leace.force_cpu"))
        if leace_force_cpu:
            logger.info("LEACE computation forced to CPU for numerical stability")
        
        leace_device_spec = self._cfg_get_optional(config, "leace.device", None)
        if leace_device_spec is None:
            leace_device_spec = self._cfg_get(config, "encoder.device")
        leace_device = "cpu" if leace_force_cpu else self._resolve_device(leace_device_spec)
        leace_compute_dtype = self._cfg_get_optional(config, "leace.compute_dtype", "float64")
        leace_batch_size = int(self._cfg_get_optional(config, "leace.batch_size", 50))
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device=leace_device,
            force_cpu=leace_force_cpu,
            compute_dtype=leace_compute_dtype,
        )
        
        # Laptop mode: smaller shards for embedding + LEACE accumulation
        chunking_cfg = config.get("gliner", {}).get("chunking", {})
        shard_size = int(chunking_cfg.get("shard_size", 0))
        if shard_size <= 0:
            shard_size = leace_batch_size
        if shard_size <= 0:
            shard_size = 512
        
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
                shard_table = table.slice(start, end - start)
                
                embeddings_shard = embedder.embed_texts(
                    shard_masked,
                    batch_size=encoder_batch_size,
                    show_progress=False,
                    output_device=encoder_output_device,
                )
                
                embeddings_shard_cpu = embeddings_shard.detach().cpu().to(dtype=torch.float32)
                
                if probe_enabled and len(probe_embeddings) < max_probe_samples:
                    remaining = max_probe_samples - sum(e.shape[0] for e in probe_embeddings)
                    if remaining > 0:
                        probe_embeddings.append(embeddings_shard_cpu[:remaining].clone())
                
                concepts_shard = torch.from_numpy(encoder.transform(shard_table))
                concept_stats = leace.accumulate_batch_concepts(
                    embeddings_shard_cpu,
                    concepts_shard,
                    concept_stats,
                )
                
                del embeddings_shard, embeddings_shard_cpu
                embed_pbar.update(1)
        finally:
            embed_pbar.close()
        
        self._cleanup_memory()
        
        if concept_stats is None:
            raise RuntimeError("LEACE accumulation failed: no concept statistics computed")
        
        # Compute projection matrix
        logger.info("Computing LEACE projection matrix")
        projection_matrix = leace.compute_projection_from_concept_stats(concept_stats)
        
        # Save outputs
        self._save_cleaned_dataset(table, masked_texts, output_dataset_path)
        torch.save(projection_matrix.cpu(), projection_matrix_path)
        logger.info(f"Saved projection matrix: {projection_matrix_path}")
        self._save_pollution_logs(pollution_logs, pollution_logs_path)
        
        logger.info("Stage 3 complete: LEACE projection computed")
        
        # Cleanup after LEACE stage
        self._cleanup_memory()
        
        return LEACEContext(
            projection_matrix=projection_matrix,
            masked_texts=masked_texts,
            pollution_logs=pollution_logs,
            probe_embeddings=torch.cat(probe_embeddings, dim=0) if probe_embeddings else None,
            total_spans=total_spans,
            stage_skipped=False,
        )
    
    # =======================================================================
    # Stage 4: Probing Stage
    # =======================================================================
    def _run_probing_stage(
        self,
        chunking_ctx: ChunkingContext,
        inference_ctx: InferenceContext,
        leace_ctx: LEACEContext,
        gliner: GLiNERDetector,
        config: Dict[str, Any],
        output_dir: Path,
        demographic_columns: Optional[List[str]] = None,
    ) -> ProbingContext:
        """
        Execute Stage 4: Amnesic drop metrics and visualizations.
        
        Args:
            chunking_ctx: Context from chunking stage.
            inference_ctx: Context from inference stage.
            leace_ctx: Context from LEACE stage.
            gliner: Initialized GLiNER detector.
            config: Pipeline configuration.
            output_dir: Directory for reports.
            demographic_columns: Optional list of demographic columns for probing.
                If provided, only these columns are probed. Supports single-label mode.
            
        Returns:
            ProbingContext with metrics by column.
        """
        logger.info("=" * 60)
        logger.info("Stage 4/4: Probing (Amnesic Drop Metrics)")
        logger.info("=" * 60)
        
        table = chunking_ctx.table_with_chunks
        posts = chunking_ctx.posts
        entities_batch = inference_ctx.entities_batch
        projection_matrix = leace_ctx.projection_matrix
        
        probe_enabled = bool(self._cfg_get(config, "probe.compute_amnesic_drop"))
        if not probe_enabled:
            logger.info("Probing disabled in config; skipping Stage 4")
            return ProbingContext(
                by_column={},
                min_amnesic_drop=0.0,
                mean_amnesic_drop=0.0,
                reports_dir=None,
                stage_skipped=True,
            )
        
        # Get or compute probe embeddings
        probe_embeddings = leace_ctx.probe_embeddings
        max_probe_samples = int(self._cfg_get(config, "probe.max_samples"))
        
        if probe_embeddings is None or probe_embeddings.shape[0] == 0:
            logger.info("Computing probe embeddings from scratch")
            # Re-embed for probing
            mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))
            encoder_output_device = str(self._cfg_get_optional(config, "encoder.output_device", "cpu"))
            encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))
            
            embedder = FrozenEmbedder(
                model_name=str(self._cfg_get(config, "encoder.model")),
                device=self._resolve_device(self._cfg_get(config, "encoder.device")),
                max_length=int(self._cfg_get(config, "encoder.max_length")),
                output_device=encoder_output_device,
                special_tokens=mask_tokens,
            )
            
            masked_subset = leace_ctx.masked_texts[:max_probe_samples]
            probe_embeddings = embedder.embed_texts(
                masked_subset,
                batch_size=encoder_batch_size,
                show_progress=True,
                output_device="cpu",
            )
        
        embeddings_before = probe_embeddings[:max_probe_samples].to(dtype=torch.float32)
        probe_table = table.slice(0, len(embeddings_before))
        
        P_cpu = projection_matrix.detach().to("cpu", dtype=torch.float32)
        embeddings_after = embeddings_before @ P_cpu.T
        
        X_before_np = embeddings_before.numpy()
        X_after_np = embeddings_after.numpy()
        
        # Build ProbeConfig from YAML
        probe_cfg = config.get("probe", {})
        probe_config = ProbeConfig(
            backend=probe_cfg.get("backend", "sklearn"),
            max_iter=int(probe_cfg.get("max_iter", 1000)),
            random_state=int(self._cfg_get(config, "seed")),
            n_folds=int(probe_cfg.get("n_folds", 5)),
            pvalue_threshold=float(probe_cfg.get("pvalue_threshold", 0.05)),
            use_class_weights=bool(probe_cfg.get("use_class_weights", True)),
        )
        use_kfold = bool(probe_cfg.get("use_kfold", True))
        
        by_column: Dict[str, Any] = {}
        by_column_extended: Dict[str, Any] = {}
        drops: List[float] = []
        separability_before: Dict[str, Any] = {}
        separability_after: Dict[str, Any] = {}
        class_imbalance: Dict[str, Any] = {}
        
        # Use filtered demographic columns if provided, else all columns
        demo_cols = demographic_columns if demographic_columns else get_demographic_columns()
        logger.info(f"Probing demographic columns: {demo_cols}")
        
        for col in demo_cols:
            labels_np = extract_probe_labels(probe_table, col)
            labels_t = torch.tensor(labels_np, dtype=torch.long)
            
            try:
                result = compute_amnesic_drop_extended(
                    embeddings_before,
                    embeddings_after,
                    labels_t,
                    train_split=float(self._cfg_get(config, "probe.train_split")),
                    random_state=int(self._cfg_get(config, "seed")),
                    config=probe_config,
                    use_kfold=use_kfold,
                    device="cpu",
                )
                
                by_column[col] = {
                    "accuracy_before": result.acc_before,
                    "accuracy_after": result.acc_after,
                    "amnesic_drop": result.amnesic_drop,
                }
                by_column_extended[col] = result.to_dict()
                drops.append(float(result.amnesic_drop))
                
                separability_before[col] = compute_embedding_separability(
                    X_before_np, labels_np, sample_size=2000
                )
                separability_after[col] = compute_embedding_separability(
                    X_after_np, labels_np, sample_size=2000
                )
                class_imbalance[col] = compute_class_imbalance_metrics(labels_np)
                
            except ValueError as exc:
                logger.warning(f"Probe skipped for column {col}: {exc}")
                by_column[col] = {
                    "accuracy_before": 0.0,
                    "accuracy_after": 0.0,
                    "amnesic_drop": 0.0,
                }
                by_column_extended[col] = {"error": str(exc)}
                drops.append(0.0)
        
        min_drop = float(min(drops)) if drops else 0.0
        mean_drop = float(np.mean(drops)) if drops else 0.0
        
        logger.info(f"Probe metrics complete: min_drop={min_drop:.2%}, mean_drop={mean_drop:.2%}")
        
        # Create reports directory
        reports_dir = output_dir / "reports"
        reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Save probe results as JSON
        probe_results = {
            "by_column": by_column,
            "by_column_extended": by_column_extended,
            "min_amnesic_drop": min_drop,
            "mean_amnesic_drop": mean_drop,
            "separability_before": separability_before,
            "separability_after": separability_after,
            "class_imbalance": class_imbalance,
            "config": {
                "backend": probe_config.backend,
                "use_kfold": use_kfold,
                "n_folds": probe_config.n_folds,
                "max_samples": len(embeddings_before),
            },
        }
        
        probe_results_path = reports_dir / "probe_results.json"
        with open(probe_results_path, "w") as f:
            json.dump(_convert_numpy_types(probe_results), f, indent=2)
        logger.info(f"Saved probe results: {probe_results_path}")
        
        logger.info("Stage 4 complete: probing finished")
        
        # Build visualization data for pipeline orchestration
        labels_dict = {
            col: extract_probe_labels(probe_table, col)
            for col in demo_cols
        }
        visualization_data = {
            "by_column_extended": by_column_extended,
            "separability_before": separability_before,
            "separability_after": separability_after,
            "benchmark_results": {},  # Laptop mode doesn't do benchmark by default
            "control_probe_results": {},
            "demo_cols": demo_cols,
            "X_before_np": X_before_np,
            "X_after_np": X_after_np,
            "labels_dict": labels_dict,
        }
        
        # Cleanup after probing stage
        self._cleanup_memory()
        
        return ProbingContext(
            by_column=by_column,
            by_column_extended=by_column_extended,
            min_amnesic_drop=min_drop,
            mean_amnesic_drop=mean_drop,
            separability_before=separability_before,
            separability_after=separability_after,
            class_imbalance=class_imbalance,
            reports_dir=reports_dir,
            stage_skipped=False,
            visualization_data=visualization_data,
            demo_cols=demo_cols,
            probe_config=probe_config,
        )
    
    # =======================================================================
    # Main Execute Method (Orchestrator)
    # =======================================================================
    def execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
        use_only_labels: Optional[List[str]] = None,
    ) -> Dict[str, Any]:
        """
        Execute Phase A with staged execution architecture (laptop-optimized).
        
        Four-Stage Pipeline:
        1. Chunking: Parallel chunking → persist post_chunked (skip if exists).
        2. Inference: Manifest-based inference with retry rounds.
        3. LEACE: Masking + embedding + batch-accumulated projection.
        4. Probing: Amnesic drop metrics and visualizations.
        
        Skip Flags (from config.execution.skip_stages):
        - skip_chunking: Start at Stage 2 (requires post_chunked populated).
        - skip_inference: Start at Stage 3 (requires inference_results.arrow).
        - skip_leace: Start at Stage 4 (requires projection_matrix.pt).
        - skip_probing: Exit after Stage 3 (skip metrics/visualizations).
        
        Args:
            use_only_labels: Optional list of demographic labels to filter to.
        """
        logger.info("=" * 80)
        logger.info("Phase A: Laptop Strategy (Four-Stage Execution)")
        logger.info("=" * 80)

        # Parse skip flags from config
        skip_cfg = SkipStagesConfig.from_config(config)
        if any([skip_cfg.skip_chunking, skip_cfg.skip_inference, skip_cfg.skip_leace, skip_cfg.skip_probing]):
            logger.info(
                f"Skip flags: chunking={skip_cfg.skip_chunking}, inference={skip_cfg.skip_inference}, "
                f"leace={skip_cfg.skip_leace}, probing={skip_cfg.skip_probing}"
            )

        # Resolve device (laptop gracefully handles missing CUDA)
        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))

        self._last_device = device
        self._last_batch_size = gliner_batch_size
        
        logger.info(f"Device: {device}, GLiNER batch: {gliner_batch_size}, Encoder batch: {encoder_batch_size}")
        
        # =======================================================================
        # Load Dataset & Initialize Components
        # =======================================================================
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(
            arrow_path=input_dataset_path,
            seed=int(self._cfg_get(config, "seed")),
        )
        table = dataset.table
        
        # Cast string columns to large_string early to prevent offset overflow
        # during subsequent take/filter operations
        table = _cast_strings_to_large(table)

        # Optional subset (strictly config-driven, recommended for laptop)
        if bool(self._cfg_get(config, "subset.enabled")):
            size = self._cfg_get(config, "subset.size")
            if size is not None and len(table) > int(size):
                rng = np.random.default_rng(int(self._cfg_get(config, "seed")))
                indices = rng.choice(len(table), size=int(size), replace=False)
                logger.info(f"Using subset (seeded): {int(size)}/{len(table)} samples")
                table = _safe_take_subset(table, indices)
        
        # =======================================================================
        # Apply --use-only Label Filter (if specified)
        # =======================================================================
        if use_only_labels:
            logger.info(f"Applying --use-only filter for labels: {use_only_labels}")
            original_count = len(table)
            
            # Filter rows where ANY specified label has a valid (non-null) value
            # This is OR semantics: row is included if it has at least one label
            mask = None
            for col in use_only_labels:
                if col not in table.column_names:
                    logger.warning(f"Column '{col}' not found in dataset; skipping filter for it")
                    continue
                col_mask = pc.is_valid(table[col])
                mask = col_mask if mask is None else pc.or_(mask, col_mask)
            
            if mask is not None:
                table = table.filter(mask)
                logger.info(f"Filtered dataset: {original_count} -> {len(table)} rows "
                           f"(kept rows with valid {use_only_labels})")
            else:
                logger.warning("No valid columns found for --use-only filter; using full dataset")
        
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        output_dir = output_dataset_path.parent
        
        logger.info(f"Loaded {len(posts)} posts")
        
        # Initialize GLiNER Detector
        logger.info("Initializing GLiNER detector and components")
        taxonomy_config, constraints_config = self._load_taxonomy_config(config)
        
        # Apply --use-only label filter to taxonomy (if specified)
        if use_only_labels:
            taxonomy_config = self._filter_taxonomy_config(taxonomy_config, use_only_labels)
            logger.info(f"Filtered taxonomy to columns: {use_only_labels}")
        
        gliner_cfg = config.get("gliner", {})
        batch_inference_cfg = gliner_cfg.get("batch_inference", {})
        
        batch_config = BatchInferenceConfig(
            enable_batching=batch_inference_cfg.get("enable_batching", True),
            batch_size=gliner_batch_size,
            num_buckets=batch_inference_cfg.get("num_buckets"),
            min_bucket_size=batch_inference_cfg.get("min_bucket_size", 2),
            enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
            strict_padding=batch_inference_cfg.get("strict_padding", False),
            seq_len_buckets=batch_inference_cfg.get("seq_len_buckets"),
        )
        
        chunking_cfg = gliner_cfg.get("chunking", {})
        budget_config = BudgetConfig(
            model_max_length=int(self._cfg_get(config, "encoder.max_length")),
            mode=chunking_cfg.get("mode", "single_sentence"),
            legacy_sequential_mode=False,
            parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 4)),
            parallel_chunking_min_texts=int(chunking_cfg.get("parallel_chunking_min_texts", 128)),
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
        
        # Hoist label embeddings to strategy level
        inference_labels = gliner.taxonomy.get_inference_labels()
        cached_label_embeddings = gliner.get_cached_label_embeddings(inference_labels)
        if cached_label_embeddings is not None:
            logger.info(f"Hoisted label embeddings for {len(inference_labels)} labels")
        
        # =======================================================================
        # Stage 1: Chunking
        # =======================================================================
        if skip_cfg.skip_chunking:
            logger.info("Skipping Stage 1 (chunking) per skip flags")
            # Validate prerequisite: post_chunked must exist
            if not is_post_chunked_fully_populated(table):
                if not skip_cfg.force_skip:
                    raise RuntimeError(
                        "Skip chunking requested but post_chunked column is not fully populated. "
                        "Run without --skip-chunking first."
                    )
                else:
                    logger.warning("Skip chunking requested but post_chunked not fully populated; continuing due to force_skip=True")
            chunking_ctx = ChunkingContext(
                table_with_chunks=table,
                posts=posts,
                post_ids=post_ids,
                stage_skipped=True,
            )
        else:
            chunking_ctx = self._run_chunking_stage(
                input_dataset_path=input_dataset_path,
                table=table,
                posts=posts,
                post_ids=post_ids,
                gliner=gliner,
                inference_labels=inference_labels,
                config=config,
            )
        
        # =======================================================================
        # Stage 2: Inference
        # =======================================================================
        if skip_cfg.skip_inference:
            logger.info("Skipping Stage 2 (inference) per skip flags")
            # Validate prerequisite: inference_results.arrow must exist
            inference_results_path = output_dir / "inference_results.arrow"
            if not inference_results_path.exists():
                if not skip_cfg.force_skip:
                    raise RuntimeError(
                        f"Skip inference requested but {inference_results_path} not found. "
                        "Run without --skip-inference first."
                    )
                else:
                    logger.warning("Skip inference requested but inference_results.arrow not found; continuing due to force_skip=True")
                    entities_batch = [[] for _ in chunking_ctx.posts]
                    inference_ctx = InferenceContext(
                        entities_batch=entities_batch,
                        skipped_doc_indices=set(range(len(chunking_ctx.posts))),
                        inference_results_path=None,
                        stage_skipped=True,
                    )
            else:
                cached_table = feather.read_table(inference_results_path)
                entities_batch = self._load_cached_inference(cached_table, len(chunking_ctx.posts))
                inference_ctx = InferenceContext(
                    entities_batch=entities_batch,
                    skipped_doc_indices=set(),
                    inference_results_path=inference_results_path,
                    stage_skipped=True,
                )
        else:
            inference_ctx = self._run_inference_stage(
                chunking_ctx=chunking_ctx,
                output_dir=output_dir,
                gliner=gliner,
                inference_labels=inference_labels,
                cached_label_embeddings=cached_label_embeddings,
                config=config,
            )
        
        # =======================================================================
        # Stage 3: LEACE
        # =======================================================================
        if skip_cfg.skip_leace:
            logger.info("Skipping Stage 3 (LEACE) per skip flags")
            # Validate prerequisite: projection_matrix.pt must exist
            if not projection_matrix_path.exists():
                if not skip_cfg.force_skip:
                    raise RuntimeError(
                        f"Skip LEACE requested but {projection_matrix_path} not found. "
                        "Run without --skip-leace first."
                    )
                else:
                    logger.warning("Skip LEACE requested but projection_matrix.pt not found; continuing due to force_skip=True")
                    projection_matrix = None
            else:
                projection_matrix = torch.load(projection_matrix_path, map_location="cpu")
            # Regenerate masked_texts and pollution_logs (needed for probing)
            masker = SpanMasker.from_taxonomy(
                taxonomy_cfg=taxonomy_config,
                tokenizer=gliner.model.data_processor.transformer_tokenizer,
            )
            masked_texts, pollution_logs = masker.mask_batch(
                chunking_ctx.posts, inference_ctx.entities_batch, chunking_ctx.post_ids
            )
            leace_ctx = LEACEContext(
                projection_matrix=projection_matrix,
                masked_texts=masked_texts,
                pollution_logs=pollution_logs,
                probe_embeddings=None,
                stage_skipped=True,
            )
        else:
            # Derive demographic columns from use_only_labels if provided
            demographic_columns = use_only_labels if use_only_labels else None
            
            leace_ctx = self._run_leace_stage(
                chunking_ctx=chunking_ctx,
                inference_ctx=inference_ctx,
                output_dataset_path=output_dataset_path,
                projection_matrix_path=projection_matrix_path,
                pollution_logs_path=pollution_logs_path,
                gliner=gliner,
                config=config,
                demographic_columns=demographic_columns,
            )
        
        # =======================================================================
        # Stage 4: Probing (skip if skip_probing is set)
        # =======================================================================
        # Derive demographic columns for probing (use same as LEACE stage)
        demographic_columns = use_only_labels if use_only_labels else None
        
        if skip_cfg.skip_probing:
            logger.info("Skipping Stage 4 (probing) per skip flags - early exit")
            probing_ctx = ProbingContext(
                by_column={},
                min_amnesic_drop=0.0,
                mean_amnesic_drop=0.0,
                reports_dir=None,
                stage_skipped=True,
            )
        else:
            probing_ctx = self._run_probing_stage(
                chunking_ctx=chunking_ctx,
                inference_ctx=inference_ctx,
                leace_ctx=leace_ctx,
                gliner=gliner,
                config=config,
                output_dir=output_dir,
                demographic_columns=demographic_columns,
            )
        
        # =======================================================================
        # Build Metadata & Return
        # =======================================================================
        total_spans = leace_ctx.total_spans if hasattr(leace_ctx, 'total_spans') and leace_ctx.total_spans else \
            sum(len(e) for e in inference_ctx.entities_batch)
        
        metadata: Dict[str, Any] = {
            "num_samples": len(chunking_ctx.posts),
            "num_pollution_spans": total_spans,
            "projection_matrix_shape": list(leace_ctx.projection_matrix.shape),
            "device": device,
            "label_filter": use_only_labels,  # None for multi-label mode, list for single-label
            "staged_execution": True,
            "stages_skipped": {
                "chunking": chunking_ctx.stage_skipped,
                "inference": inference_ctx.stage_skipped,
                "leace": leace_ctx.stage_skipped,
                "probing": probing_ctx.stage_skipped,
            },
        }
        
        # Add inference completeness info
        if hasattr(inference_ctx, 'oom_count'):
            metadata["inference_completeness"] = {
                "total_docs": len(chunking_ctx.posts),
                "processed_docs": len(chunking_ctx.posts) - len(inference_ctx.skipped_doc_indices),
                "skipped_docs": len(inference_ctx.skipped_doc_indices),
                "oom_count": inference_ctx.oom_count,
                "final_budget": inference_ctx.final_budget,
                "retry_rounds": inference_ctx.retry_rounds if hasattr(inference_ctx, 'retry_rounds') else 0,
            }
        
        # Explicit recall
        if bool(self._cfg_get(config, "gliner.compute_explicit_recall")):
            recall = compute_explicit_recall(
                chunking_ctx.posts,
                inference_ctx.entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall
        
        # Add probe metrics if probing was run
        if not probing_ctx.stage_skipped and probing_ctx.by_column:
            metadata["probe"] = {
                "by_column": probing_ctx.by_column,
                "by_column_extended": probing_ctx.by_column_extended,
                "min_amnesic_drop": probing_ctx.min_amnesic_drop,
                "mean_amnesic_drop": probing_ctx.mean_amnesic_drop,
                "threshold": float(self._cfg_get(config, "probe.amnesic_drop_threshold")),
                "max_samples": int(self._cfg_get(config, "probe.max_samples")),
            }
            
            metadata["separability"] = {
                "before": probing_ctx.separability_before,
                "after": probing_ctx.separability_after,
            }
            metadata["class_imbalance"] = probing_ctx.class_imbalance
            
            # Include visualization data for pipeline orchestration
            if probing_ctx.visualization_data:
                metadata["visualization_data"] = probing_ctx.visualization_data
            if probing_ctx.demo_cols:
                metadata["demo_cols"] = probing_ctx.demo_cols
            if probing_ctx.probe_config:
                metadata["probe_config"] = probing_ctx.probe_config
            
            # Enforce threshold if configured
            if bool(self._cfg_get(config, "quality.enforce_thresholds")):
                threshold = float(self._cfg_get(config, "probe.amnesic_drop_threshold"))
                if probing_ctx.min_amnesic_drop < threshold:
                    raise ValueError(
                        f"Amnesic drop gate failed: min={probing_ctx.min_amnesic_drop:.3f} < threshold={threshold:.3f}"
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

    def _filter_taxonomy_config(
        self, taxonomy_config: Dict[str, Any], use_only_labels: List[str]
    ) -> Dict[str, Any]:
        """
        Filter taxonomy config to only include specified demographic columns.
        
        This is the key integration point for --use-only functionality.
        When filtering is applied, only prompts for the specified columns
        will be used in GLiNER inference.
        
        Args:
            taxonomy_config: Full taxonomy configuration dict.
            use_only_labels: List of demographic columns to keep.
            
        Returns:
            Filtered taxonomy config with only specified columns.
        """
        filtered = dict(taxonomy_config)  # Shallow copy
        
        # Filter column_prompts to only include specified columns
        if "column_prompts" in filtered:
            original_columns = set(filtered["column_prompts"].keys())
            filtered["column_prompts"] = {
                col: prompts
                for col, prompts in filtered["column_prompts"].items()
                if col in use_only_labels
            }
            filtered_columns = set(filtered["column_prompts"].keys())
            removed_columns = original_columns - filtered_columns
            
            logger.info(
                f"Taxonomy filtering: kept {len(filtered_columns)} columns, "
                f"removed {len(removed_columns)}: {sorted(removed_columns)}"
            )
        
        # Filter mask_tokens to only include specified columns
        if "mask_tokens" in filtered:
            filtered["mask_tokens"] = {
                col: token
                for col, token in filtered["mask_tokens"].items()
                if col in use_only_labels
            }
        
        return filtered
