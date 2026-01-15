"""
HPC Strategy for Phase A Pollution Filtering (Staged Execution Architecture).

Implements four-stage decoupled execution for maximum throughput:
- Stage 1 (Chunking): Parallel chunking with streaming writes to post_chunked.
- Stage 2 (Inference): Global-sorted GLiNER detection with async storage.
- Stage 3 (LEACE): Masking, embedding, and projection matrix computation.
- Stage 4 (Probing): Amnesic drop metrics and visualizations.

Key Optimizations:
- Chunks persisted to Arrow column enable crash recovery.
- Global argsort by token_count minimizes padding waste.
- Label embeddings hoisted to strategy level for bi-encoder efficiency.
- pin_memory + non_blocking for overlapped PCIe transfers.
- RuntimeController autotuning for adaptive batch sizing (OOM resilience).

Skip Flags:
- --skip-chunking: Start at inference (assumes post_chunked populated)
- --skip-inference: Start at LEACE (assumes inference_results.arrow exists)
- --skip-leace: Start at probing (assumes projection_matrix.pt exists)
- --skip-probing: Exit after LEACE (skip metrics/visualizations)

Reference: Technical Reports on Staged Execution Architecture.
Implements: phaseA-D_implementation_plan.md Section 7.4 (HPC Mode)
"""

from __future__ import annotations

import gc
import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.compute as pc
import pyarrow.feather as feather
import torch
from tqdm import tqdm


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
from ..semantic_chunker import BudgetConfig, deduplicate_entities
from ..explicit_recall import compute_explicit_recall
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer
from ..probe import (
    compute_amnesic_drop,
    compute_amnesic_drop_extended,
    benchmark_solver_convergence,
    ProbeConfig,
    ProbeBackend,
    create_probe,
)
from ..concept_encoding import DemographicEncoder, extract_probe_labels
from ..async_result_storer import AsyncResultStorer, AsyncStorerConfig
from ..global_sort import flatten_chunks, gather_results, DynamicBatchIterator
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
    compute_control_probe_metrics,
    compute_per_column_amnesic_drop,
)

logger = logging.getLogger(__name__)


class HPCFilterStrategy(PollutionFilterStrategy):
    """
    HPC-optimized pollution filtering strategy with staged execution.
    
    Architecture (Four-Stage Execution):
    - Stage 1 (Chunking): CPU-saturated parallel chunking → post_chunked column.
    - Stage 2 (Inference): GPU-saturated global-sorted inference → entity spans.
    - Stage 3 (LEACE): Masking + embedding + projection matrix computation.
    - Stage 4 (Probing): Amnesic drop metrics + visualizations.
    
    Key Features:
    - Full GPU utilization (A100/H100) via global length sorting.
    - Crash recovery: stages can be skipped if artifacts exist.
    - Bi-encoder label embedding caching hoisted to strategy level.
    - BF16 mixed precision (if available).
    
    Skip Flags (entry-point / early-exit):
    - skip_chunking: Start at inference (post_chunked must exist).
    - skip_inference: Start at LEACE (inference_results.arrow must exist).
    - skip_leace: Start at probing (projection_matrix.pt must exist).
    - skip_probing: Exit after LEACE (skip metrics/visualizations).
    
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
    
    def _cleanup_memory(self) -> None:
        """
        Force garbage collection and clear CUDA cache.
        
        Should be called after each stage to ensure memory is freed
        before the next stage begins. Synchronizes CUDA for accurate timing.
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
        Falls back to HPC-optimized defaults if not specified.
        
        Args:
            config: Pipeline configuration dict.
            
        Returns:
            RuntimeController configured for HPC inference.
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
        
        # HPC defaults (optimized for A100/H100)
        runtime_config = RuntimeConfig(
            warmup_batches=int(autotuning_cfg.get("warmup_batches", 10)),
            initial_token_budget=int(autotuning_cfg.get("initial_token_budget", 32768)),
            min_token_budget=int(autotuning_cfg.get("min_token_budget", 4096)),
            max_token_budget=int(autotuning_cfg.get("max_token_budget", 524288)),
            memory_headroom_mb=float(autotuning_cfg.get("memory_headroom_mb", 2048)),
            scale_up_factor=float(autotuning_cfg.get("scale_up_factor", 1.25)),
            scale_down_factor=float(autotuning_cfg.get("scale_down_factor", 0.85)),
            oom_slash_factor=float(autotuning_cfg.get("oom_slash_factor", 0.5)),
            stability_threshold=float(autotuning_cfg.get("stability_threshold", 0.1)),
            history_window=int(autotuning_cfg.get("history_window", 10)),
            recovery_patience=int(autotuning_cfg.get("recovery_patience", 5)),
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
        logger.info("Stage 1/4: Chunking (CPU-Saturated Parallel)")
        logger.info("=" * 60)
        
        chunking_cfg = config.get("gliner", {}).get("chunking", {})
        
        # Check for resume: if post_chunked exists and is fully populated, skip
        invalid_chunk_rows: List[int] = []
        
        # Check if inference is already complete to skip Stage 1 validation
        inference_ipc_path = input_dataset_path.parent / "inference_results.arrow"
        if has_post_chunked_column(table) and inference_ipc_path.exists():
            try:
                logger.info("Checking async storage to potentially skip Stage 1 validation...")
                temp_flattened = flatten_chunks(
                    table["post_chunked"],
                    extract_texts=False,
                    include_chunk_metadata=False,
                )
                temp_storer = AsyncResultStorer(
                    output_path=inference_ipc_path,
                    num_chunks=temp_flattened.num_chunks,
                )
                if temp_storer.is_complete:
                    logger.info("Async storage complete: skipping Stage 1 chunk validation")
                    return ChunkingContext(
                        table_with_chunks=table,
                        posts=posts,
                        post_ids=post_ids,
                        stage_skipped=True,
                    )
            except Exception as e:
                logger.warning(f"Failed to check async storage for skip: {e}")
        
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
        budget_config = gliner.chunker.config
        
        num_workers = int(chunking_cfg.get("parallel_chunking_workers", 16))
        micro_batch_size = int(chunking_cfg.get("micro_batch_size", 1000))
        
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
        
        # Cleanup between stages
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
        Execute Stage 2: Global-sorted GLiNER inference with async storage.
        
        Args:
            chunking_ctx: Context from chunking stage.
            output_dir: Directory for inference_results.arrow.
            gliner: Initialized GLiNER detector.
            inference_labels: Labels for detection.
            cached_label_embeddings: Hoisted label embeddings (bi-encoder).
            config: Pipeline configuration.
            
        Returns:
            InferenceContext with detected entities per document.
        """
        logger.info("=" * 60)
        logger.info("Stage 2/4: Inference (GPU-Saturated Global-Sorted)")
        logger.info("=" * 60)
        
        table = chunking_ctx.table_with_chunks
        posts = chunking_ctx.posts
        post_ids = chunking_ctx.post_ids
        
        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        
        # Flatten chunks for lineage tracking
        logger.info("Flattening post_chunked for lineage tracking")
        post_chunked_column = table["post_chunked"]
        flattened = flatten_chunks(
            post_chunked_column,
            extract_texts=True,
            include_chunk_metadata=False,
        )
        logger.info(f"Flattened {flattened.num_docs} documents into {flattened.num_chunks} chunks")
        
        # Configure async storage path
        inference_ipc_path = output_dir / "inference_results.arrow"
        
        # Initialize async storer with crash recovery
        async_storage_cfg = config.get("execution", {}).get("async_storage", {})
        storer_config = AsyncStorerConfig(
            queue_size=int(async_storage_cfg.get("queue_size", 16)),
            flush_every_n=int(async_storage_cfg.get("flush_every_n", 10)),
            timeout_seconds=float(async_storage_cfg.get("timeout_seconds", 0.5)),
        )
        
        async_storer = AsyncResultStorer(
            output_path=inference_ipc_path,
            num_chunks=flattened.num_chunks,
            config=storer_config,
        )
        
        # Check crash recovery state
        completed_chunk_ids = async_storer.get_completed_chunk_ids()
        pending_chunk_ids = async_storer.get_pending_chunk_ids()
        logger.info(f"Crash recovery: {len(completed_chunk_ids)} completed, {len(pending_chunk_ids)} pending")
        
        # Track inference statistics
        inference_oom_count = 0
        inference_final_budget = gliner_batch_size
        skipped_chunk_indices: Set[int] = set()
        stage_skipped = False
        
        if async_storer.is_complete:
            logger.info("All chunks already processed - skipping inference")
            stage_skipped = True
        else:
            # Configure pin_memory for HPC
            pin_memory = bool(self._cfg_get_optional(config, "execution.async_prefetch.pin_memory", True))
            
            # Create RuntimeController for autotuning
            runtime_controller = self._create_runtime_controller(config)
            logger.info(f"Autotuning: initial_budget={runtime_controller.current_budget:,}")
            
            # Get prompt embeddings
            prompt_embeddings = cached_label_embeddings
            if prompt_embeddings is None:
                prompt_embeddings = gliner.get_cached_label_embeddings(inference_labels)
            
            # Compute global sort permutation
            logger.info("Computing global sort permutation by token_count")
            sort_indices = flattened.compute_sort_indices()
            
            # Create dynamic batch iterator
            batch_iterator = DynamicBatchIterator(
                flattened=flattened,
                controller=runtime_controller,
                min_batch_size=1,
                max_batch_size=gliner_batch_size * 4,
            )
            
            # Import OOM protection
            from ...hardware_ops.oom_guard import execute_with_oom_protection, OOMRecoveryError
            from ...hardware_ops.runtime import RuntimeMetrics
            from ...hardware_ops.telemetry import CUDATimer
            
            # Start async storer
            async_storer.start()
            
            pbar = tqdm(
                total=len(pending_chunk_ids),
                desc="GLiNER inference (async)",
                unit="chunk",
                dynamic_ncols=True,
            )
            
            try:
                for batch_indices in batch_iterator:
                    pending_in_batch = [idx for idx in batch_indices if idx not in completed_chunk_ids]
                    if not pending_in_batch:
                        continue
                    
                    batch_texts = [flattened.texts[i] for i in pending_in_batch]
                    batch_tokens = sum(int(flattened.token_counts[i]) for i in pending_in_batch)
                    
                    with CUDATimer() as timer:
                        try:
                            batch_entities = execute_with_oom_protection(
                                lambda bt=batch_texts: gliner._detect_batch(bt, inference_labels, prompt_embeddings),
                                controller=runtime_controller,
                                retry_limit=3,
                            )
                        except OOMRecoveryError:
                            logger.error(f"OOM recovery failed for {len(pending_in_batch)} chunks")
                            skipped_chunk_indices.update(pending_in_batch)
                            pbar.update(len(pending_in_batch))
                            inference_oom_count += 1
                            continue
                    
                    # Report metrics
                    memory_mb = torch.cuda.memory_allocated() / (1024 ** 2) if torch.cuda.is_available() else 0.0
                    runtime_controller.report_metrics(RuntimeMetrics(
                        tokens_processed=batch_tokens,
                        batch_time_ms=timer.elapsed_ms,
                        memory_used_mb=memory_mb,
                        batch_size=len(pending_in_batch),
                    ))
                    
                    async_storer.submit(pending_in_batch, batch_entities)
                    pbar.update(len(pending_in_batch))
            finally:
                pbar.close()
                async_storer.finish()
                
                storer_stats = async_storer.get_stats()
                snapshot = runtime_controller.get_snapshot()
                inference_oom_count += snapshot.oom_count
                inference_final_budget = snapshot.current_budget
                
                logger.info(f"Async storer: written={storer_stats.chunks_written}, recovered={storer_stats.chunks_recovered}")
                logger.info(f"Autotuning: state={snapshot.state}, final_budget={snapshot.current_budget:,}")
        
        # Load results and reconstruct document-level entities
        logger.info("Loading results and reconstructing per-document entities")
        flat_results = async_storer.load_results_as_flat_list(flattened.num_chunks)
        
        chunk_starts = flattened.chunk_starts
        if chunk_starts is None:
            chunk_starts = np.zeros(flattened.num_chunks, dtype=np.int32)
        
        entities_batch: List[List[Dict[str, Any]]] = []
        from ..semantic_chunker import deduplicate_entities
        
        for doc_idx in range(flattened.num_docs):
            doc_start = flattened.doc_offsets[doc_idx]
            doc_end = flattened.doc_offsets[doc_idx + 1]
            
            projected_entities: List[Dict[str, Any]] = []
            for flat_idx in range(doc_start, doc_end):
                chunk_entities = flat_results[flat_idx]
                chunk_offset = int(chunk_starts[flat_idx])
                for entity in chunk_entities:
                    projected = dict(entity)
                    projected["start"] = projected.get("start", 0) + chunk_offset
                    projected["end"] = projected.get("end", 0) + chunk_offset
                    projected_entities.append(projected)
            
            deduped = deduplicate_entities(projected_entities)
            entities_batch.append(deduped)
        
        # Compute skipped document indices
        skipped_doc_indices: Set[int] = set()
        for flat_idx in skipped_chunk_indices:
            for doc_idx in range(flattened.num_docs):
                doc_start = flattened.doc_offsets[doc_idx]
                doc_end = flattened.doc_offsets[doc_idx + 1]
                if doc_start <= flat_idx < doc_end:
                    skipped_doc_indices.add(doc_idx)
                    break
        
        if skipped_doc_indices:
            logger.error(f"INCOMPLETE: {len(skipped_doc_indices)} documents had skipped chunks")
            skipped_manifest = {
                "skipped_doc_indices": sorted(skipped_doc_indices),
                "skipped_post_ids": [post_ids[i] for i in sorted(skipped_doc_indices)],
                "skipped_chunk_count": len(skipped_chunk_indices),
                "total_oom_count": inference_oom_count,
                "final_budget": inference_final_budget,
            }
            with open(output_dir / "skipped_documents_manifest.json", "w") as f:
                json.dump(skipped_manifest, f, indent=2)
        
        logger.info(f"Stage 2 complete: {flattened.num_docs} documents reconstructed")
        
        # Cleanup after inference stage
        self._cleanup_memory()
        
        return InferenceContext(
            entities_batch=entities_batch,
            skipped_doc_indices=skipped_doc_indices,
            oom_count=inference_oom_count,
            final_budget=inference_final_budget,
            stage_skipped=stage_skipped,
            async_storage_path=inference_ipc_path,
            chunks_recovered=len(completed_chunk_ids),
        )
    
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
        taxonomy_config: Dict[str, Any],
        config: Dict[str, Any],
        demographic_columns: Optional[List[str]] = None,
    ) -> LEACEContext:
        """
        Execute Stage 3: Masking, embedding, and LEACE projection computation.
        
        Args:
            chunking_ctx: Context from chunking stage.
            inference_ctx: Context from inference stage.
            output_dataset_path: Path to save clean_dataset.arrow.
            projection_matrix_path: Path to save projection_matrix.pt.
            pollution_logs_path: Path to save pollution_logs.arrow.
            gliner: Initialized GLiNER detector.
            taxonomy_config: Taxonomy configuration for masker.
            config: Pipeline configuration.
            demographic_columns: Optional list of demographic columns for LEACE.
                If provided, only these columns are used for concept encoding.
            
        Returns:
            LEACEContext with projection matrix and masked texts.
        """
        logger.info("=" * 60)
        logger.info("Stage 3/4: LEACE (Masking + Embedding + Projection)")
        logger.info("=" * 60)
        
        table = chunking_ctx.table_with_chunks
        posts = chunking_ctx.posts
        post_ids = chunking_ctx.post_ids
        entities_batch = inference_ctx.entities_batch
        
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))
        chunking_cfg = config.get("gliner", {}).get("chunking", {})
        
        # Apply typed masks
        logger.info("Applying typed masks")
        masker = SpanMasker.from_taxonomy(
            taxonomy_cfg=taxonomy_config,
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
        )
        masked_texts, pollution_logs = masker.mask_batch(posts, entities_batch, post_ids)
        total_spans = sum(len(entities) for entities in entities_batch)
        logger.info(f"Masked {total_spans} pollution spans")
        
        # Initialize embedder with mask tokens
        logger.info("Embedding masked texts")
        mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))
        encoder_output_device = self._cfg_get_optional(config, "encoder.output_device", None)
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            output_device=encoder_output_device,
            special_tokens=mask_tokens,
        )
        
        # Use filtered demographic columns if provided, else all columns
        demo_cols_for_leace = demographic_columns if demographic_columns else get_demographic_columns()
        logger.info(f"LEACE concept encoding for columns: {demo_cols_for_leace}")
        encoder = DemographicEncoder(demo_cols_for_leace).fit(table)
        
        # LEACE configuration
        leace_device_spec = self._cfg_get_optional(config, "leace.device", None)
        if leace_device_spec is None:
            leace_device_spec = self._cfg_get(config, "encoder.device")
        leace_device = self._resolve_device(leace_device_spec)
        leace_compute_dtype = self._cfg_get_optional(config, "leace.compute_dtype", "float64")
        leace_batch_size = int(self._cfg_get_optional(config, "leace.batch_size", 10000))
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device=leace_device,
            force_cpu=bool(self._cfg_get(config, "leace.force_cpu")),
            compute_dtype=leace_compute_dtype,
        )
        
        # Sharded embedding + LEACE accumulation
        shard_size = int(chunking_cfg.get("shard_size", 0))
        if shard_size <= 0:
            shard_size = leace_batch_size
        if shard_size <= 0:
            shard_size = len(masked_texts)
        
        total_shards = max(1, (len(masked_texts) + shard_size - 1) // shard_size)
        
        probe_enabled = bool(self._cfg_get(config, "probe.compute_amnesic_drop"))
        max_probe_samples = int(self._cfg_get(config, "probe.max_samples")) if probe_enabled else 0
        probe_embeddings: List[torch.Tensor] = []
        
        concept_stats = None
        
        embed_pbar = tqdm(total=total_shards, desc="Embedding & LEACE", unit="shard", dynamic_ncols=True)
        
        try:
            for shard_idx in range(total_shards):
                start = shard_idx * shard_size
                end = min(len(masked_texts), start + shard_size)
                
                shard_masked = masked_texts[start:end]
                shard_table = table.slice(start, end - start)
                
                embeddings_shard = embedder.embed_texts(shard_masked, batch_size=encoder_batch_size, show_progress=False)
                
                if probe_enabled and len(probe_embeddings) < max_probe_samples:
                    remaining = max_probe_samples - sum(e.shape[0] for e in probe_embeddings)
                    if remaining > 0:
                        probe_embeddings.append(embeddings_shard[:remaining].detach().to("cpu", dtype=torch.float32))
                
                concepts_shard = torch.from_numpy(encoder.transform(shard_table))
                concept_stats = leace.accumulate_batch_concepts(embeddings_shard, concepts_shard, concept_stats)
                
                del embeddings_shard
                embed_pbar.update(1)
        finally:
            embed_pbar.close()
        
        if concept_stats is None:
            raise RuntimeError("LEACE accumulation failed: no concept statistics computed")
        
        # Compute projection matrix
        logger.info("Computing LEACE projection matrix")
        projection_matrix = leace.compute_projection_from_concept_stats(concept_stats)
        
        # Save outputs
        logger.info("Saving LEACE outputs")
        self._save_cleaned_dataset(table, masked_texts, output_dataset_path)
        torch.save(projection_matrix.cpu(), projection_matrix_path)
        logger.info(f"Saved projection matrix: {projection_matrix_path}")
        self._save_pollution_logs(pollution_logs, pollution_logs_path)
        
        logger.info("Stage 3 complete")
        
        # Cleanup after LEACE stage
        self._cleanup_memory()
        
        return LEACEContext(
            projection_matrix=projection_matrix,
            masked_texts=masked_texts,
            pollution_logs=pollution_logs,
            total_spans=total_spans,
            stage_skipped=False,
            probe_embeddings=probe_embeddings if probe_embeddings else None,
            table_for_probing=table,
        )
    
    # =======================================================================
    # Stage 4: Probing Stage
    # =======================================================================
    def _run_probing_stage(
        self,
        leace_ctx: LEACEContext,
        output_dir: Path,
        config: Dict[str, Any],
        demographic_columns: Optional[List[str]] = None,
    ) -> ProbingContext:
        """
        Execute Stage 4: Amnesic drop metrics and visualizations.
        
        Args:
            leace_ctx: Context from LEACE stage.
            output_dir: Directory for reports.
            config: Pipeline configuration.
            demographic_columns: Optional list of demographic columns for probing.
                If provided, only these columns are probed. Supports single-label mode.
            
        Returns:
            ProbingContext with computed metrics.
        """
        logger.info("=" * 60)
        logger.info("Stage 4/4: Probing (Amnesic Drop + Visualizations)")
        logger.info("=" * 60)
        
        probe_enabled = bool(self._cfg_get(config, "probe.compute_amnesic_drop"))
        if not probe_enabled or not leace_ctx.probe_embeddings:
            logger.info("Probing disabled or no embeddings available")
            return ProbingContext(stage_skipped=True)
        
        max_probe_samples = int(self._cfg_get(config, "probe.max_samples"))
        embeddings_before = torch.cat(leace_ctx.probe_embeddings, dim=0)[:max_probe_samples]
        probe_table = leace_ctx.table_for_probing.slice(0, len(embeddings_before))
        
        P_cpu = leace_ctx.projection_matrix.detach().to("cpu", dtype=torch.float32)
        embeddings_after = embeddings_before @ P_cpu.T
        
        X_before_np = embeddings_before.numpy()
        X_after_np = embeddings_after.numpy()
        
        # Use filtered demographic columns if provided, else all columns
        demo_cols = demographic_columns if demographic_columns else get_demographic_columns()
        logger.info(f"Probing demographic columns: {demo_cols}")
        
        # Build ProbeConfig
        probe_cfg = config.get("probe", {})
        probe_config = ProbeConfig(
            backend=probe_cfg.get("backend", "auto"),
            max_iter=int(probe_cfg.get("max_iter", 1000)),
            random_state=int(self._cfg_get(config, "seed")),
            n_folds=int(probe_cfg.get("n_folds", 5)),
            pvalue_threshold=float(probe_cfg.get("pvalue_threshold", 0.05)),
            torch_lr=float(probe_cfg.get("torch_lr", 0.01)),
            torch_epochs=int(probe_cfg.get("torch_epochs", 100)),
            torch_batch_size=int(probe_cfg.get("torch_batch_size", 256)),
            torch_weight_decay=float(probe_cfg.get("torch_weight_decay", 1e-4)),
            use_class_weights=bool(probe_cfg.get("use_class_weights", True)),
            benchmark_solvers=bool(probe_cfg.get("benchmark_solvers", True)),
        )
        use_kfold = bool(probe_cfg.get("use_kfold", True))
        
        by_column: Dict[str, Any] = {}
        by_column_extended: Dict[str, Any] = {}
        drops: List[float] = []
        benchmark_results: Dict[str, Any] = {}
        separability_before: Dict[str, Any] = {}
        separability_after: Dict[str, Any] = {}
        class_imbalance: Dict[str, Any] = {}
        labels_dict: Dict[str, np.ndarray] = {}
        
        for col in demo_cols:
            labels_np = extract_probe_labels(probe_table, col)
            labels_dict[col] = labels_np
            labels_t = torch.tensor(labels_np, dtype=torch.long)
            
            logger.info(f"Computing amnesic drop for {col}...")
            result = compute_amnesic_drop_extended(
                embeddings_before,
                embeddings_after,
                labels_t,
                train_split=float(self._cfg_get(config, "probe.train_split")),
                random_state=int(self._cfg_get(config, "seed")),
                config=probe_config,
                use_kfold=use_kfold,
                device="cuda" if torch.cuda.is_available() else "cpu",
            )
            
            by_column[col] = {
                "accuracy_before": result.acc_before,
                "accuracy_after": result.acc_after,
                "amnesic_drop": result.amnesic_drop,
            }
            by_column_extended[col] = result.to_dict()
            drops.append(float(result.amnesic_drop))
            
            separability_before[col] = compute_embedding_separability(X_before_np, labels_np, sample_size=5000)
            separability_after[col] = compute_embedding_separability(X_after_np, labels_np, sample_size=5000)
            class_imbalance[col] = compute_class_imbalance_metrics(labels_np)
            
            # Solver benchmark for first column
            if probe_config.benchmark_solvers and col == demo_cols[0]:
                logger.info(f"Benchmarking solver convergence on {col}...")
                valid_mask = labels_np != -1
                X_bench = X_before_np[valid_mask]
                y_bench = labels_np[valid_mask]
                n = len(X_bench)
                train_idx = np.arange(int(n * 0.8))
                test_idx = np.arange(int(n * 0.8), n)
                benchmark_results = benchmark_solver_convergence(
                    X_train=X_bench[train_idx],
                    y_train=y_bench[train_idx],
                    X_test=X_bench[test_idx],
                    y_test=y_bench[test_idx],
                    config=probe_config,
                )
        
        # Control probe metrics
        control_probe_results = {}
        if len(demo_cols) >= 2:
            target_col = demo_cols[0]
            control_col = demo_cols[1]
            logger.info(f"Computing control probe: target={target_col}, control={control_col}")
            control_probe_results = compute_control_probe_metrics(
                embeddings_before=X_before_np,
                embeddings_after=X_after_np,
                target_labels=labels_dict[target_col],
                control_labels=labels_dict[control_col],
                control_name=control_col,
            )
        
        min_drop = float(min(drops)) if drops else 0.0
        mean_drop = float(np.mean(drops)) if drops else 0.0
        logger.info(f"Probe metrics complete: min_drop={min_drop:.2%}, mean_drop={mean_drop:.2%}")
        
        # Quality gate check
        if bool(self._cfg_get(config, "quality.enforce_thresholds")):
            threshold = float(self._cfg_get(config, "probe.amnesic_drop_threshold"))
            if min_drop < threshold:
                logger.warning(f"Amnesic drop gate failed: min={min_drop:.3f} < threshold={threshold:.3f}")
        
        # Build visualization data for pipeline orchestration
        # (Visualization is now handled centrally in phase_a_pipeline.py)
        reports_dir = None
        viz_config = config.get("visualization", {})
        if viz_config.get("enabled", True):
            reports_dir = output_dir / "reports"
            reports_dir.mkdir(parents=True, exist_ok=True)
        
        # Build visualization_data dict for pipeline to use
        # Avoid attaching raw large arrays to metadata; include lightweight summaries
        labels_summary = {col: int(len(labels_dict[col])) for col in labels_dict}
        visualization_data = {
            "by_column_extended": by_column_extended,
            "separability_before": separability_before,
            "separability_after": separability_after,
            "benchmark_results": benchmark_results,
            "control_probe_results": control_probe_results,
            "demo_cols": demo_cols,
            "X_shapes": {
                "before": tuple(X_before_np.shape) if hasattr(X_before_np, "shape") else None,
                "after": tuple(X_after_np.shape) if hasattr(X_after_np, "shape") else None,
            },
            "labels_summary": labels_summary,
            "probe_config": probe_config,
        }
        
        logger.info("Stage 4 complete (visualization data prepared for pipeline)")
        
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
            control_probe_results=control_probe_results,
            benchmark_results=benchmark_results,
            reports_dir=reports_dir,
            stage_skipped=False,
            visualization_data=visualization_data,
            demo_cols=demo_cols,
            probe_config=probe_config,
        )
    
    # =======================================================================
    # Main Execute Orchestrator
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
        Execute Phase A with staged execution architecture.
        
        Orchestrates four stages with skip flag support:
        1. Chunking: Parallel chunking → post_chunked (skip if --skip-chunking)
        2. Inference: GLiNER detection → entities (skip if --skip-inference)
        3. LEACE: Masking + projection (skip if --skip-leace)
        4. Probing: Metrics + visualizations (skip if --skip-probing)
        
        Args:
            input_dataset_path: Path to input Arrow dataset.
            output_dataset_path: Path to save clean_dataset.arrow.
            projection_matrix_path: Path to save projection_matrix.pt.
            pollution_logs_path: Path to save pollution_logs.arrow.
            config: Pipeline configuration.
            use_only_labels: Optional list of demographic labels to filter to.
            
        Returns:
            Execution metadata dictionary.
        """
        logger.info("=" * 80)
        logger.info("Phase A: HPC Strategy (Four-Stage Execution)")
        logger.info("=" * 80)
        
        # Parse skip flags
        skip_cfg = SkipStagesConfig.from_config(config)
        skip_cfg.validate_for_entry()
        logger.info(f"Skip flags: chunking={skip_cfg.skip_chunking}, inference={skip_cfg.skip_inference}, "
                   f"leace={skip_cfg.skip_leace}, probing={skip_cfg.skip_probing}")
        
        # Validate device
        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
        if device != "cuda":
            raise RuntimeError("HPC strategy requires config gliner.device='cuda'")
        
        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        self._last_device = device
        self._last_batch_size = gliner_batch_size
        
        output_dir = output_dataset_path.parent
        
        # =======================================================================
        # Load Dataset
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
        
        # Optional subset
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
        logger.info(f"Loaded {len(posts)} posts")
        
        # =======================================================================
        # Initialize GLiNER (needed for chunking and inference)
        # =======================================================================
        gliner = None
        inference_labels = None
        cached_label_embeddings = None
        taxonomy_config = None
        
        if not skip_cfg.skip_leace:  # Need GLiNER if running any stage before probing
            logger.info("Initializing GLiNER detector")
            taxonomy_config, constraints_config = self._load_taxonomy_config(config)
            
            # Apply --use-only label filter to taxonomy (if specified)
            if use_only_labels:
                taxonomy_config = self._filter_taxonomy_config(taxonomy_config, use_only_labels)
                logger.info(f"Filtered taxonomy to columns: {use_only_labels}")
            
            gliner_cfg = config.get("gliner", {})
            batch_inference_cfg = gliner_cfg.get("batch_inference", {})
            chunking_cfg = gliner_cfg.get("chunking", {})
            
            batch_config = BatchInferenceConfig(
                enable_batching=batch_inference_cfg.get("enable_batching", True),
                batch_size=gliner_batch_size,
                num_buckets=batch_inference_cfg.get("num_buckets"),
                min_bucket_size=batch_inference_cfg.get("min_bucket_size", 4),
                enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
                strict_padding=batch_inference_cfg.get("strict_padding", False),
                seq_len_buckets=batch_inference_cfg.get("seq_len_buckets"),
            )
            
            budget_config = BudgetConfig(
                model_max_length=int(self._cfg_get(config, "encoder.max_length")),
                mode=chunking_cfg.get("mode", "single_sentence"),
                legacy_sequential_mode=False,
                parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 16)),
                parallel_chunking_min_texts=int(chunking_cfg.get("parallel_chunking_min_texts", 512)),
                gliner_max_words=int(gliner_cfg.get("gliner_max_words", 512)),
                tokens_per_word_ratio=float(gliner_cfg.get("tokens_per_word_ratio", 1.3)),
            )
            
            execution_cfg = config.get("execution", {})
            
            gliner = GLiNERDetector(
                model_name=str(self._cfg_get(config, "gliner.model")),
                device=device,
                max_length=int(self._cfg_get(config, "encoder.max_length")),
                confidence_threshold=float(self._cfg_get(config, "gliner.confidence_threshold")),
                taxonomy_config=taxonomy_config,
                constraints_config=constraints_config,
                budget_config=budget_config,
                batch_inference_config=batch_config,
                require_bi_encoder=bool(gliner_cfg.get("require_bi_encoder", False)),
                execution_config=execution_cfg,
            )
            
            inference_labels = gliner.taxonomy.get_inference_labels()
            cached_label_embeddings = gliner.get_cached_label_embeddings(inference_labels)
            if cached_label_embeddings is not None:
                logger.info(f"Hoisted label embeddings for {len(inference_labels)} labels")
        
        # =======================================================================
        # Stage 1: Chunking
        # =======================================================================
        if skip_cfg.skip_chunking:
            logger.info("Skipping Stage 1 (Chunking) - validating prerequisites")
            if not is_post_chunked_fully_populated(table):
                if not skip_cfg.force_skip:
                    raise RuntimeError("--skip-chunking requires post_chunked column to be fully populated")
                else:
                    logger.warning("--skip-chunking requested but post_chunked is not fully populated; continuing due to force_skip=True")
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
            logger.info("Skipping Stage 2 (Inference) - loading from disk")
            inference_ipc_path = output_dir / "inference_results.arrow"
            if not inference_ipc_path.exists():
                if not skip_cfg.force_skip:
                    raise RuntimeError(f"--skip-inference requires {inference_ipc_path} to exist")
                else:
                    logger.warning("--skip-inference requested but inference_results.arrow not found; continuing due to force_skip=True")
                    entities_batch = [[] for _ in posts]
                    inference_ctx = InferenceContext(
                        entities_batch=entities_batch,
                        skipped_doc_indices=set(range(len(posts))),
                        oom_count=0,
                        final_budget=0,
                        stage_skipped=True,
                        inference_results_path=None,
                    )
            else:
                # Load inference results from disk
                flattened = flatten_chunks(
                    chunking_ctx.table_with_chunks["post_chunked"],
                    extract_texts=True,
                    include_chunk_metadata=False,
                )
                async_storer = AsyncResultStorer(
                    output_path=inference_ipc_path,
                    num_chunks=flattened.num_chunks,
                )
                
                if not async_storer.is_complete:
                    if not skip_cfg.force_skip:
                        raise RuntimeError("--skip-inference requires complete inference_results.arrow")
                    else:
                        logger.warning("--skip-inference requested but inference_results.arrow is incomplete; continuing due to force_skip=True")
                        flat_results = []
                        chunk_starts = np.zeros(0, dtype=np.int32)
                        entities_batch = [[] for _ in range(flattened.num_docs)]
                        inference_ctx = InferenceContext(
                            entities_batch=entities_batch,
                            skipped_doc_indices=set(range(flattened.num_docs)),
                            oom_count=0,
                            final_budget=0,
                            stage_skipped=True,
                            inference_results_path=inference_ipc_path,
                        )
                else:
                    flat_results = async_storer.load_results_as_flat_list(flattened.num_chunks)
                    chunk_starts = flattened.chunk_starts if flattened.chunk_starts is not None else np.zeros(flattened.num_chunks, dtype=np.int32)
                    
                    entities_batch = []
                    for doc_idx in range(flattened.num_docs):
                        doc_start = flattened.doc_offsets[doc_idx]
                        doc_end = flattened.doc_offsets[doc_idx + 1]
                        projected_entities = []
                        for flat_idx in range(doc_start, doc_end):
                            chunk_entities = flat_results[flat_idx]
                            chunk_offset = int(chunk_starts[flat_idx])
                            for entity in chunk_entities:
                                projected = dict(entity)
                                projected["start"] = projected.get("start", 0) + chunk_offset
                                projected["end"] = projected.get("end", 0) + chunk_offset
                                projected_entities.append(projected)
                        entities_batch.append(deduplicate_entities(projected_entities))
            
            inference_ctx = InferenceContext(
                entities_batch=entities_batch,
                stage_skipped=True,
                async_storage_path=inference_ipc_path,
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
            logger.info("Skipping Stage 3 (LEACE) - loading projection matrix")
            if not projection_matrix_path.exists():
                if not skip_cfg.force_skip:
                    raise RuntimeError(f"--skip-leace requires {projection_matrix_path} to exist")
                else:
                    logger.warning("--skip-leace requested but projection_matrix.pt not found; continuing due to force_skip=True")
                    projection_matrix = None
            else:
                projection_matrix = torch.load(projection_matrix_path, map_location="cpu")
            
            # Load clean dataset if probing is needed
            probe_embeddings = None
            table_for_probing = None
            if not skip_cfg.skip_probing and output_dataset_path.exists():
                table_for_probing = pa.feather.read_table(output_dataset_path)
            
            leace_ctx = LEACEContext(
                projection_matrix=projection_matrix,
                masked_texts=[],
                pollution_logs=[],
                stage_skipped=True,
                probe_embeddings=probe_embeddings,
                table_for_probing=table_for_probing,
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
                taxonomy_config=taxonomy_config,
                config=config,
                demographic_columns=demographic_columns,
            )
        
        # =======================================================================
        # Stage 4: Probing
        # =======================================================================
        # Derive demographic columns for probing (use same as LEACE stage)
        demographic_columns = use_only_labels if use_only_labels else None
        
        if skip_cfg.skip_probing:
            logger.info("Skipping Stage 4 (Probing)")
            probing_ctx = ProbingContext(stage_skipped=True)
        else:
            probing_ctx = self._run_probing_stage(
                leace_ctx=leace_ctx,
                output_dir=output_dir,
                config=config,
                demographic_columns=demographic_columns,
            )
        
        # =======================================================================
        # Build Metadata
        # =======================================================================
        metadata: Dict[str, Any] = {
            "num_samples": len(chunking_ctx.table_with_chunks),
            "num_pollution_spans": leace_ctx.total_spans,
            "projection_matrix_shape": list(leace_ctx.projection_matrix.shape),
            "device": device,
            "use_bf16": self.use_bf16,
            "label_filter": use_only_labels,  # None for multi-label mode, list for single-label
            "staged_execution": True,
            "stages": {
                "chunking_skipped": chunking_ctx.stage_skipped,
                "inference_skipped": inference_ctx.stage_skipped,
                "leace_skipped": leace_ctx.stage_skipped,
                "probing_skipped": probing_ctx.stage_skipped,
            },
            "inference_completeness": {
                "total_docs": len(chunking_ctx.posts),
                "skipped_docs": len(inference_ctx.skipped_doc_indices),
                "oom_count": inference_ctx.oom_count,
                "final_budget": inference_ctx.final_budget,
            },
        }
        
        if inference_ctx.async_storage_path:
            metadata["async_storage"] = {
                "ipc_path": str(inference_ctx.async_storage_path),
                "chunks_recovered": inference_ctx.chunks_recovered,
            }
        
        # Explicit recall
        if bool(self._cfg_get(config, "gliner.compute_explicit_recall")) and gliner and not skip_cfg.skip_leace:
            recall = compute_explicit_recall(
                chunking_ctx.posts,
                inference_ctx.entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall
        
        # Add probe metrics
        if not probing_ctx.stage_skipped:
            metadata["probe"] = {
                "by_column": probing_ctx.by_column,
                "by_column_extended": probing_ctx.by_column_extended,
                "min_amnesic_drop": probing_ctx.min_amnesic_drop,
                "mean_amnesic_drop": probing_ctx.mean_amnesic_drop,
                "threshold": float(self._cfg_get(config, "probe.amnesic_drop_threshold")),
            }
            metadata["separability"] = {
                "before": probing_ctx.separability_before,
                "after": probing_ctx.separability_after,
            }
            metadata["class_imbalance"] = probing_ctx.class_imbalance
            if probing_ctx.control_probe_results:
                metadata["control_probe"] = probing_ctx.control_probe_results
            if probing_ctx.benchmark_results:
                metadata["probe"]["solver_benchmark"] = probing_ctx.benchmark_results
            if probing_ctx.reports_dir:
                metadata["reports_dir"] = str(probing_ctx.reports_dir)
            # Include visualization data for pipeline orchestration
            if probing_ctx.visualization_data:
                metadata["visualization_data"] = probing_ctx.visualization_data
            if probing_ctx.demo_cols:
                metadata["demo_cols"] = probing_ctx.demo_cols
            if probing_ctx.probe_config:
                metadata["probe_config"] = probing_ctx.probe_config
        
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
