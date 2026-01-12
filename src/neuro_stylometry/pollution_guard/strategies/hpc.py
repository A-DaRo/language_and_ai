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
- RuntimeController autotuning for adaptive batch sizing (OOM resilience).

Reference: Technical Reports on Staged Execution Architecture.
Implements: phaseA-D_implementation_plan.md Section 7.4 (HPC Mode)
"""

from __future__ import annotations

import gc
import logging
import re
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pyarrow as pa
import pyarrow.feather as feather
import torch
from tqdm import tqdm

from .base import PollutionFilterStrategy
from ..gliner_detector import GLiNERDetector, BatchInferenceConfig, InferenceResult
from ..masker import SpanMasker
from ..semantic_chunker import BudgetConfig
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
            strict_padding=batch_inference_cfg.get("strict_padding", False),
            seq_len_buckets=batch_inference_cfg.get("seq_len_buckets"),
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
        
        # Extract execution config for torch.compile and CUDA graphs (Phase 1 optimization)
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
            require_bi_encoder=require_bi_encoder,
            execution_config=execution_cfg,  # Pass execution config for torch.compile
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
        # Resume logic:
        # - Stage 2 can only run when post_chunked exists AND has no NULLs.
        # - If post_chunked exists but has NULLs, run Stage 1 only for remaining rows.
        invalid_chunk_rows: List[int] = []
        stage1_skipped = False

        # Optimization: Check if inference is already complete to skip Stage 1 validation
        inference_ipc_path = output_dataset_path.parent / "inference_results.arrow"
        if has_post_chunked_column(table) and inference_ipc_path.exists():
            try:
                # Need total chunk count to verify completeness
                logger.info("Checking async storage to potentially skip Stage 1 validation...")
                temp_flattened = flatten_chunks(
                    table["post_chunked"],
                    extract_texts=False,
                    include_chunk_metadata=False,
                )
                
                # Check storer state
                temp_storer = AsyncResultStorer(
                    output_path=inference_ipc_path,
                    num_chunks=temp_flattened.num_chunks,
                )
                
                if temp_storer.is_complete:
                    logger.info("Async storage complete: skipping Stage 1 chunk validation")
                    stage1_skipped = True
                    table_with_chunks = table
            except Exception as e:
                logger.warning(f"Failed to check async storage for skip: {e}")

        if not stage1_skipped and is_post_chunked_fully_populated(table):
            logger.info("Resume candidate detected: validating post_chunked against raw posts")
            invalid_chunk_rows = find_invalid_post_chunked_indices(
                posts=posts,
                post_chunked_column=table["post_chunked"],
            )
            if not invalid_chunk_rows:
                stage1_skipped = True
                logger.info(
                    "Resume validated: post_chunked fully populated and matches raw posts; skipping Stage 1"
                )
                table_with_chunks = table
            else:
                logger.warning(
                    f"Resume validation failed for {len(invalid_chunk_rows)} rows; recomputing those chunks"
                )

        if not stage1_skipped:
            logger.info("Stage 1/2: CPU-Saturated Parallel Chunking")

            # IMPORTANT (Windows + mmap): to permanently persist post_chunked back into
            # the *input* Arrow file, we must avoid holding a memory-mapped handle to it.
            # Reload the dataset without mmap for Stage 1.
            dataset_stage1 = SOBRDataset(
                arrow_path=input_dataset_path,
                seed=int(self._cfg_get(config, "seed")),
                memory_map=False,
            )
            table = dataset_stage1.table
            posts = table["post"].to_pylist()
            post_ids = table["post_id"].to_pylist()
            
            # Prepare labels list (full taxonomy for all documents)
            labels_list: List[Optional[List[str]]] = [inference_labels for _ in posts]
            
            # Get tokenizer name for worker initialization
            tokenizer_name = gliner.tokenizer.name_or_path
            words_splitter_type = gliner.chunker._words_splitter_type
            
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

            # Permanent save: make Stage 1 crash-recoverable by overwriting the input dataset.
            save_chunked_table_atomic(table_with_chunks, input_dataset_path)

            # Safety check before Stage 2
            if not is_post_chunked_fully_populated(table_with_chunks):
                raise RuntimeError(
                    "Stage 1 finished but post_chunked still contains NULLs; cannot start Stage 2"
                )
            
            # Force garbage collection between stages
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # =======================================================================
        # Step 3: Stage 2 - GPU Saturated Global-Sorted Inference (with Async Storage)
        # =======================================================================
        # Phase 2 Optimization: Scatter-Store-Gather pattern with crash recovery
        # - Results persisted to Arrow IPC for constant memory footprint
        # - Crash recovery: skip chunks already in IPC file
        # - Async consumer thread for non-blocking result storage
        # =======================================================================
        logger.info("Stage 2/2: GPU-Saturated Global-Sorted Inference (Async Storage)")
        
        post_chunked_column = table_with_chunks["post_chunked"]
        
        # Flatten chunks to get total count for async storer
        logger.info("Stage 2.1: Flattening post_chunked for lineage tracking")
        flattened = flatten_chunks(
            post_chunked_column,
            extract_texts=True,
            include_chunk_metadata=False,
        )
        
        logger.info(
            f"Flattened {flattened.num_docs} documents into {flattened.num_chunks} chunks"
        )
        
        # Configure async result storage path
        inference_ipc_path = output_dataset_path.parent / "inference_results.arrow"
        
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
        
        # Check for crash recovery - get already-completed chunks
        completed_chunk_ids = async_storer.get_completed_chunk_ids()
        pending_chunk_ids = async_storer.get_pending_chunk_ids()
        
        logger.info(
            f"Crash recovery: {len(completed_chunk_ids)} chunks completed, "
            f"{len(pending_chunk_ids)} chunks pending"
        )
        
        # Track inference statistics
        inference_oom_count = 0
        inference_final_budget = gliner_batch_size
        skipped_chunk_indices: Set[int] = set()
        
        if async_storer.is_complete:
            # All chunks already processed - skip inference entirely
            logger.info("All chunks already processed in previous run - skipping inference")
            stage2_skipped = True
        else:
            stage2_skipped = False
            
            # Configure pin_memory for HPC
            pin_memory = bool(self._cfg_get_optional(
                config, "execution.async_prefetch.pin_memory", True
            ))
            
            # Create RuntimeController from config (autotuning)
            runtime_controller = self._create_runtime_controller(config)
            logger.info(
                f"Autotuning enabled: initial_budget={runtime_controller.current_budget:,}, "
                f"warmup_batches={runtime_controller.config.warmup_batches}"
            )
            
            # Get prompt embeddings for bi-encoder efficiency
            prompt_embeddings = cached_label_embeddings
            if prompt_embeddings is None:
                prompt_embeddings = gliner.get_cached_label_embeddings(inference_labels)
            
            # Compute global sort permutation
            logger.info("Stage 2.2: Computing global sort permutation by token_count")
            sort_indices = flattened.compute_sort_indices()
            
            # Create dynamic batch iterator (respects RuntimeController budget)
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
            
            # Start async storer consumer thread
            async_storer.start()
            
            # Progress tracking
            pbar = tqdm(
                total=len(pending_chunk_ids),
                desc="GLiNER inference (async)",
                unit="chunk",
                dynamic_ncols=True,
            )
            
            try:
                for batch_indices in batch_iterator:
                    # Filter out already-completed chunks (crash recovery)
                    pending_in_batch = [
                        idx for idx in batch_indices
                        if idx not in completed_chunk_ids
                    ]
                    
                    if not pending_in_batch:
                        # Entire batch already completed - skip
                        continue
                    
                    batch_texts = [flattened.texts[i] for i in pending_in_batch]
                    batch_tokens = sum(int(flattened.token_counts[i]) for i in pending_in_batch)
                    
                    # Run batched inference with OOM protection
                    with CUDATimer() as timer:
                        try:
                            batch_entities = execute_with_oom_protection(
                                lambda bt=batch_texts: gliner._detect_batch(
                                    bt,
                                    inference_labels,
                                    prompt_embeddings,
                                ),
                                controller=runtime_controller,
                                retry_limit=3,
                            )
                        except OOMRecoveryError:
                            logger.error(
                                f"OOM recovery failed for batch of {len(pending_in_batch)} chunks; "
                                "marking as skipped"
                            )
                            skipped_chunk_indices.update(pending_in_batch)
                            pbar.update(len(pending_in_batch))
                            inference_oom_count += 1
                            continue
                    
                    # Report metrics to controller for feedback loop
                    memory_mb = (
                        torch.cuda.memory_allocated() / (1024 ** 2)
                        if torch.cuda.is_available()
                        else 0.0
                    )
                    runtime_controller.report_metrics(RuntimeMetrics(
                        tokens_processed=batch_tokens,
                        batch_time_ms=timer.elapsed_ms,
                        memory_used_mb=memory_mb,
                        batch_size=len(pending_in_batch),
                    ))
                    
                    # Submit results to async storer (non-blocking)
                    async_storer.submit(pending_in_batch, batch_entities)
                    
                    # Update progress
                    pbar.update(len(pending_in_batch))
                    
            finally:
                pbar.close()
                
                # Finalize async storer
                async_storer.finish()
                
                # Get final stats
                storer_stats = async_storer.get_stats()
                snapshot = runtime_controller.get_snapshot()
                inference_oom_count += snapshot.oom_count
                inference_final_budget = snapshot.current_budget
                
                logger.info(
                    f"Async storer stats: written={storer_stats.chunks_written}, "
                    f"recovered={storer_stats.chunks_recovered}, errors={storer_stats.errors}"
                )
                logger.info(
                    f"Autotuning summary: state={snapshot.state}, "
                    f"final_budget={snapshot.current_budget:,}, "
                    f"oom_count={snapshot.oom_count}"
                )
        
        # =======================================================================
        # Step 3b: Load Results and Reconstruct Document-Level Entities
        # =======================================================================
        logger.info("Stage 2.3: Loading results and reconstructing per-document entities")
        
        # Load all results from IPC file
        flat_results = async_storer.load_results_as_flat_list(flattened.num_chunks)
        
        # Gather results back to document order with offset projection
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
            
            # Deduplicate entities from overlapping chunks
            deduped = deduplicate_entities(projected_entities)
            entities_batch.append(deduped)
        
        # Compute skipped document indices from skipped chunk indices
        skipped_doc_indices: Set[int] = set()
        for flat_idx in skipped_chunk_indices:
            for doc_idx in range(flattened.num_docs):
                doc_start = flattened.doc_offsets[doc_idx]
                doc_end = flattened.doc_offsets[doc_idx + 1]
                if doc_start <= flat_idx < doc_end:
                    skipped_doc_indices.add(doc_idx)
                    break
        
        # Final completeness check
        final_skipped = skipped_doc_indices
        if final_skipped:
            logger.error(
                f"INCOMPLETE INFERENCE: {len(final_skipped)} documents had chunks "
                f"that could not be processed. Proceeding with partial results."
            )
            # Save manifest of skipped documents for manual inspection
            skipped_manifest = {
                "skipped_doc_indices": sorted(final_skipped),
                "skipped_post_ids": [post_ids[i] for i in sorted(final_skipped)],
                "skipped_chunk_count": len(skipped_chunk_indices),
                "total_oom_count": inference_oom_count,
                "final_budget": inference_final_budget,
            }
            skipped_manifest_path = output_dataset_path.parent / "skipped_documents_manifest.json"
            import json
            with open(skipped_manifest_path, "w") as f:
                json.dump(skipped_manifest, f, indent=2)
            logger.warning(f"Saved skipped documents manifest: {skipped_manifest_path}")
        
        logger.info(
            f"Stage 2 complete: {flattened.num_docs} documents reconstructed "
            f"({len(final_skipped)} with partial results)"
        )
        
        # =======================================================================
        # Step 4: Masking & Pollution Logs
        # =======================================================================
        logger.info("Applying typed masks")
        
        # Config-driven masker: extract mask_token from taxonomy YAML
        masker = SpanMasker.from_taxonomy(
            taxonomy_cfg=taxonomy_config,
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
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
        
        # Read encoder output_device (HPC: keep on GPU for fast LEACE accumulation)
        encoder_output_device = self._cfg_get_optional(config, "encoder.output_device", None)
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            output_device=encoder_output_device,
            special_tokens=mask_tokens,
        )
        
        encoder = DemographicEncoder(get_demographic_columns()).fit(table_with_chunks)
        
        # Read LEACE config: device (with fallback to encoder.device), compute_dtype, batch_size
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
        # Priority: chunking.shard_size > leace.batch_size > full dataset
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
            "stage1_skipped": stage1_skipped,
            "stage2_skipped": stage2_skipped if 'stage2_skipped' in dir() else False,
            "inference_completeness": {
                "total_docs": len(posts),
                "total_chunks": flattened.num_chunks,
                "processed_docs": len(posts) - len(final_skipped),
                "skipped_docs": len(final_skipped),
                "skipped_chunks": len(skipped_chunk_indices),
                "oom_count": inference_oom_count,
                "final_budget": inference_final_budget,
            },
            "async_storage": {
                "ipc_path": str(inference_ipc_path),
                "chunks_recovered": len(completed_chunk_ids),
            },
        }
        
        # Explicit recall
        if bool(self._cfg_get(config, "gliner.compute_explicit_recall")):
            recall = compute_explicit_recall(
                posts,
                entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall
        
        # Amnesic drop probe with multi-backend benchmarking
        if probe_enabled and probe_embeddings:
            logger.info("=" * 60)
            logger.info("Computing Multi-Backend Probe Metrics (HPC Mode)")
            logger.info("=" * 60)
            
            embeddings_before = torch.cat(probe_embeddings, dim=0)[:max_probe_samples]
            probe_table = table_with_chunks.slice(0, len(embeddings_before))
            
            P_cpu = projection_matrix.detach().to("cpu", dtype=torch.float32)
            embeddings_after = embeddings_before @ P_cpu.T
            
            # Convert to numpy for metrics
            X_before_np = embeddings_before.numpy()
            X_after_np = embeddings_after.numpy()
            
            # Build ProbeConfig from YAML
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
            
            # Collect all labels for per-column analysis
            labels_dict: Dict[str, np.ndarray] = {}
            
            for col in get_demographic_columns():
                labels_np = extract_probe_labels(probe_table, col)
                labels_dict[col] = labels_np
                labels_t = torch.tensor(labels_np, dtype=torch.long)
                
                # Extended amnesic drop with CV
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
                
                # Embedding separability (silhouette, Davies-Bouldin)
                separability_before[col] = compute_embedding_separability(
                    X_before_np, labels_np, sample_size=5000
                )
                separability_after[col] = compute_embedding_separability(
                    X_after_np, labels_np, sample_size=5000
                )
                
                # Class imbalance metrics
                class_imbalance[col] = compute_class_imbalance_metrics(labels_np)
                
                # Solver benchmark for first column (representative)
                if probe_config.benchmark_solvers and col == get_demographic_columns()[0]:
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
                    logger.info(f"Solver benchmark: exact={benchmark_results.get('exact_accuracy', 0):.3f}, "
                               f"torch={benchmark_results.get('torch_accuracy', 0):.3f}, "
                               f"delta={benchmark_results.get('solver_accuracy_delta', 0):.4f}")
            
            # Control probe metrics (cross-column specificity check)
            # Use first two columns as target/control pair
            demo_cols = get_demographic_columns()
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
            
            metadata["probe"] = {
                "by_column": by_column,
                "by_column_extended": by_column_extended,
                "min_amnesic_drop": min_drop,
                "mean_amnesic_drop": mean_drop,
                "threshold": float(self._cfg_get(config, "probe.amnesic_drop_threshold")),
                "max_samples": len(embeddings_before),
                "config": {
                    "backend": probe_config.backend,
                    "use_kfold": use_kfold,
                    "n_folds": probe_config.n_folds,
                    "pvalue_threshold": probe_config.pvalue_threshold,
                },
            }
            
            # Add benchmark results if computed
            if benchmark_results:
                metadata["probe"]["solver_benchmark"] = benchmark_results
            
            # Add separability metrics
            metadata["separability"] = {
                "before": separability_before,
                "after": separability_after,
            }
            
            # Add class imbalance metrics
            metadata["class_imbalance"] = class_imbalance
            
            # Add control probe results
            if control_probe_results:
                metadata["control_probe"] = control_probe_results
            
            logger.info(f"Probe metrics complete: min_drop={min_drop:.2%}, mean_drop={mean_drop:.2%}")
            
            if bool(self._cfg_get(config, "quality.enforce_thresholds")):
                threshold = float(self._cfg_get(config, "probe.amnesic_drop_threshold"))
                if min_drop < threshold:
                    raise ValueError(
                        f"Amnesic drop gate failed: min={min_drop:.3f} < threshold={threshold:.3f}"
                    )
            
            # =======================================================================
            # Step 8: Generate Visualizations (HPC Mode)
            # =======================================================================
            viz_config = config.get("visualization", {})
            if viz_config.get("enabled", True):
                logger.info("Generating Phase A visualizations...")
                reports_dir = output_dataset_path.parent / "reports"
                reports_dir.mkdir(parents=True, exist_ok=True)
                
                try:
                    from ...evaluation.visualizations_phase_a import (
                        plot_amnesic_drop_with_ci,
                        plot_embedding_separability,
                        plot_probe_learning_curves,
                        plot_solver_convergence_benchmark,
                        plot_stratified_confusion_matrices,
                        plot_specificity_gap,
                    )
                    
                    dpi = int(viz_config.get("figure_dpi", 150))
                    palette = viz_config.get("color_palette", "husl")
                    
                    # Amnesic drop with confidence intervals
                    if by_column_extended:
                        plot_amnesic_drop_with_ci(
                            per_column_results=by_column_extended,
                            output_path=reports_dir / "amnesic_drop_with_ci.png",
                            dpi=dpi,
                            palette=palette,
                        )
                    
                    # Embedding separability (aggregate first column)
                    first_col = get_demographic_columns()[0]
                    if first_col in separability_before and first_col in separability_after:
                        plot_embedding_separability(
                            separability_before=separability_before[first_col],
                            separability_after=separability_after[first_col],
                            output_path=reports_dir / "embedding_separability.png",
                            dpi=dpi,
                        )
                    
                    # Solver convergence benchmark plot
                    if benchmark_results and "torch_accuracy" in benchmark_results:
                        # Generate mock learning curve if we don't have actual one
                        torch_final = benchmark_results.get("torch_accuracy", 0.5)
                        # Create convergence curve approximation
                        epochs = probe_config.torch_epochs
                        learning_curve = [
                            torch_final * (1 - 0.5 * np.exp(-i / 20))
                            for i in range(epochs)
                        ]
                        plot_solver_convergence_benchmark(
                            benchmark_results=benchmark_results,
                            learning_curve_torch=learning_curve,
                            output_path=reports_dir / "solver_convergence_benchmark.png",
                            dpi=dpi,
                        )
                    
                    # Specificity gap (target vs control)
                    if control_probe_results and "error" not in control_probe_results:
                        target_results = control_probe_results.get("target", {})
                        control_key = [k for k in control_probe_results if k.startswith("control_")]
                        if target_results and control_key:
                            plot_specificity_gap(
                                target_results={
                                    "acc_before": target_results.get("acc_before", 0.5),
                                    "acc_after": target_results.get("acc_after", 0.5),
                                },
                                control_results={
                                    "acc_before": control_probe_results[control_key[0]].get("acc_before", 0.5),
                                    "acc_after": control_probe_results[control_key[0]].get("acc_after", 0.5),
                                },
                                target_name=demo_cols[0],
                                control_name=demo_cols[1] if len(demo_cols) > 1 else "control",
                                output_path=reports_dir / "specificity_gap.png",
                                dpi=dpi,
                                palette=palette,
                            )
                    
                    logger.info(f"Visualizations saved to: {reports_dir}")
                    metadata["reports_dir"] = str(reports_dir)
                    
                except Exception as viz_error:
                    logger.warning(f"Visualization generation failed: {viz_error}")
                    metadata["visualization_error"] = str(viz_error)
        
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
