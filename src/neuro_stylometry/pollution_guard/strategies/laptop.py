"""
Laptop Strategy for Phase A Pollution Filtering (Staged Execution Architecture).

Implements two-stage decoupled execution optimized for limited hardware:
- Stage 1 (CPU Saturation): Parallel chunking with streaming writes to post_chunked.
- Stage 2 (GPU/CPU Inference): Global-sorted inference with smaller micro-batches.

Key Optimizations:
- Smaller micro-batches (100-200 docs) for low RAM.
- Reduced parallel workers (4 default) for laptop cores.
- CPU fallback path for systems without CUDA.
- Batch accumulation for LEACE covariance statistics.
- RuntimeController autotuning for adaptive batch sizing (OOM resilience).

Reference: Technical Reports on Staged Execution Architecture.
Implements: phaseA-D_implementation_plan.md Section 7.3 (Laptop Mode)
"""

from __future__ import annotations

import gc
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Set

import numpy as np
import pandas as pd
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
from ..probe import compute_amnesic_drop
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

logger = logging.getLogger(__name__)


class LaptopFilterStrategy(PollutionFilterStrategy):
    """
    Laptop-optimized pollution filtering strategy with staged execution.
    
    Architecture (Staged Execution):
    - Stage 1: CPU chunking with smaller micro-batches → persist to post_chunked.
    - Stage 2: Inference on CPU or low-VRAM GPU with reduced batch sizes.
    
    Key Features:
    - Works on CPU or low-VRAM GPU (4-8GB).
    - Smaller micro-batches (100-200 docs vs 1000 for HPC).
    - Reduced parallel workers (4 vs 16 for HPC).
    - Crash recovery: if post_chunked exists, skip Stage 1.
    - Batch accumulation for LEACE (memory-efficient).
    
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
    
    def execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """
        Execute Phase A with staged execution architecture (laptop-optimized).
        
        Pipeline:
        1. Load dataset (check for existing post_chunked).
        2. Stage 1: Parallel chunking → persist post_chunked (skip if exists).
        3. Stage 2: Global-sorted inference (smaller batches).
        4. Masking & LEACE computation (batch accumulation).
        5. Save outputs.
        """
        logger.info("=" * 80)
        logger.info("Phase A: Laptop Strategy (Staged Execution)")
        logger.info("=" * 80)

        # Resolve device (laptop gracefully handles missing CUDA)
        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))

        self._last_device = device
        self._last_batch_size = gliner_batch_size
        
        logger.info(f"Device: {device}, GLiNER batch: {gliner_batch_size}, Encoder batch: {encoder_batch_size}")
        
        # =======================================================================
        # Step 1: Load Dataset
        # =======================================================================
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(
            arrow_path=input_dataset_path,
            seed=int(self._cfg_get(config, "seed")),
        )
        table = dataset.table

        # Optional subset (strictly config-driven, recommended for laptop)
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
            min_bucket_size=batch_inference_cfg.get("min_bucket_size", 2),  # Smaller for laptop
            enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
            strict_padding=batch_inference_cfg.get("strict_padding", False),
            seq_len_buckets=batch_inference_cfg.get("seq_len_buckets"),
        )
        
        chunking_cfg = gliner_cfg.get("chunking", {})
        budget_config = BudgetConfig(
            model_max_length=int(self._cfg_get(config, "encoder.max_length")),
            mode=chunking_cfg.get("mode", "single_sentence"),
            legacy_sequential_mode=False,  # Always use staged mode
            parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 4)),  # Fewer for laptop
            parallel_chunking_min_texts=int(chunking_cfg.get("parallel_chunking_min_texts", 128)),  # Lower threshold
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
        # Step 2: Stage 1 - CPU Chunking (Smaller Micro-Batches for Laptop)
        # =======================================================================
        # Check for resume: if post_chunked exists, skip Stage 1
        # Resume logic:
        # - Stage 2 can only run when post_chunked exists AND has no NULLs.
        # - If post_chunked exists but has NULLs, run Stage 1 only for remaining rows.
        invalid_chunk_rows: List[int] = []
        stage1_skipped = False

        if is_post_chunked_fully_populated(table):
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
            logger.info("Stage 1/2: CPU Chunking (Laptop Micro-Batches)")

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
            
            # Laptop-optimized: fewer workers, smaller micro-batches
            num_workers = int(chunking_cfg.get("parallel_chunking_workers", 4))
            micro_batch_size = int(chunking_cfg.get("micro_batch_size", 200))  # Smaller than HPC's 1000
            
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
            
            # Force garbage collection between stages (important for low RAM)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
        
        # =======================================================================
        # Step 3: Stage 2 - Inference with Autotuning (Laptop-Optimized)
        # =======================================================================
        logger.info("Stage 2/2: Inference (Laptop-Optimized with Autotuning)")
        
        post_chunked_column = table_with_chunks["post_chunked"]
        
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
            runtime_controller=runtime_controller,  # Enable autotuning
        )
        
        entities_batch = inference_result.entities
        
        # =======================================================================
        # Step 3b: Completeness Guard - Retry skipped documents
        # =======================================================================
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
            
            # Create a conservative recovery controller at 80% of current ceiling
            oom_ceiling = runtime_controller.oom_ceiling
            recovery_budget = int(oom_ceiling * 0.8)
            recovery_budget = max(recovery_budget, runtime_controller.config.min_token_budget)
            
            # Laptop: more conservative recovery settings
            recovery_config = RuntimeConfig(
                warmup_batches=2,  # Quick warmup
                initial_token_budget=recovery_budget,
                min_token_budget=runtime_controller.config.min_token_budget,
                max_token_budget=recovery_budget,  # Cap at recovery budget
                memory_headroom_mb=runtime_controller.config.memory_headroom_mb * 2.0,  # Extra headroom for laptop
                scale_up_factor=1.03,  # Very conservative scaling for laptop
                scale_down_factor=0.6,
                oom_slash_factor=0.3,  # More aggressive slash on retry for laptop
                stability_threshold=0.25,
                history_window=4,
                recovery_patience=2,
            )
            recovery_controller = RuntimeController(recovery_config)
            
            logger.info(
                f"Retry controller: recovery_budget={recovery_budget:,} "
                f"(80% of ceiling={oom_ceiling:,})"
            )
            
            # Build subset table for skipped documents only
            skipped_indices = sorted(inference_result.skipped_doc_indices)
            
            # Re-run inference only on skipped documents
            retry_result = gliner.inference_from_manifest(
                post_chunked_column=post_chunked_column.take(pa.array(skipped_indices)),
                labels=inference_labels,
                batch_size=max(1, gliner_batch_size // 4),  # Smaller batches for laptop
                cached_label_embeddings=cached_label_embeddings,
                pin_memory=False,  # Never pin during recovery on laptop
                show_progress=True,
                runtime_controller=recovery_controller,
            )
            
            # Merge retry results back into entities_batch
            for local_idx, doc_idx in enumerate(skipped_indices):
                if local_idx not in retry_result.skipped_doc_indices:
                    # Successfully processed this document
                    entities_batch[doc_idx] = retry_result.entities[local_idx]
            
            # Update skipped set: only docs that still failed
            new_skipped = {
                skipped_indices[local_idx]
                for local_idx in retry_result.skipped_doc_indices
            }
            
            # Update for next iteration
            inference_result = InferenceResult(
                entities=entities_batch,
                skipped_doc_indices=new_skipped,
                oom_count=inference_result.oom_count + retry_result.oom_count,
                final_budget=retry_result.final_budget,
            )
            
            # Force GC between retries (critical for laptop)
            gc.collect()
            if torch.cuda.is_available():
                torch.cuda.empty_cache()
            
            if not new_skipped:
                logger.info(f"All documents processed after {retry_round} retry round(s)")
                break
        
        # Final completeness check
        final_skipped = inference_result.skipped_doc_indices
        if final_skipped:
            logger.error(
                f"INCOMPLETE INFERENCE: {len(final_skipped)} documents could not be processed "
                f"after {max_retry_rounds} retry rounds. Proceeding with partial results."
            )
            # Save manifest of skipped documents for manual inspection
            skipped_manifest = {
                "skipped_doc_indices": sorted(final_skipped),
                "skipped_post_ids": [post_ids[i] for i in sorted(final_skipped)],
                "total_oom_count": inference_result.oom_count,
                "final_budget": inference_result.final_budget,
            }
            skipped_manifest_path = output_dataset_path.parent / "skipped_documents_manifest.json"
            import json
            with open(skipped_manifest_path, "w") as f:
                json.dump(skipped_manifest, f, indent=2)
            logger.warning(f"Saved skipped documents manifest: {skipped_manifest_path}")
        
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
        # Step 5: Embedding & LEACE (Batch Accumulation for Laptop)
        # =======================================================================
        logger.info("Embedding masked texts and computing LEACE projection")
        
        # Extract mask tokens for tokenizer alignment
        mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))
        
        # Laptop: embeddings to CPU for LEACE accumulation
        encoder_output_device = str(self._cfg_get_optional(
            config, "encoder.output_device", "cpu"
        ))
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            output_device=encoder_output_device,
            special_tokens=mask_tokens,
        )
        
        encoder = DemographicEncoder(get_demographic_columns()).fit(table_with_chunks)
        
        # Force CPU for LEACE in laptop mode (numerical stability + memory)
        leace_force_cpu = bool(self._cfg_get(config, "leace.force_cpu"))
        if leace_force_cpu:
            logger.info("LEACE computation forced to CPU for numerical stability")
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device="cpu" if leace_force_cpu else self._resolve_device(self._cfg_get(config, "encoder.device")),
            force_cpu=leace_force_cpu,
        )
        
        # Laptop mode: smaller shards for embedding + LEACE accumulation
        shard_size = int(chunking_cfg.get("shard_size", 0) or 512)  # Smaller default for laptop
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
                shard_table = table_with_chunks.slice(start, end - start)
                
                embeddings_shard = embedder.embed_texts(
                    shard_masked,
                    batch_size=encoder_batch_size,
                    show_progress=False,
                    output_device=encoder_output_device,
                )
                
                # Ensure on CPU for LEACE accumulation
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
                
                # Explicit cleanup for low RAM
                del embeddings_shard, embeddings_shard_cpu
                embed_pbar.update(1)
        finally:
            embed_pbar.close()
        
        # Force garbage collection after embedding
        gc.collect()
        
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
            "staged_execution": True,
            "stage1_skipped": stage1_skipped,
            "inference_completeness": {
                "total_docs": len(posts),
                "processed_docs": len(posts) - len(final_skipped),
                "skipped_docs": len(final_skipped),
                "oom_count": inference_result.oom_count,
                "final_budget": inference_result.final_budget,
                "retry_rounds": retry_round,
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
