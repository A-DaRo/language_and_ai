"""
HPC Strategy for Phase A Pollution Filtering.

Optimized for:
- High VRAM (40-80GB A100/H100)
- Full-batch GPU computation
- BF16/TF32 precision
- Large dataset processing

Implements: phaseA-D_implementation_plan.md Section 7.4
"""

import logging
import re
import torch
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, Any
import pandas as pd
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
from ...data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA, get_demographic_columns

logger = logging.getLogger(__name__)


class HPCFilterStrategy(PollutionFilterStrategy):
    """
    HPC-optimized pollution filtering strategy.
    
    Key features:
    - Full GPU utilization (A100/H100)
    - Larger batch sizes (16-32)
    - Full-batch LEACE computation
    - BF16 mixed precision (if available)
    
    Implements: FR-12 (HPC Mode)
    """
    
    def __init__(self):
        """Initialize HPC strategy.

        Under strict YAML authority, this strategy does not set batch size.
        Device is expected to be explicitly configured (typically "cuda").
        """
        self._last_device: str = "cuda"
        self._last_batch_size: int = 0
        self.use_bf16 = torch.cuda.is_available() and torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("BF16 mixed precision available")
        logger.info("HPCFilterStrategy initialized")
    
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self._last_device)
    
    def get_batch_size(self) -> int:
        """Get batch size."""
        return self._last_batch_size

    def _cfg_get(self, config: Dict[str, Any], path: str) -> Any:
        cur: Any = config
        for part in path.split("."):
            if not isinstance(cur, dict) or part not in cur:
                raise KeyError(f"Missing required config key: '{path}'")
            cur = cur[part]
        return cur

    def _resolve_device(self, device_spec: str) -> str:
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
        Execute Phase A on HPC.
        
        Pipeline:
        1. Load full dataset
        2. GLiNER pollution detection + masking
        3. Embed masked texts
        4. Compute LEACE projection (full-batch GPU)
        5. Save outputs
        """
        logger.info("=" * 80)
        logger.info("Phase A: HPC Strategy")
        logger.info("=" * 80)

        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
        if device != "cuda":
            raise RuntimeError("HPC strategy requires config gliner.device='cuda'")

        gliner_batch_size = int(self._cfg_get(config, "gliner.batch_size"))
        encoder_batch_size = int(self._cfg_get(config, "encoder.batch_size"))

        self._last_device = device
        self._last_batch_size = gliner_batch_size
        
        # Step 1: Load dataset
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(arrow_path=input_dataset_path, seed=int(self._cfg_get(config, "seed")))
        table = dataset.table

        # Optional subset (strictly config-driven)
        if bool(self._cfg_get(config, "subset.enabled")):
            size = self._cfg_get(config, "subset.size")
            if size is not None and len(table) > int(size):
                import numpy as np

                rng = np.random.default_rng(int(self._cfg_get(config, "seed")))
                indices = rng.choice(len(table), size=int(size), replace=False)
                logger.info(f"Using subset (seeded): {int(size)}/{len(table)} samples")
                table = table.take(pa.array(indices, type=pa.int64()))
        
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        
        logger.info(f"Loaded {len(posts)} posts")
        
        # Step 2: GLiNER detection + masking
        logger.info("Step 1/3: Pollution Detection & Masking")
        taxonomy_config, constraints_config = self._load_taxonomy_config(config)
        gliner_cfg = config.get("gliner", {})
        batch_inference_cfg = gliner_cfg.get("batch_inference", {})
        batch_config = BatchInferenceConfig(
            enable_batching=batch_inference_cfg.get("enable_batching", True),
            batch_size=int(self._cfg_get(config, "gliner.batch_size")),
            num_buckets=batch_inference_cfg.get("num_buckets"),
            min_bucket_size=batch_inference_cfg.get("min_bucket_size", 4),
            enable_prompt_caching=batch_inference_cfg.get("enable_prompt_caching", True),
        )
        chunking_cfg = gliner_cfg.get("chunking", {})
        chunk_cache_path = chunking_cfg.get("checkpoint_path") or chunking_cfg.get(
            "chunk_cache_path"
        )
        budget_config = BudgetConfig(
            model_max_length=int(self._cfg_get(config, "encoder.max_length")),
            mode=chunking_cfg.get("mode", "single_sentence"),
            legacy_sequential_mode=chunking_cfg.get("legacy_sequential_mode", False),
            parallel_chunking_workers=int(chunking_cfg.get("parallel_chunking_workers", 0)),
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
            chunk_cache_path=str(chunk_cache_path) if chunk_cache_path else None,
        )
        
        masker = SpanMasker(
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
            entity_to_mask=gliner.get_mask_tokens(),
        )

        shard_size = int(chunking_cfg.get("shard_size", 0) or 0)
        if shard_size <= 0:
            shard_size = len(posts)
        total_shards = max(1, (len(posts) + shard_size - 1) // shard_size)
        if total_shards > 1:
            logger.info(
                "Sharding enabled: %d posts per shard (%d shards total)",
                shard_size,
                total_shards,
            )

        inference_pbar = None
        if total_shards > 1:
            inference_pbar = tqdm(
                total=0,
                desc="GLiNER inference (total)",
                unit="batch",
                dynamic_ncols=True,
            )

        def _inference_total_update(count: int) -> None:
            if inference_pbar is None:
                return
            if inference_pbar.total is None:
                inference_pbar.total = 0
            inference_pbar.total += int(count)
            inference_pbar.refresh()

        def _inference_update(count: int) -> None:
            if inference_pbar is None:
                return
            inference_pbar.update(int(count))

        chunk_cache_base = Path(chunk_cache_path) if chunk_cache_path else None
        if total_shards > 1 and chunk_cache_base is not None:
            logger.info("Chunk cache base: %s", chunk_cache_base)

        compute_recall = bool(self._cfg_get(config, "gliner.compute_explicit_recall"))
        reference_patterns = (
            gliner.taxonomy.get_reference_patterns() if compute_recall else {}
        )
        total_ref = 0
        matched_ref = 0
        per_label_counts: Dict[str, Dict[str, int]] = {}

        def _update_recall_stats(texts: list, entities: list) -> None:
            nonlocal total_ref, matched_ref
            for text, entity_list in zip(texts, entities):
                ref_spans = []
                for label, patterns in reference_patterns.items():
                    for pattern in patterns:
                        for match in re.finditer(pattern, text):
                            ref_spans.append((match.start(), match.end(), label))

                total_ref += len(ref_spans)
                for start, end, label in ref_spans:
                    matched = False
                    for entity in entity_list:
                        if entity.get("label") != label:
                            continue
                        if max(start, entity["start"]) < min(end, entity["end"]):
                            matched = True
                            break
                    counts = per_label_counts.setdefault(
                        label,
                        {"matched": 0, "total": 0},
                    )
                    if matched:
                        matched_ref += 1
                        counts["matched"] += 1
                    counts["total"] += 1

        total_spans = 0
        masked_texts: list = []
        pollution_logs: list = []

        # Step 3: Embed masked texts
        # Critical (Section 5.1 of LEACE Strategy Report):
        # - LEACE MUST be computed on embeddings of MASKED text (post_masked), NOT raw text.
        # - FrozenEmbedder MUST register the same typed mask tokens as GLiNERDetector.
        logger.info("Step 2/3: Embedding Masked Texts")

        # Extract mask tokens for tokenizer alignment (Section 5.1)
        mask_tokens = list(dict.fromkeys(gliner.get_mask_tokens().values()))

        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            special_tokens=mask_tokens,  # Critical for tokenizer alignment
        )

        encoder = DemographicEncoder(get_demographic_columns()).fit(table)

        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            force_cpu=bool(self._cfg_get(config, "leace.force_cpu")),
        )

        probe_enabled = bool(self._cfg_get(config, "probe.compute_amnesic_drop"))
        max_probe_samples = (
            int(self._cfg_get(config, "probe.max_samples")) if probe_enabled else 0
        )
        probe_embeddings: list = []

        # Detect spans + mask + embed in shards
        logger.info("  Detecting pollution spans...")
        concept_stats = None
        for shard_idx in range(total_shards):
            start = shard_idx * shard_size
            end = min(len(posts), start + shard_size)
            shard_len = end - start
            if shard_len <= 0:
                continue

            logger.info(
                "  Shard %d/%d: rows %d-%d",
                shard_idx + 1,
                total_shards,
                start,
                end - 1,
            )

            shard_table = table.slice(start, shard_len)
            shard_posts = shard_table["post"].to_pylist()
            shard_post_ids = shard_table["post_id"].to_pylist()

            if total_shards > 1 and chunk_cache_base is not None:
                shard_cache = chunk_cache_base.with_name(
                    f"{chunk_cache_base.stem}.shard{shard_idx:04d}{chunk_cache_base.suffix}"
                )
                gliner.chunk_cache_path = shard_cache

            entities_shard = gliner.detect_spans_long(
                shard_posts,
                batch_size=gliner_batch_size,
                show_progress=total_shards == 1,
                inference_progress_callback=_inference_update if total_shards > 1 else None,
                inference_total_callback=_inference_total_update if total_shards > 1 else None,
            )

            masked_shard, logs_shard = masker.mask_batch(
                shard_posts,
                entities_shard,
                shard_post_ids,
            )

            masked_texts.extend(masked_shard)
            pollution_logs.extend(logs_shard)
            total_spans += sum(len(entities) for entities in entities_shard)

            if compute_recall:
                _update_recall_stats(shard_posts, entities_shard)

            embeddings_shard = embedder.embed_texts(
                masked_shard,
                batch_size=encoder_batch_size,
                show_progress=True,
            )

            if probe_enabled and len(probe_embeddings) < max_probe_samples:
                remaining = max_probe_samples - len(probe_embeddings)
                if remaining > 0:
                    embeddings_cpu = embeddings_shard.detach().to(
                        "cpu", dtype=torch.float32
                    )
                    probe_embeddings.append(embeddings_cpu[:remaining])

            concepts_shard = torch.from_numpy(encoder.transform(shard_table))
            concept_stats = leace.accumulate_batch_concepts(
                embeddings_shard,
                concepts_shard,
                concept_stats,
            )
            del embeddings_shard

        if inference_pbar is not None:
            inference_pbar.close()

        if concept_stats is None:
            raise RuntimeError("LEACE accumulation failed: no concept statistics computed")

        logger.info("  Masked %d pollution spans", total_spans)

        # Step 4: Compute LEACE projection from accumulated stats
        logger.info("Step 3/3: Computing LEACE Projection (Accumulated)")
        projection_matrix = leace.compute_projection_from_concept_stats(concept_stats)
        
        # Step 5: Save outputs
        logger.info("Saving outputs...")
        
        # Save cleaned dataset
        self._save_cleaned_dataset(
            table, masked_texts, output_dataset_path
        )
        
        # Save projection matrix
        torch.save(projection_matrix.cpu(), projection_matrix_path)
        logger.info(f"  Saved projection matrix: {projection_matrix_path}")
        
        # Save pollution logs
        self._save_pollution_logs(pollution_logs, pollution_logs_path)
        logger.info(f"  Saved pollution logs: {pollution_logs_path}")
        
        # Return metadata
        metadata = {
            "num_samples": len(table),
            "num_pollution_spans": total_spans,
            "projection_matrix_shape": list(projection_matrix.shape),
            "device": device,
            "use_bf16": self.use_bf16,
        }

        if compute_recall:
            overall = (matched_ref / total_ref) if total_ref > 0 else None
            per_label_recall = {
                label: (counts["matched"] / counts["total"])
                if counts["total"] > 0
                else None
                for label, counts in per_label_counts.items()
            }
            metadata["explicit_recall"] = {
                "overall": overall,
                "total_reference_spans": total_ref,
                "matched_reference_spans": matched_ref,
                "per_label": per_label_recall,
            }

        if probe_enabled:
            if probe_embeddings:
                embeddings_before = torch.cat(probe_embeddings, dim=0)
            else:
                embeddings_before = torch.empty((0, embedder.get_embedding_dim()))

            if embeddings_before.numel() == 0:
                logger.warning("Probe skipped: no embeddings collected")
            else:
                max_samples = self._cfg_get(config, "probe.max_samples")
                if max_samples and len(embeddings_before) > max_samples:
                    embeddings_before = embeddings_before[:max_samples]
                    probe_table = table.slice(0, int(max_samples))
                else:
                    probe_table = table.slice(0, int(len(embeddings_before)))

                P_cpu = projection_matrix.detach().to("cpu", dtype=torch.float32)
                embeddings_after = embeddings_before @ P_cpu.T

                by_column: Dict[str, Any] = {}
                drops = []
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
                    "max_samples": int(len(embeddings_before)),
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
        masked_texts: list,
        output_path: Path,
    ) -> None:
        """Save cleaned dataset with post_masked column populated."""
        df = original_table.to_pandas()
        df["post_masked"] = masked_texts
        
        cleaned_table = pa.Table.from_pandas(df, schema=SOBR_SCHEMA)
        feather.write_feather(cleaned_table, output_path)
        
        logger.info(f"  Saved cleaned dataset: {output_path}")
    
    def _save_pollution_logs(
        self,
        pollution_logs: list,
        output_path: Path,
    ) -> None:
        """Save pollution logs as Arrow table."""
        if not pollution_logs:
            logger.warning("  No pollution logs to save")
            empty_table = pa.Table.from_pylist([], schema=POLLUTION_LOG_SCHEMA)
            feather.write_feather(empty_table, output_path)
            return
        
        logs_table = pa.Table.from_pylist(pollution_logs, schema=POLLUTION_LOG_SCHEMA)
        feather.write_feather(logs_table, output_path)
        
        logger.info(f"  Saved {len(pollution_logs)} pollution log entries")

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
# HPCFilterStrategy - HPC implementation with CUDA Graphs and full-batch LEACE
