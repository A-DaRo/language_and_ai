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
import torch
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, Any
import pandas as pd

from .base import PollutionFilterStrategy
from ..gliner_detector import GLiNERDetector
from ..masker import SpanMasker
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
        gliner = GLiNERDetector(
            model_name=str(self._cfg_get(config, "gliner.model")),
            device=device,
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            confidence_threshold=float(self._cfg_get(config, "gliner.confidence_threshold")),
            taxonomy_config=taxonomy_config,
            constraints_config=constraints_config,
        )
        
        # Detect spans
        logger.info("  Detecting pollution spans...")
        entities_batch = gliner.detect_spans_long(
            posts,
            batch_size=gliner_batch_size,
            show_progress=True,
        )
        
        # Mask spans
        logger.info("  Applying typed masks...")
        masker = SpanMasker(
            tokenizer=gliner.model.data_processor.transformer_tokenizer,
            entity_to_mask=gliner.get_mask_tokens(),
        )
        masked_texts, pollution_logs = masker.mask_batch(posts, entities_batch, post_ids)
        
        total_spans = sum(len(entities) for entities in entities_batch)
        logger.info(f"  Masked {total_spans} pollution spans")
        
        # Step 3: Embed masked texts
        logger.info("Step 2/3: Embedding Masked Texts")
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
        )
        
        embeddings = embedder.embed_texts(
            masked_texts,
            batch_size=encoder_batch_size,
            show_progress=True,
        )
        embeddings_cpu = embeddings.detach().to("cpu", dtype=torch.float32)
        
        # Move embeddings to GPU for full-batch LEACE
        embeddings = embeddings.to(device)
        
        # Step 4: Compute LEACE projection (full-batch)
        logger.info("Step 3/3: Computing LEACE Projection (Full-Batch GPU)")

        encoder = DemographicEncoder(get_demographic_columns()).fit(table)
        concepts = torch.from_numpy(encoder.transform(table)).to(device, dtype=torch.float32)
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            force_cpu=bool(self._cfg_get(config, "leace.force_cpu")),
        )
        
        # Full-batch computation (no accumulation needed)
        projection_matrix = leace.compute_projection_from_concepts(embeddings, concepts)
        
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

        if bool(self._cfg_get(config, "gliner.compute_explicit_recall")):
            recall = compute_explicit_recall(
                posts,
                entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall

        if bool(self._cfg_get(config, "probe.compute_amnesic_drop")):
            max_samples = self._cfg_get(config, "probe.max_samples")
            if max_samples and len(embeddings_cpu) > max_samples:
                embeddings_before = embeddings_cpu[:max_samples]
                probe_table = table.slice(0, int(max_samples))
            else:
                embeddings_before = embeddings_cpu
                probe_table = table

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
