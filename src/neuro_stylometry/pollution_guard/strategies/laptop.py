"""
Laptop Strategy for Phase A Pollution Filtering.

Optimized for:
- Low VRAM (4-8GB)
- CPU fallback
- Batch accumulation for LEACE
- Small dataset subsets

Implements: phaseA-D_implementation_plan.md Section 7.3
"""

import logging
import torch
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, Any, List
from tqdm import tqdm
import json
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


class LaptopFilterStrategy(PollutionFilterStrategy):
    """
    Laptop-optimized pollution filtering strategy.
    
    Key features:
    - Uses CPU or low-VRAM GPU
    - Batch accumulation for LEACE covariance
    - Small batch sizes (2-4)
    - Progress bars for user feedback
    
    Implements: FR-11 (Laptop Mode)
    """
    
    def __init__(self):
        """Initialize laptop strategy.

        Note: Under strict YAML authority, this strategy does not set device or batch size.
        Those are read from the provided config.
        """
        self._last_device: str = "cpu"
        self._last_batch_size: int = 0
        logger.info("LaptopFilterStrategy initialized")
    
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
        Execute Phase A on laptop.
        
        Pipeline:
        1. Load dataset (use sobr_laptop.arrow subset)
        2. GLiNER pollution detection + masking
        3. Embed masked texts
        4. Compute LEACE projection (with batch accumulation)
        5. Save outputs
        """
        logger.info("=" * 80)
        logger.info("Phase A: Laptop Strategy")
        logger.info("=" * 80)

        device = self._resolve_device(self._cfg_get(config, "gliner.device"))
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
        # Laptop mode: Extract on GPU, then move to CPU for LEACE accumulation
        logger.info("Step 2/3: Embedding Masked Texts")
        
        # Determine output device for embeddings (config-driven, default to CPU for laptop)
        encoder_output_device = "cpu"
        try:
            encoder_output_device = str(self._cfg_get(config, "encoder.output_device"))
        except KeyError:
            pass  # Use default "cpu"
        
        embedder = FrozenEmbedder(
            model_name=str(self._cfg_get(config, "encoder.model")),
            device=self._resolve_device(self._cfg_get(config, "encoder.device")),
            max_length=int(self._cfg_get(config, "encoder.max_length")),
            output_device=encoder_output_device,
        )
        
        logger.info(f"  Laptop mode: embeddings extracted on GPU → moving to {encoder_output_device} for LEACE accumulation")
        
        embeddings = embedder.embed_texts(
            masked_texts,
            batch_size=encoder_batch_size,
            show_progress=True,
            output_device=encoder_output_device,
        )
        # Ensure embeddings are on CPU for LEACE (explicit enforcement)
        embeddings_cpu = embeddings.detach().cpu().to(dtype=torch.float32)
        
        # Step 4: Compute LEACE projection (with accumulation)
        # Laptop mode: All LEACE computation on CPU for numerical stability
        logger.info("Step 3/3: Computing LEACE Projection")

        encoder = DemographicEncoder(get_demographic_columns()).fit(table)
        
        # Force CPU for LEACE in laptop mode
        leace_force_cpu = bool(self._cfg_get(config, "leace.force_cpu"))
        if leace_force_cpu:
            logger.info("  Laptop mode: LEACE computation forced to CPU for numerical stability")
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=float(self._cfg_get(config, "leace.regularization")),
            device="cpu" if leace_force_cpu else self._resolve_device(self._cfg_get(config, "encoder.device")),
            force_cpu=leace_force_cpu,
        )
        
        # Batch accumulation for low memory
        logger.info("  Accumulating covariance statistics...")
        accumulated_stats = None
        
        leace_batch_size = int(self._cfg_get(config, "leace.batch_size"))
        for i in tqdm(
            range(0, len(embeddings_cpu), leace_batch_size),
            desc="LEACE accumulation",
            unit="batch",
        ):
            batch_embeddings = embeddings_cpu[i:i + leace_batch_size]

            table_slice = table.slice(i, min(leace_batch_size, len(table) - i))
            concepts_np = encoder.transform(table_slice)
            batch_concepts = torch.from_numpy(concepts_np)
            
            accumulated_stats = leace.accumulate_batch_concepts(
                batch_embeddings,
                batch_concepts,
                accumulated_stats,
            )
        
        # Compute projection from accumulated stats
        logger.info("  Computing projection matrix...")
        projection_matrix = leace.compute_projection_from_concept_stats(accumulated_stats)
        
        # Step 5: Save outputs
        logger.info("Saving outputs...")
        
        # Save cleaned dataset
        self._save_cleaned_dataset(
            table, masked_texts, output_dataset_path
        )
        
        # Save projection matrix
        torch.save(projection_matrix, projection_matrix_path)
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
        """
        Save cleaned dataset with post_masked column populated.
        """
        import pandas as pd
        
        # Convert to pandas, update post_masked, convert back
        df = original_table.to_pandas()
        df["post_masked"] = masked_texts
        
        # Convert back to Arrow and save
        cleaned_table = pa.Table.from_pandas(df, schema=SOBR_SCHEMA)
        feather.write_feather(cleaned_table, output_path)
        
        logger.info(f"  Saved cleaned dataset: {output_path}")
    
    def _save_pollution_logs(
        self,
        pollution_logs: list,
        output_path: Path,
    ) -> None:
        """
        Save pollution logs as Arrow table.
        """
        if not pollution_logs:
            logger.warning("  No pollution logs to save")
            # Save empty table
            empty_table = pa.Table.from_pylist([], schema=POLLUTION_LOG_SCHEMA)
            feather.write_feather(empty_table, output_path)
            return
        
        # Convert to Arrow table
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
# LaptopFilterStrategy - Laptop implementation with mini-batch accumulation
