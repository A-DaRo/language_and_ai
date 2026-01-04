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
from typing import Dict, Any
from tqdm import tqdm
import json
import pandas as pd

from .base import PollutionFilterStrategy
from ..gliner_detector import GLiNERDetector
from ..masker import SpanMasker
from ..explicit_recall import compute_explicit_recall
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer, CovarianceStats
from ..probe import compute_amnesic_drop
from ...data_engine.dataset import SOBRDataset
from ...data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA

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
        """Initialize laptop strategy."""
        self.device = "cuda" if torch.cuda.is_available() else "cpu"
        self.batch_size = 2  # Small batch for low VRAM
        logger.info(f"LaptopFilterStrategy initialized: device={self.device}")
    
    def get_device(self) -> torch.device:
        """Get PyTorch device."""
        return torch.device(self.device)
    
    def get_batch_size(self) -> int:
        """Get batch size."""
        return self.batch_size
    
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
        
        # Step 1: Load dataset
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(arrow_path=input_dataset_path, seed=config.get("seed", 42))
        table = dataset.table
        
        # Use subset for laptop (config can override)
        max_samples = config.get("max_samples", 1000)
        if len(table) > max_samples:
            logger.info(f"Using subset: {max_samples}/{len(table)} samples")
            table = table.slice(0, max_samples)
        
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        
        # Step 2: GLiNER detection + masking
        logger.info("Step 1/3: Pollution Detection & Masking")
        taxonomy_config, constraints_config = self._load_taxonomy_config(config)
        gliner = GLiNERDetector(
            device=self.device,
            max_length=512,
            confidence_threshold=config.get("gliner_threshold", 0.85),
            taxonomy_config=taxonomy_config,
            constraints_config=constraints_config,
        )
        
        # Detect spans
        logger.info("  Detecting pollution spans...")
        entities_batch = gliner.detect_spans_long(
            posts,
            batch_size=self.batch_size,
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
            model_name=config.get("embedder_model", "roberta-base"),
            device=self.device,
            max_length=512,
        )
        
        embeddings = embedder.embed_texts(
            masked_texts,
            batch_size=self.batch_size,
            show_progress=True,
        )
        embeddings_cpu = embeddings.detach().to("cpu", dtype=torch.float32)
        
        # Step 4: Compute LEACE projection (with accumulation)
        logger.info("Step 3/3: Computing LEACE Projection")
        
        # Get labels for projection (use first non-null demographic)
        labels = self._extract_labels(table, config)
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=config.get("leace_regularization", 1e-5),
            device=self.device,
            force_cpu=config.get("leace_force_cpu", True),
        )
        
        # Batch accumulation for low memory
        logger.info("  Accumulating covariance statistics...")
        accumulated_stats = None
        
        leace_batch_size = config.get("leace_batch_size", 100)
        for i in tqdm(
            range(0, len(embeddings), leace_batch_size),
            desc="LEACE accumulation",
            unit="batch",
        ):
            batch_embeddings = embeddings[i:i + leace_batch_size]
            batch_labels = labels[i:i + leace_batch_size]
            
            accumulated_stats = leace.accumulate_batch(
                batch_embeddings,
                batch_labels,
                accumulated_stats,
            )
        
        # Compute projection from accumulated stats
        logger.info("  Computing projection matrix...")
        projection_matrix = leace.compute_projection_from_stats(accumulated_stats)
        
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
            "device": self.device,
        }

        if config.get("gliner_compute_explicit_recall", False):
            recall = compute_explicit_recall(
                posts,
                entities_batch,
                gliner.taxonomy.get_reference_patterns(),
            )
            metadata["explicit_recall"] = recall

        if config.get("probe_compute_amnesic_drop", False):
            labels = self._extract_labels(table, config)
            max_samples = config.get("probe_max_samples")
            if max_samples and len(embeddings_cpu) > max_samples:
                embeddings_before = embeddings_cpu[:max_samples]
                labels_probe = labels[:max_samples]
            else:
                embeddings_before = embeddings_cpu
                labels_probe = labels

            P_cpu = projection_matrix.detach().to("cpu", dtype=torch.float32)
            embeddings_after = embeddings_before @ P_cpu.T
            acc_before, acc_after, amnesic_drop = compute_amnesic_drop(
                embeddings_before,
                embeddings_after,
                labels_probe,
                train_split=config.get("probe_train_split", 0.8),
                random_state=config.get("seed", 42),
            )
            metadata["probe"] = {
                "accuracy_before": acc_before,
                "accuracy_after": acc_after,
                "amnesic_drop": amnesic_drop,
                "max_samples": int(len(embeddings_before)),
            }
        
        logger.info("Phase A complete!")
        return metadata
    
    def _extract_labels(self, table: pa.Table, config: Dict[str, Any]) -> torch.Tensor:
        """
        Extract demographic labels for LEACE projection.
        
        Uses first non-null demographic column as specified in config.
        """
        label_column = config.get("projection_label", "nationality")
        
        if label_column not in table.column_names:
            raise ValueError(f"Label column '{label_column}' not found in table")
        
        labels_series = table[label_column].to_pandas()
        
        # Handle different label types
        if labels_series.dtype == 'object':  # String labels
            # Factorize string labels to integers
            labels_int, _ = pd.factorize(labels_series.dropna())
            # Fill NaNs with -1
            labels = labels_series.copy()
            labels[labels.notna()] = labels_int
            labels = labels.fillna(-1).astype(int)
        else:  # Numeric labels
            labels = labels_series.fillna(-1).astype(int)
        
        return torch.tensor(labels.values, dtype=torch.long)
    
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
        taxonomy_path = config.get("gliner_taxonomy_path")
        if not taxonomy_path:
            return None, None

        from omegaconf import OmegaConf

        path = Path(taxonomy_path)
        if not path.is_absolute():
            repo_root = Path(__file__).resolve().parents[4]
            path = repo_root / path

        if not path.exists():
            logger.warning(f"GLiNER taxonomy config not found: {path}")
            return None, None

        cfg = OmegaConf.to_container(OmegaConf.load(path), resolve=True)
        taxonomy_cfg = cfg.get("taxonomy", {})
        constraints_cfg = taxonomy_cfg.get("width_constraints", {})
        return taxonomy_cfg, constraints_cfg
# LaptopFilterStrategy - Laptop implementation with mini-batch accumulation
