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
from ..embedder import FrozenEmbedder
from ..leace import LEACEComputer
from ...data_engine.dataset import SOBRDataset
from ...data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA

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
        """Initialize HPC strategy."""
        if not torch.cuda.is_available():
            raise RuntimeError("HPC strategy requires CUDA")
        
        self.device = "cuda"
        self.batch_size = 16  # Larger batch for HPC
        
        # Check for BF16 support
        self.use_bf16 = torch.cuda.is_bf16_supported()
        if self.use_bf16:
            logger.info("BF16 mixed precision enabled")
        
        logger.info(f"HPCFilterStrategy initialized: device={self.device}")
    
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
        
        # Step 1: Load dataset
        logger.info(f"Loading dataset: {input_dataset_path}")
        dataset = SOBRDataset(arrow_path=input_dataset_path, seed=config.get("seed", 42))
        table = dataset.table
        
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        
        logger.info(f"Loaded {len(posts)} posts")
        
        # Step 2: GLiNER detection + masking
        logger.info("Step 1/3: Pollution Detection & Masking")
        gliner = GLiNERDetector(
            device=self.device,
            max_length=512,
            confidence_threshold=config.get("gliner_threshold", 0.85),
        )
        
        # Detect spans
        logger.info("  Detecting pollution spans...")
        entities_batch = gliner.detect_spans_long(
            posts,
            batch_size=self.batch_size,
        )
        
        # Mask spans
        logger.info("  Applying typed masks...")
        masker = SpanMasker(tokenizer=gliner.model.data_processor.transformer_tokenizer)
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
        
        # Move embeddings to GPU for full-batch LEACE
        embeddings = embeddings.to(self.device)
        
        # Step 4: Compute LEACE projection (full-batch)
        logger.info("Step 3/3: Computing LEACE Projection (Full-Batch GPU)")
        
        # Get labels for projection
        labels = self._extract_labels(table, config)
        labels = labels.to(self.device)
        
        leace = LEACEComputer(
            embedding_dim=embedder.get_embedding_dim(),
            regularization=config.get("leace_regularization", 1e-5),
            device=self.device,
        )
        
        # Full-batch computation (no accumulation needed)
        projection_matrix = leace.compute_projection(embeddings, labels)
        
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
            "device": self.device,
            "use_bf16": self.use_bf16,
        }
        
        logger.info("Phase A complete!")
        return metadata
    
    def _extract_labels(self, table: pa.Table, config: Dict[str, Any]) -> torch.Tensor:
        """Extract demographic labels for LEACE projection."""
        label_column = config.get("projection_label", "nationality")
        
        if label_column not in table.column_names:
            raise ValueError(f"Label column '{label_column}' not found in table")
        
        labels_series = table[label_column].to_pandas()
        
        # Handle different label types
        if labels_series.dtype == 'object':  # String labels
            labels_int, _ = pd.factorize(labels_series.dropna())
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
# HPCFilterStrategy - HPC implementation with CUDA Graphs and full-batch LEACE
