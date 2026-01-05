#!/usr/bin/env python
"""
Generate golden test fixtures for determinism testing.

This script generates:
1. A small SOBR dataset subset with fixed seed for reproducibility
2. Reference Phase A artifacts (projection matrices, clean datasets, pollution logs)
   for both laptop and HPC modes

Usage:
    python scripts/generate_golden_fixtures.py --output-dir tests/golden/fixtures --seed 42

The generated fixtures can be used for:
- Determinism verification tests (identical artifacts on rerun)
- Cross-mode equivalence tests (laptop vs HPC projection correlation)
- Golden reference validation tests
"""

import argparse
import logging
import sys
from pathlib import Path

import numpy as np
import pyarrow as pa
import torch

# Add src to path for development
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro_stylometry.data_engine.dataset import SOBRDataset
from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA
from neuro_stylometry.hardware_ops.detection import ProfileType

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s - %(name)s - %(levelname)s - %(message)s",
)
logger = logging.getLogger(__name__)


def set_all_seeds(seed: int) -> None:
    """Set all random seeds for reproducibility."""
    import random
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed_all(seed)
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False


def generate_small_dataset(
    source_dataset_path: Path,
    output_path: Path,
    num_authors: int = 100,
    max_posts_per_author: int = 50,
    seed: int = 42,
) -> None:
    """
    Generate a small dataset subset for golden tests.
    
    Args:
        source_dataset_path: Path to source SOBR Arrow file.
        output_path: Path to write small dataset.
        num_authors: Number of authors to include.
        max_posts_per_author: Maximum posts per author.
        seed: Random seed for reproducibility.
    """
    set_all_seeds(seed)
    
    logger.info(f"Loading source dataset: {source_dataset_path}")
    dataset = SOBRDataset(arrow_path=source_dataset_path, seed=seed)
    table = dataset.table
    
    # Get unique authors
    author_ids = table["author_id"].to_pandas().unique()
    logger.info(f"Source dataset has {len(author_ids)} unique authors")
    
    # Sample authors
    rng = np.random.default_rng(seed)
    sampled_authors = rng.choice(
        author_ids, 
        size=min(num_authors, len(author_ids)), 
        replace=False
    )
    logger.info(f"Sampled {len(sampled_authors)} authors")
    
    # Filter table to sampled authors
    author_mask = pa.compute.is_in(
        table["author_id"],
        pa.array(sampled_authors)
    )
    filtered_table = table.filter(author_mask)
    
    # Further sample posts per author if needed
    if max_posts_per_author is not None:
        # Group by author and limit posts
        df = filtered_table.to_pandas()
        sampled_df = df.groupby("author_id", group_keys=False).apply(
            lambda x: x.sample(
                n=min(len(x), max_posts_per_author),
                random_state=seed
            )
        )
        filtered_table = pa.Table.from_pandas(sampled_df, schema=SOBR_SCHEMA)
    
    logger.info(f"Final dataset: {len(filtered_table)} posts from {len(sampled_authors)} authors")
    
    # Save to Arrow
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    import pyarrow.feather as feather
    feather.write_feather(filtered_table, output_path)
    logger.info(f"Saved golden dataset: {output_path}")


def run_phase_a_for_golden(
    dataset_path: Path,
    output_dir: Path,
    mode: str,
    seed: int = 42,
) -> dict:
    """
    Run Phase A pipeline and save artifacts for golden tests.
    
    Args:
        dataset_path: Path to input dataset.
        output_dir: Output directory for artifacts.
        mode: "laptop" or "hpc".
        seed: Random seed.
        
    Returns:
        Metadata dict from pipeline execution.
    """
    set_all_seeds(seed)
    
    from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
    from neuro_stylometry.factories.strategy_factory import StrategyFactory
    from neuro_stylometry.config import load_pipeline_config
    
    logger.info(f"Running Phase A in {mode} mode with seed={seed}")

    if mode == "laptop":
        profile_type = ProfileType.LAPTOP
    else:
        profile_type = ProfileType.HPC
    
    # Create strategy
    strategy = StrategyFactory.create_filter_strategy(profile_type)
    
    # Load config
    config = load_pipeline_config(mode=mode)
    
    # Override seed
    config["seed"] = seed
    
    # Create pipeline
    pipeline = PhaseAPipeline(strategy=strategy, config=config)
    
    # Execute
    artifacts = pipeline.run(
        input_dataset_path=dataset_path,
        output_dir=output_dir,
    )
    
    logger.info(f"Phase A complete for {mode} mode:")
    logger.info(f"  Clean dataset: {artifacts.clean_dataset_path}")
    logger.info(f"  Projection matrix: {artifacts.projection_matrix_path}")
    logger.info(f"  Pollution logs: {artifacts.pollution_logs_path}")
    
    return artifacts.metadata


def main():
    parser = argparse.ArgumentParser(
        description="Generate golden test fixtures for determinism testing"
    )
    parser.add_argument(
        "--source-dataset",
        type=Path,
        default=Path("artifacts/data/sobr_laptop.arrow"),
        help="Source dataset Arrow file",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("tests/golden/fixtures"),
        help="Output directory for golden fixtures",
    )
    parser.add_argument(
        "--seed",
        type=int,
        default=42,
        help="Random seed for reproducibility",
    )
    parser.add_argument(
        "--num-authors",
        type=int,
        default=100,
        help="Number of authors to include in golden dataset",
    )
    parser.add_argument(
        "--skip-phase-a",
        action="store_true",
        help="Skip Phase A execution (only generate dataset)",
    )
    parser.add_argument(
        "--modes",
        nargs="+",
        default=["laptop"],
        choices=["laptop", "hpc"],
        help="Modes to generate golden fixtures for",
    )
    
    args = parser.parse_args()
    
    # Ensure output directory exists
    args.output_dir.mkdir(parents=True, exist_ok=True)
    
    # Step 1: Generate small dataset
    golden_dataset_path = args.output_dir / f"sobr_small_seed{args.seed}.arrow"
    
    if not golden_dataset_path.exists():
        if not args.source_dataset.exists():
            logger.error(f"Source dataset not found: {args.source_dataset}")
            logger.info("Run: python scripts/create_laptop_dataset.py first")
            sys.exit(1)
        
        generate_small_dataset(
            source_dataset_path=args.source_dataset,
            output_path=golden_dataset_path,
            num_authors=args.num_authors,
            seed=args.seed,
        )
    else:
        logger.info(f"Golden dataset already exists: {golden_dataset_path}")
    
    if args.skip_phase_a:
        logger.info("Skipping Phase A execution (--skip-phase-a)")
        return
    
    # Step 2: Run Phase A for each mode
    for mode in args.modes:
        mode_output_dir = args.output_dir / f"phase_a_{mode}_seed{args.seed}"
        mode_output_dir.mkdir(parents=True, exist_ok=True)
        
        try:
            metadata = run_phase_a_for_golden(
                dataset_path=golden_dataset_path,
                output_dir=mode_output_dir,
                mode=mode,
                seed=args.seed,
            )
            
            # Copy projection matrix with standardized name
            src_P = mode_output_dir / "projection_matrix.pt"
            dst_P = args.output_dir / f"expected_P_{mode}_seed{args.seed}.pt"
            if src_P.exists():
                import shutil
                shutil.copy(src_P, dst_P)
                logger.info(f"Copied projection matrix: {dst_P}")
                
        except Exception as e:
            logger.error(f"Failed to run Phase A in {mode} mode: {e}")
            import traceback
            traceback.print_exc()
    
    logger.info("Golden fixture generation complete!")
    logger.info(f"Output directory: {args.output_dir}")


if __name__ == "__main__":
    main()
