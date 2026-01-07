#!/usr/bin/env python3
"""
Create Laptop-Mode Debug Dataset from Full SOBR Arrow File.

This script generates a lightweight, representative subset for development:
    - Author-centric sampling (preserves multi-post-per-author distribution)
    - Schema & encoding preservation (ensures structural fidelity)
    - Configurable target size (default: 10,000 posts)
    
The subset allows rapid iteration during development while ensuring that
code validated on the laptop transfers seamlessly to HPC environment.
"""

import argparse
import logging
import sys
from pathlib import Path
import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from collections import defaultdict

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, validate_schema


def setup_logging(verbose: bool = False):
    """Configure logging."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(levelname)s - %(message)s',
        handlers=[logging.StreamHandler(sys.stdout)]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Create laptop-mode debug dataset from full SOBR Arrow file",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Create 10k sample subset
    python scripts/create_laptop_dataset.py \\
        --input artifacts/data/sobr.arrow \\
        --output artifacts/data/sobr_laptop.arrow \\
        --target-size 10000
    
    # Create 5k sample subset with specific seed
    python scripts/create_laptop_dataset.py \\
        --input artifacts/data/sobr.arrow \\
        --output artifacts/data/sobr_laptop.arrow \\
        --target-size 5000 \\
        --seed 123
        """
    )
    
    parser.add_argument(
        '--input',
        type=Path,
        default=Path('artifacts/data/sobr.arrow'),
        help='Path to full SOBR Arrow file'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('artifacts/data/sobr_laptop.arrow'),
        help='Output path for laptop subset (default: artifacts/data/sobr_laptop.arrow)'
    )
    
    parser.add_argument(
        '--target-size',
        type=int,
        default=10000,
        help='Approximate number of posts in subset (default: 10000)'
    )
    
    parser.add_argument(
        '--seed',
        type=int,
        default=42,
        help='Random seed for reproducibility (default: 42)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging'
    )
    
    return parser.parse_args()


def load_full_dataset(input_path: Path) -> pa.Table:
    """
    Load the full Arrow dataset.
    
    Args:
        input_path: Path to full SOBR Arrow file.
        
    Returns:
        PyArrow Table.
    """
    logger = logging.getLogger(__name__)
    
    if not input_path.exists():
        raise FileNotFoundError(f"Input file not found: {input_path}")
    
    logger.info(f"Loading full dataset from {input_path}...")
    table = feather.read_table(input_path)
    
    # Validate schema
    validate_schema(table, SOBR_SCHEMA)
    logger.info(f"Loaded {len(table):,} rows")
    
    return table


def sample_authors_by_post_count(
    table: pa.Table,
    target_size: int,
    seed: int,
) -> list:
    """
    Sample authors (not rows) to achieve approximately target_size posts.
    
    This preserves the "multiple posts per author" distribution pattern.
    
    Args:
        table: Full PyArrow Table.
        target_size: Target number of posts.
        seed: Random seed.
        
    Returns:
        List of selected author indices (integer dictionary codes).
    """
    logger = logging.getLogger(__name__)
    
    # Extract author_id column and compute integer codes robustly
    author_col = table['author_id']
    author_series = author_col.to_pandas()
    author_indices = pd.factorize(author_series)[0]
    
    # Count posts per author
    author_post_counts = defaultdict(int)
    for author_idx in author_indices:
        author_post_counts[int(author_idx)] += 1
    
    # Convert to list of (author_idx, post_count) tuples
    authors = list(author_post_counts.items())
    logger.info(f"Total authors in dataset: {len(authors)}")
    logger.info(f"Avg posts per author: {np.mean([count for _, count in authors]):.1f}")
    
    # Shuffle authors
    rng = np.random.default_rng(seed)
    rng.shuffle(authors)
    
    # Greedily select authors until we reach target size
    selected_authors = []
    cumulative_posts = 0
    
    for author_idx, post_count in authors:
        selected_authors.append(author_idx)
        cumulative_posts += post_count
        
        if cumulative_posts >= target_size:
            break
    
    logger.info(f"Selected {len(selected_authors)} authors "
               f"→ {cumulative_posts:,} posts (target: {target_size:,})")
    
    return selected_authors


def extract_subset(
    table: pa.Table,
    selected_authors: list,
) -> pa.Table:
    """
    Extract rows corresponding to selected authors.
    
    Args:
        table: Full PyArrow Table.
        selected_authors: List of author dictionary indices.
        
    Returns:
        Subset PyArrow Table with same schema.
    """
    logger = logging.getLogger(__name__)
    
    # Get author_id column and compute integer codes
    author_col = table['author_id']
    author_series = author_col.to_pandas()
    author_indices = pd.factorize(author_series)[0]
    
    # Find row indices where author is in selected set
    selected_author_set = set([int(x) for x in selected_authors])
    row_mask = np.isin(author_indices, list(selected_author_set))
    
    # Extract subset
    subset_table = table.filter(pa.array(row_mask))
    
    logger.info(f"Extracted {len(subset_table):,} rows")
    
    # Validate schema preservation
    validate_schema(subset_table, SOBR_SCHEMA)
    
    return subset_table


def compute_statistics(full_table: pa.Table, subset_table: pa.Table) -> dict:
    """
    Compute comparison statistics between full and subset datasets.
    
    Returns:
        Dictionary with statistics.
    """
    def get_author_count(table):
        author_col = table['author_id']
        author_series = author_col.to_pandas()
        return len(np.unique(pd.factorize(author_series)[0]))
    
    full_authors = get_author_count(full_table)
    subset_authors = get_author_count(subset_table)
    
    full_posts = len(full_table)
    subset_posts = len(subset_table)
    
    return {
        'full_authors': full_authors,
        'full_posts': full_posts,
        'subset_authors': subset_authors,
        'subset_posts': subset_posts,
        'author_ratio': subset_authors / full_authors,
        'post_ratio': subset_posts / full_posts,
        'avg_posts_per_author_full': full_posts / full_authors,
        'avg_posts_per_author_subset': subset_posts / subset_authors,
    }


def print_statistics(stats: dict):
    """Pretty-print subset statistics."""
    print("\n" + "=" * 80)
    print("SUBSET CREATION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Full Dataset:")
    print(f"  • Authors: {stats['full_authors']:,}")
    print(f"  • Posts:   {stats['full_posts']:,}")
    print(f"  • Avg Posts/Author: {stats['avg_posts_per_author_full']:.1f}")
    
    print(f"\n📊 Laptop Subset:")
    print(f"  • Authors: {stats['subset_authors']:,}")
    print(f"  • Posts:   {stats['subset_posts']:,}")
    print(f"  • Avg Posts/Author: {stats['avg_posts_per_author_subset']:.1f}")
    
    print(f"\n📉 Reduction:")
    print(f"  • Author Ratio: {stats['author_ratio']:.1%}")
    print(f"  • Post Ratio:   {stats['post_ratio']:.1%}")
    
    print("\n✓ Subset preserves author-post distribution pattern")
    print("=" * 80 + "\n")


def main():
    """Main execution flow."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("Laptop Dataset Creation")
    logger.info("=" * 80)
    logger.info(f"Input:  {args.input.absolute()}")
    logger.info(f"Output: {args.output.absolute()}")
    logger.info(f"Target: {args.target_size:,} posts")
    logger.info(f"Seed:   {args.seed}")
    logger.info("")
    
    try:
        # Load full dataset
        full_table = load_full_dataset(args.input)
        
        # Sample authors
        logger.info("Sampling authors...")
        selected_authors = sample_authors_by_post_count(
            table=full_table,
            target_size=args.target_size,
            seed=args.seed
        )
        
        # Extract subset
        logger.info("Extracting subset...")
        subset_table = extract_subset(full_table, selected_authors)
        
        # Compute statistics
        stats = compute_statistics(full_table, subset_table)
        
        # Write subset
        logger.info(f"Writing subset to {args.output}...")
        args.output.parent.mkdir(parents=True, exist_ok=True)
        
        feather.write_feather(
            subset_table,
            args.output,
            compression='lz4'
        )
        
        file_size_mb = args.output.stat().st_size / (1024 ** 2)
        logger.info(f"Written: {file_size_mb:.2f} MB")
        
        # Print summary
        print_statistics(stats)
        
        logger.info(f"Laptop dataset saved to: {args.output.absolute()}")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Error during subset creation: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
