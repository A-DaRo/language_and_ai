#!/usr/bin/env python3
"""
Convert SOBR Pandas DataFrames to unified Arrow format.

This script orchestrates the one-time ingestion process:
    1. Validates presence of 8 demographic CSV files
    2. Triggers the Aggregation → Registry → Explosion pipeline
    3. Serializes the unified dataset to Arrow/Feather format
    4. Reports compression statistics and data quality metrics
"""

import argparse
import logging
import sys
from pathlib import Path
from tqdm import tqdm

# Add src to path for imports
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

from neuro_stylometry.data_engine.converter import PandasToArrowConverter


def setup_logging(verbose: bool = False):
    """Configure logging with optional verbose output."""
    level = logging.DEBUG if verbose else logging.INFO
    
    logging.basicConfig(
        level=level,
        format='%(asctime)s - %(name)s - %(levelname)s - %(message)s',
        handlers=[
            logging.StreamHandler(sys.stdout),
            logging.FileHandler('conversion.log', mode='w')
        ]
    )


def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(
        description="Convert SOBR CSV files to unified Arrow format",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Example usage:
    # Basic conversion
    python scripts/convert_pandas_to_arrow.py \\
        --raw-data-dir datasets/ \\
        --output artifacts/data/sobr.arrow
    
    # With verbose logging
    python scripts/convert_pandas_to_arrow.py \\
        --raw-data-dir datasets/ \\
        --output artifacts/data/sobr.arrow \\
        --verbose
        """
    )
    
    parser.add_argument(
        '--raw-data-dir',
        type=Path,
        default=Path('datasets'),
        help='Directory containing the 8 demographic CSV files (default: datasets/)'
    )
    
    parser.add_argument(
        '--output',
        type=Path,
        default=Path('artifacts/data/sobr.arrow'),
        help='Output path for unified Arrow file (default: artifacts/data/sobr.arrow)'
    )
    
    parser.add_argument(
        '--verbose',
        action='store_true',
        help='Enable verbose logging (DEBUG level)'
    )
    
    return parser.parse_args()


def validate_input_directory(raw_data_dir: Path):
    """
    Perform early validation of input directory.
    
    Args:
        raw_data_dir: Path to directory with CSV files.
        
    Raises:
        SystemExit: If validation fails.
    """
    if not raw_data_dir.exists():
        logging.error(f"Input directory does not exist: {raw_data_dir}")
        sys.exit(1)
    
    if not raw_data_dir.is_dir():
        logging.error(f"Input path is not a directory: {raw_data_dir}")
        sys.exit(1)
    
    # Count CSV files
    csv_files = list(raw_data_dir.glob("*.csv"))
    logging.info(f"Found {len(csv_files)} CSV files in {raw_data_dir}")
    
    for csv_file in csv_files:
        logging.info(f"  - {csv_file.name}")


def print_statistics(stats: dict):
    """
    Pretty-print conversion statistics.
    
    Args:
        stats: Statistics dictionary from converter.
    """
    print("\n" + "=" * 80)
    print("CONVERSION SUMMARY")
    print("=" * 80)
    
    print(f"\n📊 Dataset Statistics:")
    print(f"  • Total Authors:  {stats['total_authors']:,}")
    print(f"  • Total Posts:    {stats['total_posts']:,}")
    print(f"  • Avg Posts/Author: {stats['total_posts'] / stats['total_authors']:.1f}")
    
    print(f"\n💾 File Size:")
    print(f"  • Arrow File:     {stats['file_size_mb']:.2f} MB")
    
    print(f"\n🔍 Data Quality (NULL percentages):")
    null_stats = stats['null_percentages']
    for col, pct in sorted(null_stats.items()):
        bar_length = int(pct / 2)  # Scale to 50 chars max
        bar = '█' * bar_length + '░' * (50 - bar_length)
        print(f"  {col:20s} │{bar}│ {pct:5.1f}%")
    
    print("\n" + "=" * 80)
    print("✓ Conversion completed successfully!")
    print("=" * 80 + "\n")


def main():
    """Main execution flow."""
    args = parse_args()
    
    # Setup logging
    setup_logging(verbose=args.verbose)
    logger = logging.getLogger(__name__)
    
    logger.info("=" * 80)
    logger.info("SOBR Dataset Conversion: Pandas → Arrow")
    logger.info("=" * 80)
    logger.info(f"Raw data directory: {args.raw_data_dir.absolute()}")
    logger.info(f"Output Arrow file:  {args.output.absolute()}")
    logger.info("")
    
    try:
        # Validate input
        validate_input_directory(args.raw_data_dir)
        
        # Initialize converter
        logger.info("Initializing converter...")
        converter = PandasToArrowConverter(raw_data_dir=args.raw_data_dir)
        
        # Execute conversion with progress tracking
        logger.info("\nStarting conversion pipeline...\n")
        
        with tqdm(total=4, desc="Overall Progress", unit="phase") as pbar:
            # Phase 1: Aggregation
            pbar.set_description("Phase 1: Aggregation")
            stats = converter.convert(output_path=args.output)
            pbar.update(4)  # All phases complete
        
        # Print results
        print_statistics(stats)
        
        # Success message
        logger.info(f"Output written to: {args.output.absolute()}")
        logger.info("Conversion log saved to: conversion.log")
        
    except FileNotFoundError as e:
        logger.error(f"File not found: {e}")
        sys.exit(1)
    except ValueError as e:
        logger.error(f"Validation error: {e}")
        sys.exit(1)
    except Exception as e:
        logger.exception(f"Unexpected error during conversion: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()
