"""
PandasToArrowConverter: Transform 8 CSV files into unified Arrow dataset.

Implements the Aggregation → Registry → Explosion pipeline to handle
author deduplication and demographic merging via Full Outer Join.
"""

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import Dict, List, Optional, Set
from collections import defaultdict
import logging

from .schemas import SOBR_SCHEMA, validate_schema, get_categorical_columns

logger = logging.getLogger(__name__)


class PandasToArrowConverter:
    """
    Convert 8 demographic CSV files to a unified Arrow Table.
    
    Pipeline:
        1. Aggregation: Collapse each CSV by author_id, deduplicate posts
        2. Registry: Full Outer Join to create unified author profiles
        3. Explosion: Expand posts back to row-per-post format
        4. Serialization: Write Arrow Table with LZ4 compression
    """
    
    # Expected CSV filenames
    REQUIRED_FILES = [
        "birth_year.csv",
        "gender.csv",
        "nationality.csv",
        "political_leaning.csv",
        "extrovert_introvert.csv",
        "sensing_intuitive.csv",
        "feeling_thinking.csv",
        "judging_perceiving.csv",
    ]
    
    # CSV column name -> Schema column name mapping
    COLUMN_MAPPING = {
        "birth_year": "birth_year",  # No change
        "female": "female",  # gender.csv contains 'female' column
        "nationality": "nationality",  # No change
        "political_leaning": "political_leaning",  # No change
        "extrovert_introvert": "extrovert",  # Rename
        "sensing_intuitive": "sensing",  # Rename
        "feeling_thinking": "feeling",  # Rename
        "judging_perceiving": "judging",  # Rename
    }
    
    def __init__(self, raw_data_dir: Path):
        """
        Initialize converter with path to directory containing 8 CSVs.
        
        Args:
            raw_data_dir: Path to directory with demographic CSV files.
            
        Raises:
            FileNotFoundError: If any required CSV is missing.
        """
        self.raw_data_dir = Path(raw_data_dir)
        self._validate_input_files()
        
    def _validate_input_files(self) -> None:
        """Verify all required CSV files exist."""
        missing = []
        for filename in self.REQUIRED_FILES:
            if not (self.raw_data_dir / filename).exists():
                missing.append(filename)
        
        if missing:
            raise FileNotFoundError(
                f"Missing required CSV files: {', '.join(missing)}\n"
                f"Expected location: {self.raw_data_dir}"
            )
        
        logger.info(f"Validated presence of {len(self.REQUIRED_FILES)} CSV files")
    
    def convert(self, output_path: Path) -> Dict[str, any]:
        """
        Execute the full conversion pipeline.
        
        Args:
            output_path: Where to write the Arrow file (e.g., sobr.arrow).
            
        Returns:
            Dictionary with conversion statistics:
                - total_authors: Unique author count
                - total_posts: Total post count after deduplication
                - compression_ratio: Raw CSV size / Arrow size
                - null_percentages: % NULL for each demographic column
        """
        logger.info("=" * 80)
        logger.info("PHASE 1: AGGREGATION - Collapsing CSVs by author_id")
        logger.info("=" * 80)
        author_aggregates = self._aggregate_by_author()
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 2: REGISTRY - Creating unified author profiles")
        logger.info("=" * 80)
        author_registry = self._create_author_registry(author_aggregates)
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 3: EXPLOSION - Expanding posts to row-per-post")
        logger.info("=" * 80)
        exploded_df = self._explode_posts(author_registry)
        
        logger.info("\n" + "=" * 80)
        logger.info("PHASE 4: SERIALIZATION - Writing Arrow Table")
        logger.info("=" * 80)
        stats = self._serialize_to_arrow(exploded_df, output_path)
        
        return stats
    
    def _aggregate_by_author(self) -> Dict[str, pd.DataFrame]:
        """
        Phase 1: Read each CSV and collapse by author_id.
        
        Returns:
            Dict mapping filename -> aggregated DataFrame with columns:
                - author_id
                - <demographic_label> (resolved via majority vote)
                - posts (Set of unique post texts)
        """
        aggregates = {}
        
        for filename in self.REQUIRED_FILES:
            csv_path = self.raw_data_dir / filename
            logger.info(f"  Processing {filename}...")
            
            # Read CSV
            df = pd.read_csv(csv_path)
            
            # Ensure required columns exist (CSVs have misspelled 'auhtor_ID')
            if 'auhtor_ID' not in df.columns or 'post' not in df.columns:
                raise ValueError(f"{filename} missing 'auhtor_ID' or 'post' column")
            
            # Rename misspelled column to correct name
            df = df.rename(columns={'auhtor_ID': 'author_id'})
            
            # Extract demographic column (the one that's not author_id/post)
            demographic_col = [c for c in df.columns if c not in ['author_id', 'post']][0]
            
            # Rename demographic column to match schema
            # Find the target name from COLUMN_MAPPING
            schema_col_name = demographic_col  # Default to original
            for csv_col, schema_col in self.COLUMN_MAPPING.items():
                if csv_col == demographic_col:
                    schema_col_name = schema_col
                    break
            
            if schema_col_name != demographic_col:
                df = df.rename(columns={demographic_col: schema_col_name})
                demographic_col = schema_col_name
            
            # Group by author and aggregate
            grouped = df.groupby('author_id').agg({
                demographic_col: lambda x: self._resolve_conflict(x),
                'post': lambda x: set(x)  # Deduplicate posts
            }).reset_index()
            
            aggregates[filename] = grouped
            logger.info(f"    -> {len(grouped)} unique authors, "
                       f"{sum(len(posts) for posts in grouped['post'])} unique posts")
        
        return aggregates
    
    def _resolve_conflict(self, values: pd.Series) -> any:
        """
        Resolve conflicting demographic labels for same author.
        
        Strategy: First non-null value (conflict is rare in practice).
        """
        non_null = values.dropna()
        if len(non_null) == 0:
            return None
        return non_null.iloc[0]
    
    def _create_author_registry(self, aggregates: Dict[str, pd.DataFrame]) -> pd.DataFrame:
        """
        Phase 2: Full Outer Join across all 8 aggregated DataFrames.
        
        Returns:
            DataFrame with one row per unique author, containing:
                - author_id
                - All demographic columns (nullable)
                - posts (Set of unique posts across all CSVs)
        """
        # Start with first DataFrame
        first_file = self.REQUIRED_FILES[0]
        registry = aggregates[first_file].copy()
        
        # Sequentially merge remaining DataFrames
        for filename in self.REQUIRED_FILES[1:]:
            df = aggregates[filename]
            
            # Get demographic column name
            demo_col = [c for c in df.columns if c not in ['author_id', 'post']][0]
            
            # Full outer join on author_id
            registry = registry.merge(
                df,
                on='author_id',
                how='outer',
                suffixes=('', f'_{demo_col}')
            )
            
            # Merge post sets
            if 'post_' + demo_col in registry.columns:
                # Combine post sets from both sides
                registry['post'] = registry.apply(
                    lambda row: (row['post'] if isinstance(row['post'], set) else set()) | 
                               (row[f'post_{demo_col}'] if isinstance(row[f'post_{demo_col}'], set) else set()),
                    axis=1
                )
                registry = registry.drop(columns=[f'post_{demo_col}'])
        
        logger.info(f"  Registry created: {len(registry)} unique authors")
        
        # Compute NULL statistics
        for col in registry.columns:
            if col not in ['author_id', 'post']:
                null_pct = registry[col].isna().sum() / len(registry) * 100
                logger.info(f"    {col}: {null_pct:.1f}% NULL")
        
        return registry
    
    def _explode_posts(self, registry: pd.DataFrame) -> pd.DataFrame:
        """
        Phase 3: Transform from row-per-author to row-per-post.
        
        Expands the 'posts' Set column, replicating demographic labels
        for each post associated with that author.
        """
        rows = []
        post_id_counter = 0
        
        for _, author_row in registry.iterrows():
            author_id = author_row['author_id']
            posts = author_row['post']
            
            # Extract demographic columns (everything except author_id and post)
            demographics = {
                col: author_row[col] 
                for col in author_row.index 
                if col not in ['author_id', 'post']
            }
            
            # Create one row per post
            for post_text in posts:
                rows.append({
                    'post_id': f"post_{post_id_counter:08d}",
                    'author_id': author_id,
                    'post': post_text,
                    'post_masked': "",  # Populated in Phase A
                    'text_length': len(post_text),
                    'split': None,  # Assigned during dataset splitting
                    **demographics
                })
                post_id_counter += 1
        
        df = pd.DataFrame(rows)
        logger.info(f"  Exploded to {len(df)} total posts")
        
        return df
    
    def _serialize_to_arrow(self, df: pd.DataFrame, output_path: Path) -> Dict[str, any]:
        """
        Phase 4: Convert DataFrame to Arrow Table and serialize.
        
        Applies dictionary encoding and LZ4 compression.
        """
        # Apply categorical dtypes for dictionary encoding
        categorical_cols = get_categorical_columns()
        for col in categorical_cols:
            if col in df.columns:
                df[col] = df[col].astype('category')
        
        # Convert to Arrow Table
        table = pa.Table.from_pandas(df, schema=SOBR_SCHEMA, preserve_index=False)
        
        # Validate schema
        validate_schema(table, SOBR_SCHEMA)
        logger.info("Schema validation passed")
        
        # Compute statistics
        arrow_size_bytes = table.nbytes
        
        # Write with LZ4 compression
        output_path = Path(output_path)
        output_path.parent.mkdir(parents=True, exist_ok=True)
        
        feather.write_feather(
            table,
            output_path,
            compression='lz4'
        )
        
        file_size_mb = output_path.stat().st_size / (1024 ** 2)
        logger.info(f"  Written to {output_path}")
        logger.info(f"  File size: {file_size_mb:.2f} MB")
        
        # Compute null percentages
        null_stats = {}
        from .schemas import get_demographic_columns
        demographic_cols = get_demographic_columns()
        
        for col in table.column_names:
            if col in demographic_cols:
                null_count = pa.compute.sum(pa.compute.is_null(table[col])).as_py()
                null_pct = null_count / len(table) * 100
                null_stats[col] = null_pct
        
        return {
            'total_authors': len(df['author_id'].unique()),
            'total_posts': len(df),
            'file_size_mb': file_size_mb,
            'null_percentages': null_stats,
        }
