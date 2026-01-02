#!/usr/bin/env python3
"""
Ingest SOBR assignment CSVs into a unified Arrow dataset with split metadata.
"""

from __future__ import annotations

import argparse
import csv
import json
import random
from pathlib import Path
from typing import Dict, Iterable, List, Tuple

import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
from tqdm import tqdm


CSV_SPECS = {
    "birth_year.csv": ("birth_year", "Int32"),
    "extrovert_introvert.csv": ("extrovert", "Int8"),
    "feeling_thinking.csv": ("feeling", "Int8"),
    "gender.csv": ("female", "Int8"),
    "judging_perceiving.csv": ("judging", "Int8"),
    "nationality.csv": ("nationality", "string"),
    "political_leaning.csv": ("political_leaning", "string"),
    "sensing_intuitive.csv": ("sensing", "Int8"),
}

CATEGORICAL_COLUMNS = {"author_id", "nationality", "political_leaning"}


def read_label_csv(path: Path, label_column: str, label_dtype: str) -> pd.DataFrame:
    df = pd.read_csv(
        path,
        encoding="utf-8",
        dtype={"auhtor_ID": "string", "post": "string"},
        quoting=csv.QUOTE_MINIMAL,
        keep_default_na=True,
    )
    df = df.rename(columns={"auhtor_ID": "author_id"})
    if label_column not in df.columns:
        raise ValueError(f"Missing label column {label_column} in {path}")
    df[label_column] = df[label_column].astype(label_dtype)
    return df[["author_id", "post", label_column]]


def merge_label_frames(frames: List[pd.DataFrame]) -> pd.DataFrame:
    merged = frames[0]
    for frame in frames[1:]:
        merged = merged.merge(frame, on=["author_id", "post"], how="outer")
    return merged


def dedupe_conflicts(
    df: pd.DataFrame, label_columns: Iterable[str]
) -> Tuple[pd.DataFrame, Dict[str, int]]:
    conflict_counts: Dict[str, int] = {col: 0 for col in label_columns}

    def collapse(group: pd.DataFrame) -> pd.Series:
        base = {
            "author_id": group["author_id"].iloc[0],
            "post": group["post"].iloc[0],
        }
        for col in label_columns:
            values = group[col].dropna().unique()
            if len(values) > 1:
                conflict_counts[col] += 1
                base[col] = values[0]
            elif len(values) == 1:
                base[col] = values[0]
            else:
                base[col] = pd.NA
        return pd.Series(base)

    deduped = df.groupby(["author_id", "post"], dropna=False).apply(collapse)
    deduped = deduped.reset_index(drop=True)
    return deduped, conflict_counts


def author_splits(
    author_ids: List[str],
    train_ratio: float,
    val_ratio: float,
    seed: int,
) -> Dict[str, List[str]]:
    rng = random.Random(seed)
    shuffled = author_ids[:]
    rng.shuffle(shuffled)
    total = len(shuffled)
    train_end = int(total * train_ratio)
    val_end = train_end + int(total * val_ratio)
    return {
        "train": shuffled[:train_end],
        "val": shuffled[train_end:val_end],
        "test": shuffled[val_end:],
    }


def write_arrow(df: pd.DataFrame, output_path: Path) -> None:
    table = pa.Table.from_pandas(df, preserve_index=False)
    for col in CATEGORICAL_COLUMNS:
        if col in table.column_names:
            table = table.set_column(
                table.schema.get_field_index(col),
                col,
                table[col].dictionary_encode(),
            )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(
        table,
        output_path,
        compression="lz4",
        version=2,
    )


def write_report(
    output_dir: Path,
    df: pd.DataFrame,
    conflict_counts: Dict[str, int],
    splits: Dict[str, List[str]],
    seed: int,
) -> None:
    report_path = output_dir / "ingestion_report.md"
    with report_path.open("w", encoding="utf-8") as f:
        f.write("# Ingestion Report\n\n")
        f.write("| Metric | Value |\n| :--- | :--- |\n")
        f.write(f"| Rows | {len(df)} |\n")
        f.write(f"| Columns | {len(df.columns)} |\n")
        f.write(f"| Unique Authors | {df['author_id'].nunique()} |\n")
        f.write(f"| Seed | {seed} |\n")
        f.write("\n## Conflict Counts\n\n")
        f.write("| Column | Conflicts |\n| :--- | ---: |\n")
        for col, count in conflict_counts.items():
            f.write(f"| {col} | {count} |\n")
        f.write("\n## Split Sizes (Authors)\n\n")
        f.write("| Split | Authors |\n| :--- | ---: |\n")
        for split, authors in splits.items():
            f.write(f"| {split} | {len(authors)} |\n")


def main() -> None:
    parser = argparse.ArgumentParser(description="SOBR ingestion pipeline")
    parser.add_argument(
        "--input-dir",
        type=Path,
        default=Path("assignment_data"),
        help="Directory containing SOBR CSV files",
    )
    parser.add_argument(
        "--output-dir",
        type=Path,
        default=Path("artifacts/data"),
        help="Directory to write Arrow and metadata",
    )
    parser.add_argument("--seed", type=int, default=42, help="Split seed")
    parser.add_argument("--train-ratio", type=float, default=0.8)
    parser.add_argument("--val-ratio", type=float, default=0.1)
    args = parser.parse_args()

    frames = []
    for filename, (label_col, label_dtype) in tqdm(
        CSV_SPECS.items(),
        desc="Loading CSV files",
        unit="file",
    ):
        path = args.input_dir / filename
        if not path.exists():
            raise FileNotFoundError(f"Missing input file: {path}")
        frames.append(read_label_csv(path, label_col, label_dtype))

    with tqdm(total=5, desc="Pipeline", unit="step") as pipeline_bar:
        merged = merge_label_frames(frames)
        pipeline_bar.update(1)

        label_cols = [spec[0] for spec in CSV_SPECS.values()]
        deduped, conflicts = dedupe_conflicts(merged, label_cols)
        pipeline_bar.update(1)

        arrow_path = args.output_dir / "sobr_unified.arrow"
        write_arrow(deduped, arrow_path)
        pipeline_bar.update(1)

        authors = deduped["author_id"].dropna().unique().tolist()
        splits = author_splits(authors, args.train_ratio, args.val_ratio, args.seed)

        splits_path = args.output_dir / "splits.json"
        splits_path.parent.mkdir(parents=True, exist_ok=True)
        with splits_path.open("w", encoding="utf-8") as f:
            json.dump(
                {
                    "seed": args.seed,
                    "train_ratio": args.train_ratio,
                    "val_ratio": args.val_ratio,
                    "splits": splits,
                },
                f,
                ensure_ascii=True,
            )
        pipeline_bar.update(1)

        write_report(args.output_dir, deduped, conflicts, splits, args.seed)
        pipeline_bar.update(1)


if __name__ == "__main__":
    main()
