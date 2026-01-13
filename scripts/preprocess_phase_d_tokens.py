#!/usr/bin/env python3
"""Pre-tokenize Phase D datasets into Arrow for offline bucketing."""

from __future__ import annotations

import argparse
import logging
from pathlib import Path
from typing import List

import pyarrow as pa
import pyarrow.feather as feather
from transformers import AutoTokenizer

logger = logging.getLogger(__name__)


def _parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="Pre-tokenize Phase D Arrow datasets.")
    parser.add_argument("--input", required=True, type=Path, help="Input Arrow file")
    parser.add_argument("--output", required=True, type=Path, help="Output Arrow file")
    parser.add_argument("--model", default="roberta-base", help="Tokenizer model name")
    parser.add_argument(
        "--text-field",
        default="post",
        help="Text column to tokenize (e.g., post or post_masked)",
    )
    parser.add_argument(
        "--column-prefix",
        default=None,
        help="Prefix for token columns (defaults to text field)",
    )
    parser.add_argument("--max-length", type=int, default=512, help="Max token length")
    parser.add_argument("--batch-size", type=int, default=1024, help="Tokenization batch size")
    parser.add_argument(
        "--overwrite-columns",
        action="store_true",
        help="Overwrite existing token columns in the output file",
    )
    return parser.parse_args()


def _select_token_type(vocab_size: int) -> pa.DataType:
    if vocab_size <= 65535:
        return pa.uint16()
    return pa.int32()


def _tokenize_column(
    table: pa.Table,
    *,
    tokenizer,
    text_field: str,
    max_length: int,
    batch_size: int,
    token_type: pa.DataType,
) -> tuple[pa.ChunkedArray, pa.ChunkedArray, pa.ChunkedArray]:
    input_id_chunks: List[pa.Array] = []
    mask_chunks: List[pa.Array] = []
    count_chunks: List[pa.Array] = []

    column = table[text_field]
    for batch in table.to_batches(max_chunksize=batch_size):
        texts = batch.column(batch.schema.get_field_index(text_field)).to_pylist()
        enc = tokenizer(
            texts,
            padding=False,
            truncation=True,
            max_length=max_length,
            return_attention_mask=False,
        )
        input_ids = enc["input_ids"]
        input_id_chunks.append(pa.array(input_ids, type=pa.list_(token_type)))
        counts = [len(seq) for seq in input_ids]
        count_chunks.append(pa.array(counts, type=pa.int32()))
        mask_chunks.append(
            pa.array([[1] * count for count in counts], type=pa.list_(pa.int8()))
        )

    return (
        pa.chunked_array(input_id_chunks),
        pa.chunked_array(mask_chunks),
        pa.chunked_array(count_chunks),
    )


def main() -> None:
    args = _parse_args()
    logging.basicConfig(level=logging.INFO, format="%(levelname)s - %(message)s")

    output_path = Path(args.output)
    base_path = output_path if output_path.exists() else Path(args.input)
    table = feather.read_table(base_path, memory_map=True)

    if args.text_field not in table.column_names:
        raise ValueError(f"Missing text field '{args.text_field}' in {base_path}")

    prefix = args.column_prefix or args.text_field
    input_name = f"{prefix}_input_ids"
    mask_name = f"{prefix}_attention_mask"
    count_name = f"{prefix}_token_count"

    for name in (input_name, mask_name, count_name):
        if name in table.column_names and not args.overwrite_columns:
            raise ValueError(
                f"Column '{name}' already exists in {base_path}. "
                "Use --overwrite-columns to replace it."
            )

    tokenizer = AutoTokenizer.from_pretrained(args.model)
    token_type = _select_token_type(int(tokenizer.vocab_size))

    logger.info("Tokenizing %s with %s", args.text_field, args.model)
    input_ids, attention_mask, token_count = _tokenize_column(
        table,
        tokenizer=tokenizer,
        text_field=args.text_field,
        max_length=args.max_length,
        batch_size=args.batch_size,
        token_type=token_type,
    )

    new_table = table
    if input_name in new_table.column_names:
        new_table = new_table.drop([input_name])
    if mask_name in new_table.column_names:
        new_table = new_table.drop([mask_name])
    if count_name in new_table.column_names:
        new_table = new_table.drop([count_name])

    new_table = new_table.append_column(input_name, input_ids)
    new_table = new_table.append_column(mask_name, attention_mask)
    new_table = new_table.append_column(count_name, token_count)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    feather.write_feather(new_table, output_path)
    logger.info("Wrote tokenized Arrow file to %s", output_path)


if __name__ == "__main__":
    main()
