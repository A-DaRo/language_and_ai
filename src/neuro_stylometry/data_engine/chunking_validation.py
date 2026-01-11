"""Validation utilities for staged execution chunking artifacts.

These helpers are used for safe resume logic:
- Only skip Stage 1 when `post_chunked` is present, fully populated, and
  consistent with the raw `post` text.

Important nuance:
The semantic chunker strips sentence whitespace and (in accumulate mode)
re-joins sentences with single spaces. Hard-split chunks may also reflow
whitespace and include overlapping spans. Therefore, validation uses a
whitespace-normalized comparison.
"""

from __future__ import annotations

import re
from typing import Iterable, List, Optional

import pyarrow as pa


_WS_RE = re.compile(r"\s+")


def _norm_ws(text: str) -> str:
    return _WS_RE.sub(" ", text).strip()


def find_invalid_post_chunked_indices(
    *,
    posts: List[str],
    post_chunked_column: pa.Array,
    sample_indices: Optional[Iterable[int]] = None,
    max_invalid: Optional[int] = None,
) -> List[int]:
    """Return row indices where post_chunked does not match the corresponding post.

    Rules (tolerant to semantic_chunker whitespace behavior):
    - For each chunk, `normalize_ws(chunk.text)` must equal
      `normalize_ws(post[start:end])`.
    - Chunk spans must be within bounds and ordered (non-decreasing start).
    - Any *gaps* between successive chunks may only contain whitespace.
      Overlaps are allowed (hard-split overlap).
    - Coverage: chunks must cover all non-whitespace characters of the post.

    Empty posts (whitespace-only) are valid with an empty chunk list.

    Args:
        posts: List of raw post strings.
        post_chunked_column: Arrow list<struct> column (Array or ChunkedArray).
        sample_indices: Optional iterable of row indices to validate.
            If provided, only these rows are checked.
        max_invalid: Optional early-exit cap for number of invalid indices collected.

    Returns:
        List of invalid row indices.
    """
    if len(posts) != len(post_chunked_column):
        raise ValueError(
            f"Length mismatch: posts={len(posts)} vs post_chunked={len(post_chunked_column)}"
        )

    if sample_indices is None:
        indices = range(len(posts))
    else:
        indices = sample_indices

    invalid: List[int] = []

    for i in indices:
        post = posts[i] or ""
        post_norm = _norm_ws(post)

        try:
            chunk_list = post_chunked_column[i].as_py()
        except Exception:
            chunk_list = None

        if chunk_list is None:
            invalid.append(i)
        else:
            # post_chunked is list[dict] (or empty list)
            if not chunk_list:
                if post_norm != "":
                    invalid.append(i)
            else:
                # Sort by declared start offset (should already be ordered)
                try:
                    chunk_list_sorted = sorted(chunk_list, key=lambda c: int(c.get("start", 0)))
                except Exception:
                    chunk_list_sorted = chunk_list

                # Determine non-whitespace bounds
                # Find first/last non-ws char in original post
                first_non_ws = None
                last_non_ws_end = None
                for idx, ch in enumerate(post):
                    if not ch.isspace():
                        first_non_ws = idx
                        break
                for idx in range(len(post) - 1, -1, -1):
                    if not post[idx].isspace():
                        last_non_ws_end = idx + 1
                        break

                if first_non_ws is None or last_non_ws_end is None:
                    # whitespace-only post; chunks must be empty
                    invalid.append(i)
                else:
                    prev_end = None
                    min_start = None
                    max_end = None
                    ok = True

                    for chunk in chunk_list_sorted:
                        try:
                            start = int(chunk.get("start", 0))
                            end = int(chunk.get("end", 0))
                            chunk_text = str(chunk.get("text", ""))
                        except Exception:
                            ok = False
                            break

                        if start < 0 or end < 0 or start > end or end > len(post):
                            ok = False
                            break

                        if min_start is None or start < min_start:
                            min_start = start
                        if max_end is None or end > max_end:
                            max_end = end

                        # Gaps are allowed only if they contain whitespace
                        if prev_end is not None and start > prev_end:
                            gap = post[prev_end:start]
                            if _norm_ws(gap) != "":
                                ok = False
                                break

                        # Per-chunk content check (whitespace-normalized)
                        raw_span = post[start:end]
                        if _norm_ws(raw_span) != _norm_ws(chunk_text):
                            ok = False
                            break

                        # Ordering (allow overlap)
                        prev_end = max(prev_end, end) if prev_end is not None else end

                    if ok:
                        if min_start is None or max_end is None:
                            ok = False
                        else:
                            if min_start > first_non_ws:
                                ok = False
                            if max_end < last_non_ws_end:
                                ok = False

                    if not ok:
                        invalid.append(i)

        if max_invalid is not None and len(invalid) >= max_invalid:
            break

    return invalid
