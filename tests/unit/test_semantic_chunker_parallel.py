import multiprocessing as mp
import os
from pathlib import Path
from tempfile import TemporaryDirectory
from time import perf_counter

import pytest

from neuro_stylometry.pollution_guard.semantic_chunker import (
    BudgetConfig,
    ChunkInfo,
    SemanticChunker,
    project_entity_offsets,
)


def _build_local_fast_tokenizer(tmp_dir: Path):
    """Create a minimal fast tokenizer saved to a local directory.

    This avoids network downloads and satisfies SemanticChunker parallel worker init,
    which uses AutoTokenizer.from_pretrained(..., local_files_only=True).
    """
    from tokenizers import Tokenizer
    from tokenizers.models import WordLevel
    from tokenizers.pre_tokenizers import Whitespace
    from transformers import PreTrainedTokenizerFast

    vocab = {
        "[PAD]": 0,
        "[UNK]": 1,
        "hello": 2,
        "world": 3,
        "I": 4,
        "am": 5,
        "a": 6,
        "test": 7,
        "token": 8,
        "sentence": 9,
        "This": 10,
        "is": 11,
        "long": 12,
        "Very": 13,
        "repeat": 14,
        "repeat-repeat": 15,
        "underscored_token": 16,
    }

    tk = Tokenizer(WordLevel(vocab=vocab, unk_token="[UNK]"))
    tk.pre_tokenizer = Whitespace()

    hf_tok = PreTrainedTokenizerFast(
        tokenizer_object=tk,
        unk_token="[UNK]",
        pad_token="[PAD]",
    )

    hf_tok.save_pretrained(str(tmp_dir))

    from transformers import AutoTokenizer

    return AutoTokenizer.from_pretrained(str(tmp_dir), use_fast=True, local_files_only=True)


def _make_docs():
    docs = [
        "",  # empty
        "   ",  # whitespace-only
        "Hello world.",
        "This is a test sentence. This is another sentence!",
        "I am a test token sentence. Very long " + ("repeat " * 200) + ".",
        "Punctuation-heavy: hello, world! (test) #42.",
        "Hyphenated repeat-repeat and underscored_token should be split consistently.",
        "Runon " + ("word " * 800),  # pathological run-on sentence to force hard split
    ]
    labels_list = [
        ["NAME"],
        ["NAME"],
        ["NAME"],
        ["NAME", "DATE"],
        ["NAME", "DATE"],
        ["NAME"],
        ["NAME"],
        ["NAME", "DATE"],
    ]
    return docs, labels_list


@pytest.mark.unit
def test_words_splitter_parity_with_gliner_words_splitter_whitespace():
    from gliner.data_processing.tokenizer import WordsSplitter

    text = "Hyphenated repeat-repeat and underscored_token; plus punctuation!"
    ws = WordsSplitter(splitter_type="whitespace")

    with TemporaryDirectory() as td:
        tokenizer = _build_local_fast_tokenizer(Path(td))
        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=BudgetConfig(model_max_length=64),
            words_splitter=ws,
            words_splitter_type="whitespace",
        )

    assert chunker._split_words(text) == list(ws(text))


@pytest.mark.unit
def test_parallel_chunking_matches_sequential_exactly():
    from gliner.data_processing.tokenizer import WordsSplitter

    # Ensure a consistent start method for safety in tests.
    try:
        mp.set_start_method("spawn", force=False)
    except RuntimeError:
        pass

    docs, labels_list = _make_docs()

    with TemporaryDirectory() as td:
        tokenizer = _build_local_fast_tokenizer(Path(td))
        ws = WordsSplitter(splitter_type="whitespace")

        base_cfg = BudgetConfig(
            model_max_length=64,
            gliner_max_words=64,
            tokens_per_word_ratio=1.3,
            mode="accumulate",
        )

        seq_cfg = base_cfg
        seq_cfg.parallel_chunking_workers = 0
        seq_cfg.parallel_chunking_min_texts = 0

        par_cfg = BudgetConfig(**seq_cfg.__dict__)
        par_cfg.parallel_chunking_workers = 2
        par_cfg.parallel_chunking_min_texts = 1

        chunker_seq = SemanticChunker(
            tokenizer=tokenizer,
            config=seq_cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )
        chunker_par = SemanticChunker(
            tokenizer=tokenizer,
            config=par_cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )

        seq = chunker_seq.chunk_texts(docs, labels_list)
        par = chunker_par.chunk_texts(docs, labels_list)

    assert seq == par


@pytest.mark.unit
def test_chunk_offsets_are_valid_and_substrings_match():
    from gliner.data_processing.tokenizer import WordsSplitter

    docs, labels_list = _make_docs()

    with TemporaryDirectory() as td:
        tokenizer = _build_local_fast_tokenizer(Path(td))
        ws = WordsSplitter(splitter_type="whitespace")
        cfg = BudgetConfig(model_max_length=64, gliner_max_words=64, mode="accumulate")
        cfg.parallel_chunking_workers = 2
        cfg.parallel_chunking_min_texts = 1

        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )

        results = chunker.chunk_texts(docs, labels_list)

    for doc, chunks in zip(docs, results):
        for ch in chunks:
            assert 0 <= ch.char_start < ch.char_end <= len(doc)
            assert doc[ch.char_start : ch.char_end] == ch.text


@pytest.mark.unit
def test_project_entity_offsets_projects_into_document_space():
    doc = "Hello world."
    chunk = ChunkInfo(text="world", char_start=6, char_end=11)
    entity = {"start": 0, "end": 5, "label": "X", "score": 0.9}

    projected = project_entity_offsets(entity, chunk)
    assert projected["start"] == 6
    assert projected["end"] == 11
    assert doc[projected["start"] : projected["end"]] == "world"


@pytest.mark.unit
def test_parallel_chunking_repeatability_no_nondeterminism():
    from gliner.data_processing.tokenizer import WordsSplitter

    docs, labels_list = _make_docs()

    with TemporaryDirectory() as td:
        tokenizer = _build_local_fast_tokenizer(Path(td))
        ws = WordsSplitter(splitter_type="whitespace")
        cfg = BudgetConfig(model_max_length=64, gliner_max_words=64, mode="accumulate")
        cfg.parallel_chunking_workers = 2
        cfg.parallel_chunking_min_texts = 1

        chunker = SemanticChunker(
            tokenizer=tokenizer,
            config=cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )

        first = chunker.chunk_texts(docs, labels_list)
        for _ in range(5):
            cur = chunker.chunk_texts(docs, labels_list)
            assert cur == first


@pytest.mark.unit
def test_parallel_is_not_slower_on_reasonable_corpus_smoke():
    """Opt-in micro-benchmark for speedup.

    NOTE: On Windows (spawn start method), process startup overhead can dominate
    for small corpora, making parallel chunking slower even when correct.
    Run locally with RUN_BENCHMARKS=1 for a meaningful measurement.
    """
    from gliner.data_processing.tokenizer import WordsSplitter

    # Make the workload large enough to amortize process startup overhead.
    docs = ["This is a test sentence. " * 40 for _ in range(5000)]
    labels_list = [["NAME", "DATE"] for _ in docs]

    with TemporaryDirectory() as td:
        tokenizer = _build_local_fast_tokenizer(Path(td))
        ws = WordsSplitter(splitter_type="whitespace")

        seq_cfg = BudgetConfig(model_max_length=64, gliner_max_words=64, mode="accumulate")
        seq_cfg.parallel_chunking_workers = 0
        seq_cfg.parallel_chunking_min_texts = 0

        par_cfg = BudgetConfig(model_max_length=64, gliner_max_words=64, mode="accumulate")
        par_cfg.parallel_chunking_workers = 4
        par_cfg.parallel_chunking_min_texts = 1

        chunker_seq = SemanticChunker(
            tokenizer=tokenizer,
            config=seq_cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )
        chunker_par = SemanticChunker(
            tokenizer=tokenizer,
            config=par_cfg,
            words_splitter=ws,
            words_splitter_type="whitespace",
        )

        t0 = perf_counter()
        _ = chunker_seq.chunk_texts(docs, labels_list)
        t1 = perf_counter()
        _ = chunker_par.chunk_texts(docs, labels_list)
        t2 = perf_counter()

    seq_s = t1 - t0
    par_s = t2 - t1

    # Expect at least a modest win when the benchmark is enabled.
    assert par_s < seq_s
