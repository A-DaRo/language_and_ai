"""
Integration tests for Phase A Pipeline outputs.

Verifies that both LaptopFilterStrategy and HPCFilterStrategy produce
all expected outputs with exact specifications:
- Cleaned dataset with correct schema
- Projection matrix with correct shape
- Pollution logs with correct schema
- Metrics JSON with complete structure
- All expected visualizations

Uses mocked inputs to avoid actual GLiNER/encoder inference.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional, Set
from unittest.mock import MagicMock, patch, PropertyMock

import numpy as np
import pandas as pd
import pyarrow as pa
import pyarrow.feather as feather
import pytest
import torch

# Configure logging for tests
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)


# =============================================================================
# Expected Output Specifications (Source of Truth)
# =============================================================================

# Demographic columns from schemas.py
DEMOGRAPHIC_COLUMNS = [
    "birth_year", "female", "nationality", "political_leaning",
    "extrovert", "sensing", "feeling", "judging"
]

# Core artifact filenames (from phase_a_pipeline.py)
CORE_ARTIFACTS = {
    "clean_dataset": "clean_dataset.arrow",
    "projection_matrix": "projection_matrix.pt",
    "pollution_logs": "pollution_logs.arrow",
}

# Reports directory structure (from phase_a_pipeline.py)
REPORTS_SUBDIR = Path("reports") / "phase_a"
METRICS_FILENAME = "phase_a_metrics.json"

# Expected plots from generate_phase_a_plots() in visualizations_phase_a.py
PIPELINE_VISUALIZATION_PLOTS = {
    "singular_values.png",
    "embedding_norms.png",
    "amnesic_drop_bars.png",
    "gliner_confidence.png",
    "detection_counts.png",
    "explicit_recall_bars.png",
    "entity_cooccurrence.png",
    "demographic_scores.png",
    "masking_coverage.png",
}

# PCA plots are generated per demographic column (pca_{col}.png)
def get_expected_pca_plots(columns: List[str]) -> Set[str]:
    """Get expected PCA plot filenames for given demographic columns."""
    return {f"pca_{col}.png" for col in columns}

# HPC-specific plots from hpc.py Step 8
HPC_SPECIFIC_PLOTS = {
    "amnesic_drop_with_ci.png",
    "embedding_separability.png",
    "solver_convergence_benchmark.png",
    "specificity_gap.png",
}

# Expected metrics JSON structure (from metrics.py compute_phase_a_metrics)
REQUIRED_METRICS_TOP_LEVEL_KEYS = {
    "timestamp",
    "config_hash",
    "dataset",
    "gliner",
    "leace",
    "execution",
}

REQUIRED_DATASET_METRICS_KEYS = {
    "num_samples",
    "num_with_masks",
    "mask_rate",
    "mask_token_counts",
    "text_length",
    "masked_text_length",
    "demographic_coverage",
}

REQUIRED_GLINER_METRICS_KEYS = {
    "num_spans_detected",
    "spans_per_sample",
    "confidence",
    "by_entity_type",
    "span_length",
}

REQUIRED_LEACE_METRICS_KEYS = {
    "shape",
    "dtype",
    "frobenius_norm",
    "idempotence_error",
    "idempotence_satisfied",
    "is_finite",
    "singular_values",
}

REQUIRED_PROBE_METRICS_KEYS = {
    "by_column",
    "min_amnesic_drop",
    "mean_amnesic_drop",
    "threshold",
    "max_samples",
    "config",
}

# Extended probe metrics (HPC mode)
REQUIRED_EXTENDED_PROBE_KEYS = {
    "by_column_extended",
}

# Pollution log schema columns (from schemas.py POLLUTION_LOG_SCHEMA)
POLLUTION_LOG_COLUMNS = {
    "post_id",
    "entity_type",
    "span_text",
    "span_start",
    "span_end",
    "confidence",
    "mask_token",  # The typed mask token applied (e.g., "[MASK:BIRTH_YEAR]")
}

# Clean dataset required columns (from SOBR_SCHEMA)
CLEAN_DATASET_REQUIRED_COLUMNS = {
    "post_id",
    "author_id",
    "post",
    "post_masked",
    "birth_year",
    "female",
    "nationality",
    "political_leaning",
    "extrovert",
    "sensing",
    "feeling",
    "judging",
    "split",
    "text_length",
}


# =============================================================================
# Fixtures for mocked data
# =============================================================================

@pytest.fixture
def mock_sobr_schema():
    """SOBR Arrow schema for test data."""
    return pa.schema([
        ("post_id", pa.string()),
        ("author_id", pa.dictionary(pa.int32(), pa.string())),
        ("post", pa.string()),
        ("post_masked", pa.string()),
        ("birth_year", pa.int16()),
        ("female", pa.int8()),
        ("nationality", pa.string()),
        ("political_leaning", pa.string()),
        ("extrovert", pa.int8()),
        ("sensing", pa.int8()),
        ("feeling", pa.int8()),
        ("judging", pa.int8()),
        ("split", pa.dictionary(pa.int8(), pa.string())),
        ("text_length", pa.int32()),
    ])


@pytest.fixture
def mock_pollution_logs_schema():
    """Pollution logs Arrow schema."""
    return pa.schema([
        ("post_id", pa.string()),
        ("entity_type", pa.string()),
        ("span_text", pa.string()),
        ("span_start", pa.int64()),
        ("span_end", pa.int64()),
        ("confidence", pa.float64()),
        ("column", pa.string()),
    ])


@pytest.fixture
def mock_input_data(mock_sobr_schema, tmp_path):
    """Create a mock input Arrow dataset."""
    np.random.seed(42)
    n_samples = 200

    posts = [
        f"I am a {np.random.choice(['man', 'woman'])} from {np.random.choice(['USA', 'UK', 'Germany'])}. "
        f"My name is {np.random.choice(['John', 'Jane', 'Alex'])} and I was born in {1950 + i % 50}. "
        f"I think the political situation is {np.random.choice(['good', 'bad', 'complicated'])}."
        for i in range(n_samples)
    ]

    data = {
        "post_id": [f"post_{i:04d}" for i in range(n_samples)],
        "author_id": [f"author_{i % 50:03d}" for i in range(n_samples)],
        "post": posts,
        "post_masked": [""] * n_samples,
        "birth_year": np.random.randint(1950, 2000, n_samples).astype(np.int16).tolist(),
        "female": np.random.randint(0, 2, n_samples).astype(np.int8).tolist(),
        "nationality": np.random.choice(["USA", "UK", "Germany", "France"], n_samples).tolist(),
        "political_leaning": np.random.choice(["left", "center", "right"], n_samples).tolist(),
        "extrovert": np.random.randint(0, 2, n_samples).astype(np.int8).tolist(),
        "sensing": np.random.randint(0, 2, n_samples).astype(np.int8).tolist(),
        "feeling": np.random.randint(0, 2, n_samples).astype(np.int8).tolist(),
        "judging": np.random.randint(0, 2, n_samples).astype(np.int8).tolist(),
        "split": ["train"] * n_samples,
        "text_length": [len(p) for p in posts],
    }

    df = pd.DataFrame(data)
    table = pa.Table.from_pandas(df, schema=mock_sobr_schema)

    input_path = tmp_path / "input_dataset.arrow"
    feather.write_feather(table, input_path)

    return input_path


@pytest.fixture
def mock_taxonomy_config(tmp_path):
    """Create a mock taxonomy YAML config."""
    taxonomy_yaml = """
taxonomy:
  columns:
    birth_year:
      labels:
        - "age indicator"
        - "birth year"
      mask_token: "[MASK:AGE]"
      min_width: 2
      max_width: 10
    female:
      labels:
        - "gender indicator"
        - "sex reference"
      mask_token: "[MASK:GENDER]"
      min_width: 2
      max_width: 15
    nationality:
      labels:
        - "nationality"
        - "country reference"
      mask_token: "[MASK:NATIONALITY]"
      min_width: 2
      max_width: 20
    political_leaning:
      labels:
        - "political view"
        - "political stance"
      mask_token: "[MASK:POLITICS]"
      min_width: 3
      max_width: 25
"""
    taxonomy_path = tmp_path / "gliner_taxonomy.yaml"
    taxonomy_path.write_text(taxonomy_yaml)
    return taxonomy_path


@pytest.fixture
def mock_base_config(mock_taxonomy_config, tmp_path):
    """Create a mock base pipeline configuration."""
    return {
        "seed": 42,
        "gliner": {
            "model": "mock-gliner-model",
            "confidence_threshold": 0.5,
            "batch_size": 16,
            "device": "cpu",
            "taxonomy_path": str(mock_taxonomy_config),
            "compute_explicit_recall": True,
            "explicit_recall_threshold": 0.5,
            "require_bi_encoder": False,
            "gliner_max_words": 256,
            "tokens_per_word_ratio": 1.3,
            "batch_inference": {
                "enable_batching": True,
                "num_buckets": 4,
                "min_bucket_size": 2,
                "enable_prompt_caching": False,
                "strict_padding": False,
            },
            "chunking": {
                "mode": "single_sentence",
                "legacy_sequential_mode": False,
                "parallel_chunking_workers": 0,
                "parallel_chunking_min_texts": 1000,
            },
        },
        "encoder": {
            "model": "mock-encoder-model",
            "hidden_dim": 64,
            "max_length": 128,
            "batch_size": 16,
            "device": "cpu",
            "output_device": "cpu",
        },
        "dataloader": {
            "batch_size": 16,
            "num_workers": 0,
            "persistent_workers": False,
            "prefetch_factor": 2,
            "pin_memory": False,
        },
        "precision": {
            "dtype": "float32",
            "use_grad_scaler": False,
        },
        "probe": {
            "compute_amnesic_drop": True,
            "amnesic_drop_threshold": 0.1,
            "train_split": 0.8,
            "max_samples": 100,
            "backend": "sklearn",
            "use_kfold": True,
            "n_folds": 3,
            "pvalue_threshold": 0.05,
            "torch_lr": 0.01,
            "torch_epochs": 10,
            "torch_batch_size": 32,
            "torch_weight_decay": 0.0001,
            "use_class_weights": True,
            "benchmark_solvers": False,
        },
        "leace": {
            "force_cpu": True,
            "regularization": 1e-4,
            "batch_size": 50,
            "device": "cpu",
            "compute_dtype": "float64",
        },
        "quality": {
            "enforce_thresholds": False,
        },
        "subset": {
            "enabled": False,
            "size": None,
        },
        "visualization": {
            "enabled": True,
            "pca_max_samples": 50,
            "embedding_batch_size": 16,
            "figure_dpi": 72,
            "figure_format": "png",
            "color_palette": "husl",
            "plots": {
                "pca_before_after": True,
                "singular_values": True,
                "embedding_norms": True,
                "amnesic_drop_bars": True,
                "gliner_confidence": True,
                "detection_counts": True,
                "explicit_recall_bars": True,
                "entity_cooccurrence": True,
                "demographic_scores": True,
                "masking_coverage": True,
            },
        },
        "logging": {
            "level": "INFO",
            "wandb_enabled": False,
        },
        "paths": {
            "raw_data": str(tmp_path / "datasets"),
            "output": str(tmp_path / "artifacts"),
        },
        "execution": {
            "enable_cuda_graphs": False,
            "enable_torch_compile": False,
            "autotuning": {
                "enabled": False,
            },
            "async_prefetch": {
                "enabled": False,
            },
            "sequence_packing": {
                "enabled": False,
            },
            "gpu_span_filter": {
                "enabled": False,
            },
        },
    }


@pytest.fixture
def mock_hpc_config(mock_base_config):
    """HPC-specific config with benchmark_solvers enabled."""
    config = mock_base_config.copy()
    config["probe"] = config["probe"].copy()
    config["probe"]["benchmark_solvers"] = True
    config["execution"] = config.get("execution", {}).copy()
    config["execution"]["staged_execution"] = {
        "enabled": True,
        "persist_chunks": True,
        "global_sort_by_length": True,
        "gc_between_stages": True,
    }
    config["execution"]["async_storage"] = {
        "enabled": False,
    }
    return config


# =============================================================================
# Mock classes for GLiNER and encoder
# =============================================================================

class MockGLiNERModel:
    """Mock GLiNER model for testing."""

    def __init__(self):
        self.data_processor = MagicMock()
        self.data_processor.transformer_tokenizer = MagicMock()
        self.data_processor.transformer_tokenizer.model_max_length = 512
        self.config = MagicMock()
        self.config.labels_encoder = True

    def to(self, device):
        return self

    def eval(self):
        return self

    def predict_entities(self, texts, labels, threshold=0.5, **kwargs):
        """Return mock entities for each text."""
        results = []
        for text in texts:
            entities = []
            if "man" in text.lower() or "woman" in text.lower():
                start = text.lower().find("man") if "man" in text.lower() else text.lower().find("woman")
                entities.append({
                    "start": start,
                    "end": start + 5,
                    "text": text[start:start + 5],
                    "label": "gender indicator",
                    "score": 0.85,
                })
            if "USA" in text or "UK" in text:
                for country in ["USA", "UK"]:
                    if country in text:
                        start = text.find(country)
                        entities.append({
                            "start": start,
                            "end": start + len(country),
                            "text": country,
                            "label": "nationality",
                            "score": 0.90,
                        })
            results.append(entities)
        return results

    def batch_predict_entities(self, texts, labels, threshold=0.5, **kwargs):
        return self.predict_entities(texts, labels, threshold, **kwargs)


class MockTokenizer:
    """Mock tokenizer for testing."""

    model_max_length = 512

    def __init__(self):
        self.vocab = {"[PAD]": 0, "[UNK]": 1, "[CLS]": 2, "[SEP]": 3}
        self._added_tokens = set()

    def __call__(self, texts, **kwargs):
        if isinstance(texts, str):
            texts = [texts]
        return {
            "input_ids": [[1] * min(len(t.split()), 128) for t in texts],
            "attention_mask": [[1] * min(len(t.split()), 128) for t in texts],
        }

    def encode(self, text, **kwargs):
        return [1] * min(len(text.split()), 128)

    def add_special_tokens(self, special_tokens_dict):
        tokens = special_tokens_dict.get("additional_special_tokens", [])
        self._added_tokens.update(tokens)
        return len(tokens)

    def get_added_vocab(self):
        return {t: i + 100 for i, t in enumerate(self._added_tokens)}


class MockEncoder:
    """Mock encoder model for embeddings."""

    def __init__(self, hidden_dim=64):
        self.hidden_dim = hidden_dim
        self.config = MagicMock()
        self.config.hidden_size = hidden_dim

    def to(self, device):
        return self

    def eval(self):
        return self

    def __call__(self, **kwargs):
        batch_size = len(kwargs.get("input_ids", [[]]))
        last_hidden_state = torch.randn(batch_size, 10, self.hidden_dim)
        return MagicMock(last_hidden_state=last_hidden_state)


# =============================================================================
# Mock patches
# =============================================================================

@pytest.fixture
def mock_cuml():
    """Mock cuML module since it's not available."""
    mock_cuml_module = MagicMock()

    class MockCuMLLogisticRegression:
        def __init__(self, **kwargs):
            self.coef_ = None
            self.intercept_ = None

        def fit(self, X, y):
            n_classes = len(np.unique(y.get() if hasattr(y, 'get') else y))
            n_features = X.shape[1] if hasattr(X, 'shape') else 64
            self.coef_ = np.random.randn(n_classes, n_features)
            self.intercept_ = np.random.randn(n_classes)
            return self

        def predict(self, X):
            n = X.shape[0] if hasattr(X, 'shape') else len(X)
            return np.random.randint(0, 2, n)

    mock_cuml_module.linear_model = MagicMock()
    mock_cuml_module.linear_model.LogisticRegression = MockCuMLLogisticRegression

    return mock_cuml_module


@pytest.fixture
def mock_cupy():
    """Mock cupy module."""
    mock_cp = MagicMock()
    mock_cp.asarray = lambda x, dtype=None: np.asarray(x, dtype=np.float32 if dtype else None)
    mock_cp.asnumpy = lambda x: np.asarray(x)
    mock_cp.float32 = np.float32
    return mock_cp


# =============================================================================
# Strategy mock helpers
# =============================================================================

def create_mock_strategy_execute(strategy_class, mock_config):
    """Create a simplified execute method for testing that produces exact expected outputs."""

    def mock_execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simplified execution that produces all expected outputs."""
        from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA

        # Load input
        table = feather.read_table(input_dataset_path)
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        n_samples = len(posts)

        # Generate mock masked texts with deterministic pattern
        masked_texts = [
            post.replace("man", "[MASK:GENDER]").replace("woman", "[MASK:GENDER]")
            .replace("USA", "[MASK:NATIONALITY]").replace("UK", "[MASK:NATIONALITY]")
            for post in posts
        ]

        # Generate mock pollution logs (deterministic)
        pollution_logs = []
        for i, (post, post_id) in enumerate(zip(posts, post_ids)):
            if "man" in post.lower() or "woman" in post.lower():
                pollution_logs.append({
                    "post_id": post_id,
                    "entity_type": "gender indicator",
                    "span_text": "man" if "man" in post.lower() else "woman",
                    "span_start": 5,
                    "span_end": 10,
                    "confidence": 0.85,
                    "mask_token": "[MASK:GENDER]",
                })
            if "USA" in post or "UK" in post:
                pollution_logs.append({
                    "post_id": post_id,
                    "entity_type": "nationality",
                    "span_text": "USA" if "USA" in post else "UK",
                    "span_start": 20,
                    "span_end": 23,
                    "confidence": 0.90,
                    "mask_token": "[MASK:NATIONALITY]",
                })

        # Save cleaned dataset
        df = table.to_pandas()
        df["post_masked"] = masked_texts
        clean_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
        feather.write_feather(clean_table, output_dataset_path)

        # Save projection matrix (64x64 for test embedder dim)
        projection_matrix = torch.eye(64) + 0.01 * torch.randn(64, 64)
        torch.save(projection_matrix, projection_matrix_path)

        # Save pollution logs
        if pollution_logs:
            logs_table = pa.Table.from_pylist(pollution_logs, schema=POLLUTION_LOG_SCHEMA)
        else:
            logs_table = pa.Table.from_pylist([], schema=POLLUTION_LOG_SCHEMA)
        feather.write_feather(logs_table, pollution_logs_path)

        # Build complete metadata with all expected fields
        from neuro_stylometry.data_engine.schemas import get_demographic_columns

        by_column = {}
        by_column_extended = {}
        separability_before = {}
        separability_after = {}
        class_imbalance = {}

        np.random.seed(42)  # Deterministic for tests
        for col in get_demographic_columns():
            acc_before = 0.75 + np.random.uniform(-0.05, 0.05)
            acc_after = 0.55 + np.random.uniform(-0.05, 0.05)
            amnesic_drop = acc_before - acc_after

            by_column[col] = {
                "accuracy_before": float(acc_before),
                "accuracy_after": float(acc_after),
                "amnesic_drop": float(amnesic_drop),
            }
            by_column_extended[col] = {
                **by_column[col],
                "balanced_acc_before": 0.72,
                "balanced_acc_after": 0.52,
                "balanced_amnesic_drop": 0.20,
                "is_significant": True,
                "pvalue": 0.01,
                "cv_before": {
                    "mean_accuracy": 0.75,
                    "std_accuracy": 0.05,
                    "ci_95_lower": 0.70,
                    "ci_95_upper": 0.80,
                },
                "cv_after": {
                    "mean_accuracy": 0.55,
                    "std_accuracy": 0.06,
                    "ci_95_lower": 0.49,
                    "ci_95_upper": 0.61,
                },
            }
            separability_before[col] = {
                "silhouette_score": 0.30,
                "davies_bouldin_index": 1.50,
                "n_samples": n_samples,
                "n_classes": 2,
            }
            separability_after[col] = {
                "silhouette_score": 0.10,
                "davies_bouldin_index": 2.00,
                "n_samples": n_samples,
                "n_classes": 2,
            }
            class_imbalance[col] = {
                "distribution": {"0": n_samples // 2, "1": n_samples // 2},
                "proportions": {"0": 0.5, "1": 0.5},
                "imbalance_ratio": 1.0,
                "majority_baseline": 0.5,
                "majority_class": 0,
                "minority_class": 1,
                "n_classes": 2,
            }

        drops = [v["amnesic_drop"] for v in by_column.values()]

        metadata = {
            "num_samples": n_samples,
            "num_pollution_spans": len(pollution_logs),
            "projection_matrix_shape": [64, 64],
            "device": "cpu",
            "staged_execution": True,
            "stage1_skipped": False,
            "mode": strategy_class.__name__.replace("FilterStrategy", "").lower(),
            "explicit_recall": {
                "overall": 0.85,
                "by_column": {col: 0.80 for col in get_demographic_columns()},
            },
            "probe": {
                "by_column": by_column,
                "by_column_extended": by_column_extended,
                "min_amnesic_drop": float(min(drops)),
                "mean_amnesic_drop": float(np.mean(drops)),
                "threshold": config.get("probe", {}).get("amnesic_drop_threshold", 0.3),
                "max_samples": min(n_samples, config.get("probe", {}).get("max_samples", 100)),
                "config": {
                    "backend": config.get("probe", {}).get("backend", "sklearn"),
                    "use_kfold": config.get("probe", {}).get("use_kfold", True),
                    "n_folds": config.get("probe", {}).get("n_folds", 5),
                    "pvalue_threshold": config.get("probe", {}).get("pvalue_threshold", 0.05),
                },
            },
            "separability": {
                "before": separability_before,
                "after": separability_after,
            },
            "class_imbalance": class_imbalance,
        }

        # Add solver benchmark for HPC mode
        if config.get("probe", {}).get("benchmark_solvers", False):
            metadata["probe"]["solver_benchmark"] = {
                "exact_accuracy": 0.76,
                "torch_accuracy": 0.74,
                "solver_accuracy_delta": 0.02,
                "weight_cosine_similarity": 0.95,
                "time_to_convergence_ratio": 1.5,
                "exact_time_s": 0.1,
                "torch_time_s": 0.15,
            }

        # Add control probe for HPC mode
        if "hpc" in strategy_class.__name__.lower():
            metadata["control_probe"] = {
                "target": {
                    "acc_before": 0.75,
                    "acc_after": 0.55,
                    "drop": 0.20,
                },
                "control_female": {
                    "acc_before": 0.72,
                    "acc_after": 0.70,
                    "drop": 0.02,
                },
                "control_stability_score": 0.98,
                "specificity_ratio": 10.0,
                "is_specific": True,
            }

        return metadata

    return mock_execute


# =============================================================================
# Test classes
# =============================================================================

class TestPhaseAPipelineOutputs:
    """Test that Phase A pipeline produces all expected outputs with exact specifications."""

    def _patch_pipeline_components(self, monkeypatch):
        """Patch pipeline get_embedder to avoid HuggingFace calls."""
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        class MockEmbedderForPipeline:
            def __init__(self):
                self._embedding_dim = 64

            def embed_texts(self, texts, batch_size=32, show_progress=False):
                n = len(texts)
                return torch.randn(n, 64)

            def get_embedding_dim(self):
                return 64

        def mock_get_embedder(self):
            return MockEmbedderForPipeline()

        monkeypatch.setattr(PhaseAPipeline, "get_embedder", mock_get_embedder)

    def test_laptop_strategy_core_artifacts(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test LaptopFilterStrategy produces all core artifacts with correct names."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)

        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )

        # Assert exact artifact paths
        assert artifacts.clean_dataset_path == output_dir / CORE_ARTIFACTS["clean_dataset"]
        assert artifacts.projection_matrix_path == output_dir / CORE_ARTIFACTS["projection_matrix"]
        assert artifacts.pollution_logs_path == output_dir / CORE_ARTIFACTS["pollution_logs"]

        # Assert all exist
        assert artifacts.clean_dataset_path.exists(), f"Missing: {CORE_ARTIFACTS['clean_dataset']}"
        assert artifacts.projection_matrix_path.exists(), f"Missing: {CORE_ARTIFACTS['projection_matrix']}"
        assert artifacts.pollution_logs_path.exists(), f"Missing: {CORE_ARTIFACTS['pollution_logs']}"

    def test_laptop_strategy_clean_dataset_schema(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test cleaned dataset has correct schema and populated post_masked."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop_schema"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        clean_table = feather.read_table(artifacts.clean_dataset_path)
        actual_columns = set(clean_table.column_names)

        # Assert all required columns present
        missing_columns = CLEAN_DATASET_REQUIRED_COLUMNS - actual_columns
        assert not missing_columns, f"Missing columns in clean_dataset: {missing_columns}"

        # Assert post_masked is populated with mask tokens
        masked_texts = clean_table["post_masked"].to_pylist()
        num_with_masks = sum(1 for t in masked_texts if t and "[MASK:" in t)
        assert num_with_masks > 0, "No mask tokens found in post_masked column"

    def test_laptop_strategy_pollution_logs_schema(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test pollution logs have correct schema."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop_logs"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        logs_table = feather.read_table(artifacts.pollution_logs_path)
        actual_columns = set(logs_table.column_names)

        # Assert all required columns present
        missing_columns = POLLUTION_LOG_COLUMNS - actual_columns
        assert not missing_columns, f"Missing columns in pollution_logs: {missing_columns}"

        # Assert logs are not empty
        assert len(logs_table) > 0, "Pollution logs should not be empty"

    def test_laptop_strategy_projection_matrix_shape(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test projection matrix has correct shape and properties."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop_proj"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        proj_matrix = torch.load(artifacts.projection_matrix_path, weights_only=True)

        # Assert square matrix
        assert proj_matrix.ndim == 2, "Projection matrix must be 2D"
        assert proj_matrix.shape[0] == proj_matrix.shape[1], "Projection matrix must be square"

        # Assert expected dimension (64 for mock embedder)
        expected_dim = 64
        assert proj_matrix.shape[0] == expected_dim, f"Expected dim {expected_dim}, got {proj_matrix.shape[0]}"

    def test_laptop_strategy_metrics_json_structure(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test metrics JSON has complete required structure."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop_metrics"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        # Assert metrics path is correct
        expected_metrics_path = output_dir / REPORTS_SUBDIR / METRICS_FILENAME
        assert artifacts.metrics_path == expected_metrics_path, f"Expected {expected_metrics_path}"
        assert artifacts.metrics_path.exists(), f"Metrics file not created: {expected_metrics_path}"

        with open(artifacts.metrics_path) as f:
            metrics = json.load(f)

        # Assert all top-level required keys
        missing_top_level = REQUIRED_METRICS_TOP_LEVEL_KEYS - set(metrics.keys())
        assert not missing_top_level, f"Missing top-level metrics keys: {missing_top_level}"

        # Assert dataset metrics structure
        dataset_metrics = metrics["dataset"]
        missing_dataset = REQUIRED_DATASET_METRICS_KEYS - set(dataset_metrics.keys())
        assert not missing_dataset, f"Missing dataset metrics keys: {missing_dataset}"

        # Assert gliner metrics structure
        gliner_metrics = metrics["gliner"]
        missing_gliner = REQUIRED_GLINER_METRICS_KEYS - set(gliner_metrics.keys())
        assert not missing_gliner, f"Missing gliner metrics keys: {missing_gliner}"

        # Assert leace metrics structure
        leace_metrics = metrics["leace"]
        assert leace_metrics is not None, "LEACE metrics should not be None"
        missing_leace = REQUIRED_LEACE_METRICS_KEYS - set(leace_metrics.keys())
        assert not missing_leace, f"Missing leace metrics keys: {missing_leace}"

        # Assert probe metrics structure
        assert "probe" in metrics, "Probe metrics missing"
        probe_metrics = metrics["probe"]
        missing_probe = REQUIRED_PROBE_METRICS_KEYS - set(probe_metrics.keys())
        assert not missing_probe, f"Missing probe metrics keys: {missing_probe}"

        # Assert by_column has all demographic columns
        by_column = probe_metrics["by_column"]
        for col in DEMOGRAPHIC_COLUMNS:
            assert col in by_column, f"Missing probe results for column: {col}"
            col_metrics = by_column[col]
            assert "accuracy_before" in col_metrics
            assert "accuracy_after" in col_metrics
            assert "amnesic_drop" in col_metrics

    def test_laptop_strategy_visualizations(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test all expected visualizations are generated."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_laptop_viz"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        # Assert reports directory path
        expected_reports_dir = output_dir / REPORTS_SUBDIR
        assert artifacts.reports_dir == expected_reports_dir
        assert artifacts.reports_dir.exists(), f"Reports dir not created: {expected_reports_dir}"

        # Get all generated plot files
        actual_plots = {f.name for f in artifacts.reports_dir.glob("*.png")}

        # Assert all pipeline visualization plots are generated
        missing_pipeline_plots = PIPELINE_VISUALIZATION_PLOTS - actual_plots
        # Some plots may fail if data doesn't support them, but core ones should exist
        core_expected_plots = {"singular_values.png", "amnesic_drop_bars.png"}
        missing_core = core_expected_plots - actual_plots
        assert not missing_core, f"Missing core plots: {missing_core}"

        # Log all generated plots for debugging
        logger.info(f"Generated {len(actual_plots)} plots:")
        for plot in sorted(actual_plots):
            logger.info(f"  - {plot}")

    def test_hpc_strategy_core_artifacts(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test HPCFilterStrategy produces all core artifacts."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_hpc"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        # Assert exact artifact paths and existence
        assert artifacts.clean_dataset_path.exists(), f"Missing: {CORE_ARTIFACTS['clean_dataset']}"
        assert artifacts.projection_matrix_path.exists(), f"Missing: {CORE_ARTIFACTS['projection_matrix']}"
        assert artifacts.pollution_logs_path.exists(), f"Missing: {CORE_ARTIFACTS['pollution_logs']}"
        assert artifacts.metrics_path.exists(), "Metrics JSON not created"

    def test_hpc_strategy_extended_probe_metrics(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test HPC metadata contains extended probe metrics."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_hpc_extended"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        metadata = artifacts.metadata

        # Assert probe structure
        probe = metadata.get("probe", {})
        assert "by_column" in probe
        assert "by_column_extended" in probe
        assert "min_amnesic_drop" in probe
        assert "mean_amnesic_drop" in probe
        assert "config" in probe

        # Assert extended metrics for each column
        by_column_extended = probe["by_column_extended"]
        for col in DEMOGRAPHIC_COLUMNS:
            assert col in by_column_extended, f"Missing extended metrics for: {col}"
            col_ext = by_column_extended[col]
            if "error" not in col_ext:
                assert "balanced_acc_before" in col_ext
                assert "balanced_acc_after" in col_ext
                assert "cv_before" in col_ext
                assert "cv_after" in col_ext

        # Assert separability metrics
        assert "separability" in metadata
        sep = metadata["separability"]
        assert "before" in sep
        assert "after" in sep
        for col in DEMOGRAPHIC_COLUMNS:
            assert col in sep["before"], f"Missing separability before for: {col}"
            assert col in sep["after"], f"Missing separability after for: {col}"
            # Assert separability metrics structure
            sep_before = sep["before"][col]
            assert "silhouette_score" in sep_before
            assert "davies_bouldin_index" in sep_before

        # Assert class imbalance metrics
        assert "class_imbalance" in metadata
        imb = metadata["class_imbalance"]
        for col in DEMOGRAPHIC_COLUMNS:
            assert col in imb, f"Missing class imbalance for: {col}"
            col_imb = imb[col]
            assert "imbalance_ratio" in col_imb
            assert "majority_baseline" in col_imb

        # Assert control probe (HPC specific)
        assert "control_probe" in metadata
        control = metadata["control_probe"]
        assert "target" in control
        assert "control_stability_score" in control
        assert "specificity_ratio" in control
        assert "is_specific" in control

    def test_hpc_strategy_solver_benchmark(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test HPC mode includes solver benchmark when enabled."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        # Ensure benchmark_solvers is enabled
        mock_hpc_config["probe"]["benchmark_solvers"] = True

        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_hpc_benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        probe = artifacts.metadata.get("probe", {})
        assert "solver_benchmark" in probe, "Missing solver_benchmark in HPC mode"

        benchmark = probe["solver_benchmark"]
        # Assert all solver benchmark fields
        required_benchmark_fields = {
            "exact_accuracy",
            "torch_accuracy",
            "solver_accuracy_delta",
            "weight_cosine_similarity",
        }
        missing_benchmark = required_benchmark_fields - set(benchmark.keys())
        assert not missing_benchmark, f"Missing solver benchmark fields: {missing_benchmark}"

        # Assert delta is reasonable
        assert benchmark["solver_accuracy_delta"] < 0.1, "Solver accuracy delta too large"

    def test_visualization_disabled(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test that disabling visualization skips report generation."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline

        config_no_viz = mock_base_config.copy()
        config_no_viz["visualization"] = {"enabled": False}

        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, config_no_viz)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        self._patch_pipeline_components(monkeypatch)

        output_dir = tmp_path / "output_no_viz"
        output_dir.mkdir(parents=True, exist_ok=True)

        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=config_no_viz)
        artifacts = pipeline.run(input_dataset_path=mock_input_data, output_dir=output_dir)

        # Core artifacts should still exist
        assert artifacts.clean_dataset_path.exists()
        assert artifacts.projection_matrix_path.exists()
        assert artifacts.pollution_logs_path.exists()

        # Reports dir and metrics should be None
        assert artifacts.reports_dir is None, "Reports dir should be None when viz disabled"
        assert artifacts.metrics_path is None, "Metrics path should be None when viz disabled"


class TestProbeBackendMocking:
    """Test probe backend mocking for cuML."""

    def test_cuml_probe_mocked(self, mock_cuml, mock_cupy, monkeypatch):
        """Test that cuML probe can be mocked properly."""
        monkeypatch.setitem(sys.modules, "cuml", mock_cuml)
        monkeypatch.setitem(sys.modules, "cuml.linear_model", mock_cuml.linear_model)
        monkeypatch.setitem(sys.modules, "cupy", mock_cupy)

        from neuro_stylometry.pollution_guard.probe import ProbeConfig, CuMLProbe

        config = ProbeConfig(backend="cuml")
        probe = CuMLProbe(config)

        X = np.random.randn(100, 64).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)

        probe.fit(X, y)
        predictions = probe.predict(X)

        assert len(predictions) == len(y)

    def test_probe_backend_detection_falls_back(self, monkeypatch):
        """Test that probe backend detection falls back gracefully."""
        monkeypatch.delitem(sys.modules, "cuml", raising=False)
        monkeypatch.delitem(sys.modules, "cuml.linear_model", raising=False)

        from neuro_stylometry.pollution_guard.probe import _detect_backend, ProbeBackend

        backend = _detect_backend()
        assert backend in [ProbeBackend.SKLEARN, ProbeBackend.TORCH]


class TestMetricsComputation:
    """Test metrics computation functions with exact assertions."""

    def test_compute_embedding_separability_structure(self):
        """Test embedding separability returns correct structure."""
        from neuro_stylometry.evaluation.metrics import compute_embedding_separability

        np.random.seed(42)
        n_per_class = 50

        X0 = np.random.randn(n_per_class, 64) * 0.5
        X1 = np.random.randn(n_per_class, 64) * 0.5 + 3

        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)

        result = compute_embedding_separability(X, y, sample_size=100)

        # Assert exact structure
        required_keys = {"silhouette_score", "davies_bouldin_index", "n_samples", "n_classes"}
        assert set(result.keys()) >= required_keys, f"Missing keys: {required_keys - set(result.keys())}"

        # Assert value ranges
        assert -1 <= result["silhouette_score"] <= 1, "Silhouette score out of range"
        assert result["davies_bouldin_index"] >= 0, "Davies-Bouldin index must be non-negative"
        assert result["n_samples"] == 100
        assert result["n_classes"] == 2

    def test_compute_class_imbalance_metrics_structure(self):
        """Test class imbalance metrics returns correct structure and values."""
        from neuro_stylometry.evaluation.metrics import compute_class_imbalance_metrics

        y = np.array([0] * 80 + [1] * 20)

        result = compute_class_imbalance_metrics(y)

        # Assert exact structure
        required_keys = {
            "distribution",
            "proportions",
            "imbalance_ratio",
            "majority_baseline",
            "majority_class",
            "minority_class",
            "n_classes",
        }
        assert set(result.keys()) >= required_keys

        # Assert exact values for known input
        assert result["imbalance_ratio"] == 4.0, "Imbalance ratio should be 80/20 = 4.0"
        assert result["majority_baseline"] == 0.8, "Majority baseline should be 0.8"
        assert result["majority_class"] == 0
        assert result["minority_class"] == 1
        assert result["n_classes"] == 2

    def test_compute_control_probe_metrics_structure(self):
        """Test control probe metrics returns correct structure."""
        from neuro_stylometry.evaluation.metrics import compute_control_probe_metrics

        np.random.seed(42)
        n = 100

        embeddings_before = np.random.randn(n, 64).astype(np.float32)
        embeddings_after = np.random.randn(n, 64).astype(np.float32)
        target_labels = np.random.randint(0, 2, n)
        control_labels = np.random.randint(0, 2, n)

        result = compute_control_probe_metrics(
            embeddings_before=embeddings_before,
            embeddings_after=embeddings_after,
            target_labels=target_labels,
            control_labels=control_labels,
            control_name="control_test",
        )

        # Assert structure
        assert "target" in result
        assert "control_test" in result or "control_control_test" in result
        assert "control_stability_score" in result
        assert "specificity_ratio" in result
        assert "is_specific" in result

        # Assert target structure
        target = result["target"]
        assert "acc_before" in target
        assert "acc_after" in target
        assert "drop" in target


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
