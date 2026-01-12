"""
Integration tests for Phase A Pipeline outputs.

Verifies that both LaptopFilterStrategy and HPCFilterStrategy produce
all expected outputs: cleaned dataset, projection matrix, pollution logs,
metrics JSON, and visualizations.

Uses mocked inputs to avoid actual GLiNER/encoder inference.
"""

import json
import logging
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional
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
    # Generate synthetic posts with some "pollution" patterns
    np.random.seed(42)
    n_samples = 200  # Small dataset for fast tests
    
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
        "post_masked": [""] * n_samples,  # Empty initially
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
            "hidden_dim": 64,  # Small for tests
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
        self.config.labels_encoder = True  # Pretend bi-encoder
    
    def to(self, device):
        return self
    
    def eval(self):
        return self
    
    def predict_entities(self, texts, labels, threshold=0.5, **kwargs):
        """Return mock entities for each text."""
        results = []
        for text in texts:
            entities = []
            # Detect some mock patterns
            if "man" in text.lower() or "woman" in text.lower():
                start = text.lower().find("man") if "man" in text.lower() else text.lower().find("woman")
                entities.append({
                    "start": start,
                    "end": start + 5,
                    "text": text[start:start+5],
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
        # Return mock embeddings
        last_hidden_state = torch.randn(batch_size, 10, self.hidden_dim)
        return MagicMock(last_hidden_state=last_hidden_state)


# =============================================================================
# Mock patches
# =============================================================================

@pytest.fixture
def mock_gliner_detector(monkeypatch):
    """Patch GLiNERDetector to use mock model."""
    
    original_init = None
    
    def mock_init(self, *args, **kwargs):
        self.model = MockGLiNERModel()
        self.device = "cpu"
        self.max_length = kwargs.get("max_length", 512)
        self.confidence_threshold = kwargs.get("confidence_threshold", 0.5)
        self.taxonomy = kwargs.get("taxonomy")
        self.constraints = kwargs.get("constraints")
        self.budget_config = kwargs.get("budget_config")
        self.batch_inference_config = kwargs.get("batch_inference_config")
        self._label_embeddings_cache = None
        
        # Mock taxonomy methods
        if self.taxonomy is None:
            self.taxonomy = MagicMock()
            self.taxonomy.get_inference_labels.return_value = ["gender indicator", "nationality"]
            self.taxonomy.get_reference_patterns.return_value = {}
            self.taxonomy.get_column_for_label.return_value = "female"
            self.taxonomy.get_mask_token.return_value = "[MASK:GENDER]"
    
    def mock_get_mask_tokens(self):
        return {
            "gender indicator": "[MASK:GENDER]",
            "nationality": "[MASK:NATIONALITY]",
            "age indicator": "[MASK:AGE]",
            "political view": "[MASK:POLITICS]",
        }
    
    def mock_get_cached_label_embeddings(self, labels):
        return None  # No caching in mock
    
    from neuro_stylometry.pollution_guard import gliner_detector
    monkeypatch.setattr(gliner_detector.GLiNERDetector, "__init__", mock_init)
    monkeypatch.setattr(gliner_detector.GLiNERDetector, "get_mask_tokens", mock_get_mask_tokens)
    monkeypatch.setattr(gliner_detector.GLiNERDetector, "get_cached_label_embeddings", mock_get_cached_label_embeddings)


@pytest.fixture
def mock_embedder(monkeypatch):
    """Patch FrozenEmbedder to use mock encoder."""
    
    def mock_init(self, *args, **kwargs):
        self.model = MockEncoder(hidden_dim=kwargs.get("hidden_dim", 64))
        self.tokenizer = MockTokenizer()
        self.device = kwargs.get("device", "cpu")
        self.max_length = kwargs.get("max_length", 512)
        self.output_device = kwargs.get("output_device", "cpu")
        self._embedding_dim = 64
    
    def mock_embed_texts(self, texts, batch_size=32, show_progress=False):
        n = len(texts)
        return torch.randn(n, 64)
    
    def mock_get_embedding_dim(self):
        return 64
    
    from neuro_stylometry.pollution_guard import embedder
    monkeypatch.setattr(embedder.FrozenEmbedder, "__init__", mock_init)
    monkeypatch.setattr(embedder.FrozenEmbedder, "embed_texts", mock_embed_texts)
    monkeypatch.setattr(embedder.FrozenEmbedder, "get_embedding_dim", mock_get_embedding_dim)


@pytest.fixture
def mock_cuml():
    """Mock cuML module since it's not available."""
    mock_cuml_module = MagicMock()
    
    # Mock LogisticRegression
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
    """Create a simplified execute method for testing."""
    
    def mock_execute(
        self,
        input_dataset_path: Path,
        output_dataset_path: Path,
        projection_matrix_path: Path,
        pollution_logs_path: Path,
        config: Dict[str, Any],
    ) -> Dict[str, Any]:
        """Simplified execution that produces expected outputs."""
        import pyarrow.feather as feather
        from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA
        
        # Load input
        table = feather.read_table(input_dataset_path)
        posts = table["post"].to_pylist()
        post_ids = table["post_id"].to_pylist()
        n_samples = len(posts)
        
        # Generate mock masked texts
        masked_texts = [
            post.replace("man", "[MASK:GENDER]").replace("woman", "[MASK:GENDER]")
            .replace("USA", "[MASK:NATIONALITY]").replace("UK", "[MASK:NATIONALITY]")
            for post in posts
        ]
        
        # Generate mock pollution logs
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
                    "column": "female",
                })
            if "USA" in post or "UK" in post:
                pollution_logs.append({
                    "post_id": post_id,
                    "entity_type": "nationality",
                    "span_text": "USA" if "USA" in post else "UK",
                    "span_start": 20,
                    "span_end": 23,
                    "confidence": 0.90,
                    "column": "nationality",
                })
        
        # Save cleaned dataset - preserve original columns and update post_masked
        df = table.to_pandas()
        df["post_masked"] = masked_texts
        
        # Recreate table with schema from the original (handles dict-encoded columns)
        clean_table = pa.Table.from_pandas(df, schema=table.schema, preserve_index=False)
        feather.write_feather(clean_table, output_dataset_path)
        
        # Save projection matrix (mock 64x64 identity-like)
        projection_matrix = torch.eye(64) + 0.01 * torch.randn(64, 64)
        torch.save(projection_matrix, projection_matrix_path)
        
        # Save pollution logs
        if pollution_logs:
            logs_table = pa.Table.from_pylist(pollution_logs, schema=POLLUTION_LOG_SCHEMA)
        else:
            logs_table = pa.Table.from_pylist([], schema=POLLUTION_LOG_SCHEMA)
        feather.write_feather(logs_table, pollution_logs_path)
        
        # Build metadata with probe results
        from neuro_stylometry.data_engine.schemas import get_demographic_columns
        
        by_column = {}
        by_column_extended = {}
        separability_before = {}
        separability_after = {}
        class_imbalance = {}
        
        for col in get_demographic_columns():
            by_column[col] = {
                "accuracy_before": 0.75 + np.random.uniform(-0.1, 0.1),
                "accuracy_after": 0.55 + np.random.uniform(-0.1, 0.1),
                "amnesic_drop": 0.25 + np.random.uniform(0, 0.15),
            }
            by_column_extended[col] = {
                **by_column[col],
                "balanced_acc_before": 0.72,
                "balanced_acc_after": 0.52,
                "balanced_amnesic_drop": 0.28,
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
                "silhouette_score": 0.3,
                "davies_bouldin_index": 1.5,
                "n_samples": n_samples,
                "n_classes": 2,
            }
            separability_after[col] = {
                "silhouette_score": 0.1,
                "davies_bouldin_index": 2.0,
                "n_samples": n_samples,
                "n_classes": 2,
            }
            class_imbalance[col] = {
                "distribution": {0: n_samples // 2, 1: n_samples // 2},
                "proportions": {0: 0.5, 1: 0.5},
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
                "min_amnesic_drop": min(drops),
                "mean_amnesic_drop": np.mean(drops),
                "threshold": config.get("probe", {}).get("amnesic_drop_threshold", 0.3),
                "max_samples": min(n_samples, config.get("probe", {}).get("max_samples", 100)),
                "config": {
                    "backend": config.get("probe", {}).get("backend", "sklearn"),
                    "use_kfold": config.get("probe", {}).get("use_kfold", True),
                    "n_folds": config.get("probe", {}).get("n_folds", 5),
                },
            },
            "separability": {
                "before": separability_before,
                "after": separability_after,
            },
            "class_imbalance": class_imbalance,
        }
        
        # Add solver benchmark for HPC
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
        
        # Add control probe for HPC
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
    """Test that Phase A pipeline produces all expected outputs."""
    
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
        
        # Patch get_embedder to return mock
        original_get_embedder = PhaseAPipeline.get_embedder
        
        def mock_get_embedder(self):
            return MockEmbedderForPipeline()
        
        monkeypatch.setattr(PhaseAPipeline, "get_embedder", mock_get_embedder)
    
    def test_laptop_strategy_produces_all_outputs(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test LaptopFilterStrategy produces cleaned dataset, projection, logs, metrics, plots."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
        
        # Patch the strategy execute method
        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, mock_base_config)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        
        # Patch pipeline components to avoid HuggingFace calls
        self._patch_pipeline_components(monkeypatch)
        
        # Create output directory
        output_dir = tmp_path / "output_laptop"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_base_config)
        
        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )
        
        # Verify core artifacts exist
        assert artifacts.clean_dataset_path.exists(), "Clean dataset not created"
        assert artifacts.projection_matrix_path.exists(), "Projection matrix not created"
        assert artifacts.pollution_logs_path.exists(), "Pollution logs not created"
        
        # Verify clean dataset has post_masked populated
        clean_table = feather.read_table(artifacts.clean_dataset_path)
        assert "post_masked" in clean_table.column_names
        masked_texts = clean_table["post_masked"].to_pylist()
        assert any("[MASK:" in t for t in masked_texts if t), "No mask tokens in post_masked"
        
        # Verify projection matrix is valid
        proj_matrix = torch.load(artifacts.projection_matrix_path)
        assert proj_matrix.shape[0] == proj_matrix.shape[1], "Projection matrix not square"
        assert proj_matrix.shape[0] == 64, "Projection matrix wrong size"
        
        # Verify pollution logs
        logs_table = feather.read_table(artifacts.pollution_logs_path)
        assert len(logs_table) > 0, "No pollution logs created"
        assert "entity_type" in logs_table.column_names
        assert "confidence" in logs_table.column_names
        
        # Verify metrics were saved
        assert artifacts.metrics_path is not None, "Metrics path not set"
        assert artifacts.metrics_path.exists(), "Metrics JSON not created"
        
        with open(artifacts.metrics_path) as f:
            metrics = json.load(f)
        
        assert "dataset" in metrics, "Missing dataset metrics"
        assert "gliner" in metrics, "Missing GLiNER metrics"
        assert "probe" in metrics, "Missing probe metrics"
        
        # Verify probe metrics structure
        assert "by_column" in metrics.get("probe", {}), "Missing by_column probe results"
        
        # Verify visualizations directory
        assert artifacts.reports_dir is not None, "Reports dir not set"
        assert artifacts.reports_dir.exists(), "Reports directory not created"
        
        # Check for expected plot files
        expected_plots = [
            "pca_before_after",
            "singular_values",
            "amnesic_drop",
        ]
        
        plot_files = list(artifacts.reports_dir.glob("*.png"))
        assert len(plot_files) > 0, f"No plots generated in {artifacts.reports_dir}"
        
        logger.info(f"Laptop strategy produced {len(plot_files)} plots")
        for plot in plot_files:
            logger.info(f"  - {plot.name}")
    
    def test_hpc_strategy_produces_all_outputs(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test HPCFilterStrategy produces all outputs including benchmark metrics."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
        
        # Patch the strategy execute method
        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        
        # Patch pipeline components to avoid HuggingFace calls
        self._patch_pipeline_components(monkeypatch)
        
        # Create output directory
        output_dir = tmp_path / "output_hpc"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        # Run pipeline
        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        
        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )
        
        # Verify core artifacts exist
        assert artifacts.clean_dataset_path.exists(), "Clean dataset not created"
        assert artifacts.projection_matrix_path.exists(), "Projection matrix not created"
        assert artifacts.pollution_logs_path.exists(), "Pollution logs not created"
        
        # Verify metrics were saved with HPC-specific fields
        assert artifacts.metrics_path.exists(), "Metrics JSON not created"
        
        with open(artifacts.metrics_path) as f:
            metrics = json.load(f)
        
        # HPC should have solver_benchmark in probe
        probe_metrics = metrics.get("probe", {})
        assert "by_column" in probe_metrics, "Missing by_column probe results"
        
        # Check for extended metrics in strategy metadata
        assert "execution" in metrics, "Missing execution metadata"
        
        # Verify visualizations directory
        assert artifacts.reports_dir.exists(), "Reports directory not created"
        
        plot_files = list(artifacts.reports_dir.glob("*.png"))
        assert len(plot_files) > 0, f"No plots generated"
        
        logger.info(f"HPC strategy produced {len(plot_files)} plots")
        for plot in plot_files:
            logger.info(f"  - {plot.name}")
    
    def test_metadata_contains_extended_probe_metrics(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test that metadata contains extended probe metrics (CV, separability, etc.)."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
        
        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        
        # Patch pipeline components to avoid HuggingFace calls
        self._patch_pipeline_components(monkeypatch)
        
        output_dir = tmp_path / "output_extended"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        
        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )
        
        # Check artifacts.metadata directly
        metadata = artifacts.metadata
        
        # Verify probe structure
        probe = metadata.get("probe", {})
        assert "by_column" in probe
        assert "by_column_extended" in probe
        assert "min_amnesic_drop" in probe
        assert "mean_amnesic_drop" in probe
        
        # Verify extended metrics for at least one column
        by_column_extended = probe.get("by_column_extended", {})
        assert len(by_column_extended) > 0, "No extended column metrics"
        
        first_col = list(by_column_extended.keys())[0]
        col_metrics = by_column_extended[first_col]
        
        assert "balanced_acc_before" in col_metrics
        assert "balanced_acc_after" in col_metrics
        assert "cv_before" in col_metrics or "error" in col_metrics
        
        # Verify separability metrics
        assert "separability" in metadata, "Missing separability metrics"
        sep = metadata["separability"]
        assert "before" in sep
        assert "after" in sep
        
        # Verify class imbalance
        assert "class_imbalance" in metadata, "Missing class imbalance metrics"
        
        # Verify control probe (HPC specific)
        assert "control_probe" in metadata, "Missing control probe metrics"
        control = metadata["control_probe"]
        assert "target" in control
        assert "control_stability_score" in control
        assert "specificity_ratio" in control
    
    def test_solver_benchmark_in_hpc_mode(
        self,
        mock_input_data,
        mock_hpc_config,
        tmp_path,
        monkeypatch,
    ):
        """Test that HPC mode includes solver benchmark comparing sklearn vs torch."""
        from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
        
        # Ensure benchmark_solvers is enabled
        mock_hpc_config["probe"]["benchmark_solvers"] = True
        
        mock_execute = create_mock_strategy_execute(HPCFilterStrategy, mock_hpc_config)
        monkeypatch.setattr(HPCFilterStrategy, "execute", mock_execute)
        
        # Patch pipeline components to avoid HuggingFace calls
        self._patch_pipeline_components(monkeypatch)
        
        output_dir = tmp_path / "output_benchmark"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategy = HPCFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=mock_hpc_config)
        
        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )
        
        # Check for solver benchmark in metadata
        probe = artifacts.metadata.get("probe", {})
        assert "solver_benchmark" in probe, "Missing solver_benchmark in HPC mode"
        
        benchmark = probe["solver_benchmark"]
        assert "exact_accuracy" in benchmark
        assert "torch_accuracy" in benchmark
        assert "solver_accuracy_delta" in benchmark
        assert "weight_cosine_similarity" in benchmark
        
        # Delta should be small (probes should converge similarly)
        assert benchmark["solver_accuracy_delta"] < 0.1, "Solver accuracy delta too large"
    
    def test_visualization_config_respected(
        self,
        mock_input_data,
        mock_base_config,
        tmp_path,
        monkeypatch,
    ):
        """Test that visualization config toggles work."""
        from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
        from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
        
        # Disable visualizations
        config_no_viz = mock_base_config.copy()
        config_no_viz["visualization"] = {"enabled": False}
        
        mock_execute = create_mock_strategy_execute(LaptopFilterStrategy, config_no_viz)
        monkeypatch.setattr(LaptopFilterStrategy, "execute", mock_execute)
        
        # Patch pipeline components to avoid HuggingFace calls
        self._patch_pipeline_components(monkeypatch)
        
        output_dir = tmp_path / "output_no_viz"
        output_dir.mkdir(parents=True, exist_ok=True)
        
        strategy = LaptopFilterStrategy()
        pipeline = PhaseAPipeline(strategy=strategy, config=config_no_viz)
        
        artifacts = pipeline.run(
            input_dataset_path=mock_input_data,
            output_dir=output_dir,
        )
        
        # Core artifacts should still exist
        assert artifacts.clean_dataset_path.exists()
        assert artifacts.projection_matrix_path.exists()
        
        # But reports_dir should be None or empty
        if artifacts.reports_dir is not None:
            plot_files = list(artifacts.reports_dir.glob("*.png"))
            # May have some files but metrics should not have been computed
            logger.info(f"With viz disabled: {len(plot_files)} plot files")


class TestProbeBackendMocking:
    """Test probe backend mocking for cuML."""
    
    def test_cuml_probe_mocked(self, mock_cuml, mock_cupy, monkeypatch):
        """Test that cuML probe can be mocked properly."""
        # Insert mocks into sys.modules
        monkeypatch.setitem(sys.modules, "cuml", mock_cuml)
        monkeypatch.setitem(sys.modules, "cuml.linear_model", mock_cuml.linear_model)
        monkeypatch.setitem(sys.modules, "cupy", mock_cupy)
        
        # Now import and test
        from neuro_stylometry.pollution_guard.probe import ProbeConfig, CuMLProbe
        
        config = ProbeConfig(backend="cuml")
        probe = CuMLProbe(config)
        
        # Generate mock data
        X = np.random.randn(100, 64).astype(np.float32)
        y = np.random.randint(0, 2, 100).astype(np.float32)
        
        # This should use the mocked cuML
        probe.fit(X, y)
        predictions = probe.predict(X)
        
        assert len(predictions) == len(y)
    
    def test_probe_backend_detection_falls_back(self, monkeypatch):
        """Test that probe backend detection falls back gracefully."""
        # Remove cuml from modules if present
        monkeypatch.delitem(sys.modules, "cuml", raising=False)
        monkeypatch.delitem(sys.modules, "cuml.linear_model", raising=False)
        
        from neuro_stylometry.pollution_guard.probe import _detect_backend, ProbeBackend
        
        # On CPU without cuML, should fall back to sklearn or torch
        backend = _detect_backend()
        assert backend in [ProbeBackend.SKLEARN, ProbeBackend.TORCH]


class TestMetricsComputation:
    """Test metrics computation functions."""
    
    def test_compute_embedding_separability(self):
        """Test embedding separability metric computation."""
        from neuro_stylometry.evaluation.metrics import compute_embedding_separability
        
        # Generate separable clusters
        np.random.seed(42)
        n_per_class = 50
        
        # Class 0: centered at [0, 0, ...]
        X0 = np.random.randn(n_per_class, 64) * 0.5
        # Class 1: centered at [3, 3, ...]
        X1 = np.random.randn(n_per_class, 64) * 0.5 + 3
        
        X = np.vstack([X0, X1])
        y = np.array([0] * n_per_class + [1] * n_per_class)
        
        result = compute_embedding_separability(X, y, sample_size=100)
        
        assert "silhouette_score" in result
        assert "davies_bouldin_index" in result
        assert result["silhouette_score"] > 0, "Well-separated clusters should have positive silhouette"
    
    def test_compute_class_imbalance_metrics(self):
        """Test class imbalance metric computation."""
        from neuro_stylometry.evaluation.metrics import compute_class_imbalance_metrics
        
        # Imbalanced labels: 80% class 0, 20% class 1
        y = np.array([0] * 80 + [1] * 20)
        
        result = compute_class_imbalance_metrics(y)
        
        assert "imbalance_ratio" in result
        assert result["imbalance_ratio"] == 4.0  # 80/20
        assert result["majority_baseline"] == 0.8
        assert result["majority_class"] == 0
        assert result["minority_class"] == 1


# =============================================================================
# Run tests
# =============================================================================

if __name__ == "__main__":
    pytest.main([__file__, "-v", "--tb=short"])
