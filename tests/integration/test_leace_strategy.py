"""
Integration Test: LEACE & Strategy Pattern

Validates:
- FrozenEmbedder extracts embeddings correctly
- LEACEComputer computes projection matrix
- Projection matrix is idempotent (P^2 ≈ P)
- LaptopFilterStrategy executes end-to-end
- Strategy pattern integration
"""

import pytest
import torch
from pathlib import Path
import numpy as np

from neuro_stylometry.pollution_guard.embedder import FrozenEmbedder
from neuro_stylometry.pollution_guard.leace import LEACEComputer, CovarianceStats
from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
from neuro_stylometry.data_engine.dataset import SOBRDataset


@pytest.fixture
def device():
    """Get available device."""
    return "cuda" if torch.cuda.is_available() else "cpu"


@pytest.fixture
def frozen_embedder(device):
    """Initialize frozen embedder."""
    return FrozenEmbedder(
        model_name="roberta-base",
        device=device,
        max_length=512,
    )


@pytest.fixture
def sample_texts():
    """Sample texts for embedding."""
    return [
        "The quick brown fox jumps over the lazy dog.",
        "Machine learning is transforming technology.",
        "Natural language processing enables computers to understand text.",
        "Python is a versatile programming language.",
    ]


@pytest.fixture
def laptop_dataset_path():
    """Path to laptop dataset."""
    path = Path("artifacts/data/sobr_laptop.arrow")
    if not path.exists():
        pytest.skip(f"Laptop dataset not found at {path}")
    return path


@pytest.fixture
def output_paths(tmp_path):
    """Create temporary output paths for testing."""
    return {
        "clean_dataset": tmp_path / "clean_sobr.arrow",
        "projection_matrix": tmp_path / "projection_matrix.pt",
        "pollution_logs": tmp_path / "pollution_logs.arrow",
    }


class TestFrozenEmbedder:
    """Test frozen embedder functionality."""
    
    def test_embedder_initialization(self, frozen_embedder, device):
        """Test that embedder initializes correctly."""
        assert frozen_embedder.model is not None
        assert str(frozen_embedder.device) == device
        assert frozen_embedder.embedding_dim == 768  # RoBERTa-base
    
    def test_embedding_extraction(self, frozen_embedder, sample_texts):
        """Test CLS embedding extraction."""
        # Implements FR-09: Frozen embedding
        embeddings = frozen_embedder.embed_texts(
            sample_texts,
            batch_size=2,
            show_progress=False,
        )
        
        # Check shape
        assert embeddings.shape == (len(sample_texts), 768)
        
        # Check dtype
        assert embeddings.dtype == torch.float32
        
        # Check not all zeros
        assert not torch.allclose(embeddings, torch.zeros_like(embeddings))
    
    def test_frozen_weights(self, frozen_embedder):
        """Test that model weights are frozen."""
        for param in frozen_embedder.model.parameters():
            assert not param.requires_grad, "Model weights should be frozen"


class TestLEACEComputer:
    """Test LEACE projection computation."""
    
    def test_leace_initialization(self, device):
        """Test LEACE computer initialization."""
        leace = LEACEComputer(
            embedding_dim=768,
            regularization=1e-5,
            device=device,
        )
        
        assert leace.embedding_dim == 768
        assert leace.regularization == 1e-5
    
    def test_projection_computation(self, device):
        """Test projection matrix computation."""
        # Implements FR-10: LEACE math
        # Create synthetic embeddings with 2 classes
        torch.manual_seed(42)
        
        num_samples = 100
        embedding_dim = 64
        
        # Class 0: centered at [1, 0, 0, ...]
        embeddings_0 = torch.randn(num_samples // 2, embedding_dim) + torch.cat([
            torch.ones(1), torch.zeros(embedding_dim - 1)
        ])
        
        # Class 1: centered at [-1, 0, 0, ...]
        embeddings_1 = torch.randn(num_samples // 2, embedding_dim) + torch.cat([
            -torch.ones(1), torch.zeros(embedding_dim - 1)
        ])
        
        embeddings = torch.cat([embeddings_0, embeddings_1], dim=0)
        labels = torch.cat([
            torch.zeros(num_samples // 2, dtype=torch.long),
            torch.ones(num_samples // 2, dtype=torch.long),
        ])
        
        # Compute projection
        leace = LEACEComputer(
            embedding_dim=embedding_dim,
            regularization=1e-5,
            device=device,
        )
        
        projection_matrix = leace.compute_projection(embeddings, labels)
        
        # Check shape
        assert projection_matrix.shape == (embedding_dim, embedding_dim)
        
        # Check dtype
        assert projection_matrix.dtype == torch.float32
    
    def test_idempotence(self, device):
        """Test that projection matrix is idempotent (P^2 ≈ P)."""
        # Implements FR-10: Idempotence verification
        torch.manual_seed(42)
        
        # Create simple synthetic data
        num_samples = 50
        embedding_dim = 32
        
        embeddings = torch.randn(num_samples, embedding_dim)
        labels = torch.randint(0, 2, (num_samples,))
        
        leace = LEACEComputer(
            embedding_dim=embedding_dim,
            regularization=1e-5,
            device=device,
        )
        
        P = leace.compute_projection(embeddings, labels)
        
        # Compute P^2
        P_squared = P @ P
        
        # Check idempotence: |P^2 - P| < 1e-5
        error = torch.abs(P_squared - P).max().item()
        
        assert error < 1e-5, f"Idempotence error {error:.2e} exceeds tolerance"
    
    def test_batch_accumulation(self, device):
        """Test batch accumulation for low-memory mode."""
        # Implements FR-11: Batch accumulation
        torch.manual_seed(42)
        
        num_samples = 100
        embedding_dim = 32
        batch_size = 25
        
        # Full data
        embeddings = torch.randn(num_samples, embedding_dim)
        labels = torch.randint(0, 2, (num_samples,))
        
        leace = LEACEComputer(
            embedding_dim=embedding_dim,
            regularization=1e-5,
            device=device,
        )
        
        # Method 1: Full-batch
        P_full = leace.compute_projection(embeddings, labels)
        
        # Method 2: Batch accumulation
        accumulated_stats = None
        for i in range(0, num_samples, batch_size):
            batch_embeddings = embeddings[i:i + batch_size]
            batch_labels = labels[i:i + batch_size]
            
            accumulated_stats = leace.accumulate_batch(
                batch_embeddings,
                batch_labels,
                accumulated_stats,
            )
        
        P_accumulated = leace.compute_projection_from_stats(accumulated_stats)
        
        # Compare (should be similar but may differ due to accumulation)
        # Allow higher tolerance for accumulated version
        difference = torch.abs(P_full - P_accumulated).max().item()
        
        # Accumulated version may have higher numerical error
        assert difference < 0.1, f"Difference {difference:.2e} too large"


class TestLaptopFilterStrategy:
    """Test laptop filter strategy end-to-end."""
    
    def test_strategy_initialization(self):
        """Test laptop strategy initialization."""
        strategy = LaptopFilterStrategy()
        
        assert strategy.get_device() is not None
        assert strategy.get_batch_size() > 0
    
    def test_end_to_end_execution(self, laptop_dataset_path, output_paths):
        """Test end-to-end Phase A execution with laptop strategy."""
        # Implements FR-11: Full pipeline
        strategy = LaptopFilterStrategy()
        
        config = {
            "seed": 42,
            "max_samples": 50,  # Small subset for testing
            "gliner_threshold": 0.85,
            "embedder_model": "roberta-base",
            "leace_regularization": 1e-5,
            "leace_batch_size": 25,
            "projection_label": "nationality",
        }
        
        metadata = strategy.execute(
            input_dataset_path=laptop_dataset_path,
            output_dataset_path=output_paths["clean_dataset"],
            projection_matrix_path=output_paths["projection_matrix"],
            pollution_logs_path=output_paths["pollution_logs"],
            config=config,
        )
        
        # Check metadata
        assert "num_samples" in metadata
        assert "num_pollution_spans" in metadata
        assert "projection_matrix_shape" in metadata
        
        # Check outputs exist
        assert output_paths["clean_dataset"].exists(), "Clean dataset not created"
        assert output_paths["projection_matrix"].exists(), "Projection matrix not saved"
        assert output_paths["pollution_logs"].exists(), "Pollution logs not saved"
        
        # Load and validate projection matrix
        P = torch.load(output_paths["projection_matrix"])
        
        # Check shape (should be square)
        assert P.shape[0] == P.shape[1], "Projection matrix not square"
        assert P.shape[0] == 768, "Projection matrix wrong dimension (expected RoBERTa 768)"
        
        # Check idempotence
        P_squared = P @ P
        error = torch.abs(P_squared - P).max().item()
        assert error < 1e-4, f"Projection matrix not idempotent: error={error:.2e}"
        
        print(f"\nPhase A Execution Summary:")
        print(f"  Samples processed: {metadata['num_samples']}")
        print(f"  Pollution spans detected: {metadata['num_pollution_spans']}")
        print(f"  Projection matrix shape: {metadata['projection_matrix_shape']}")
        print(f"  Idempotence error: {error:.2e}")
    
    def test_label_extraction(self, laptop_dataset_path):
        """Test demographic label extraction."""
        strategy = LaptopFilterStrategy()
        
        dataset = SOBRDataset(arrow_path=laptop_dataset_path, seed=42)
        table = dataset.table.slice(0, 10)  # Small subset
        
        config = {"projection_label": "nationality"}
        
        labels = strategy._extract_labels(table, config)
        
        # Check shape
        assert len(labels) == len(table)
        
        # Check dtype
        assert labels.dtype == torch.long
