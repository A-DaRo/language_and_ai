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
from neuro_stylometry.pollution_guard.leace import LEACEComputer
from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
from neuro_stylometry.data_engine.dataset import SOBRDataset
from neuro_stylometry.config import load_pipeline_config
from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType


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
def varied_length_texts():
    short = "hello world"
    medium = " ".join(["token"] * 128)
    long = " ".join(["token"] * 600)  # will be truncated to max_length
    return [short, medium, long]


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
    
    def test_embedder_initialization(self, require_real_models, frozen_embedder, device):
        """Test that embedder initializes correctly."""
        assert frozen_embedder.model is not None
        assert str(frozen_embedder.device) == device
        assert frozen_embedder.embedding_dim == 768  # RoBERTa-base
    
    def test_embedding_extraction(self, require_real_models, frozen_embedder, sample_texts):
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

    def test_embedding_varied_lengths(self, require_real_models, frozen_embedder, varied_length_texts):
        embeddings = frozen_embedder.embed_texts(
            varied_length_texts,
            batch_size=2,
            show_progress=False,
        )
        assert embeddings.shape == (len(varied_length_texts), 768)
    
    def test_frozen_weights(self, require_real_models, frozen_embedder):
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
    
    def test_projection_computation_concepts(self, device):
        """Test projection computation with multi-dimensional concept matrix Z."""
        torch.manual_seed(42)

        n = 200
        d = 64

        z1 = torch.randint(0, 2, (n, 1), dtype=torch.float32)
        z2 = torch.randint(0, 2, (n, 1), dtype=torch.float32)
        Z = torch.cat([z1, z2], dim=1)  # (n, 2)

        noise = torch.randn(n, d)
        leak = torch.zeros(d)
        leak[0] = 2.0
        leak[1] = -1.5
        X = noise + z1 * leak.unsqueeze(0) + z2 * torch.roll(leak, shifts=1).unsqueeze(0)

        leace = LEACEComputer(embedding_dim=d, regularization=1e-5, device=device)
        P = leace.compute_projection_from_concepts(X, Z)

        assert P.shape == (d, d)
        assert P.dtype == torch.float32

        P_cpu = P.detach().to("cpu", dtype=torch.float32)
        Xp = X @ P_cpu.T
        Zc = Z - Z.mean(dim=0, keepdim=True)
        Xpc = Xp - Xp.mean(dim=0, keepdim=True)
        cross = (Xpc.T @ Zc) / float(n)
        assert torch.linalg.norm(cross).item() < 0.2
    
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
    
    def test_batch_accumulation_concepts(self, device):
        """Test concept-stat batch accumulation for low-memory mode."""
        torch.manual_seed(42)

        n = 200
        d = 32
        k = 4
        bs = 50

        X = torch.randn(n, d)
        Z = torch.randn(n, k)

        leace = LEACEComputer(embedding_dim=d, regularization=1e-5, device=device)

        P_full = leace.compute_projection_from_concepts(X, Z)

        acc = None
        for i in range(0, n, bs):
            acc = leace.accumulate_batch_concepts(X[i:i + bs], Z[i:i + bs], acc)

        P_acc = leace.compute_projection_from_concept_stats(acc)

        difference = torch.abs(P_full - P_acc).max().item()
        assert difference < 0.1, f"Difference {difference:.2e} too large"


class TestLaptopFilterStrategy:
    """Test laptop filter strategy end-to-end."""
    
    def test_strategy_initialization(self):
        """Test laptop strategy initialization."""
        strategy = LaptopFilterStrategy()
        
        assert strategy.get_device() is not None
        assert isinstance(strategy.get_batch_size(), int)
    
    def test_end_to_end_execution(self, require_real_models, laptop_dataset_path, output_paths):
        """Test end-to-end Phase A execution with laptop strategy."""
        # Implements FR-11: Full pipeline
        profile = HardwareDetector.detect()
        if profile.profile_type == ProfileType.HPC:
            pytest.skip("Laptop strategy E2E skipped on HPC profile")

        strategy = LaptopFilterStrategy()
        
        test_cfg = Path("tests/fixtures/pipeline_small.yaml")
        config = load_pipeline_config(mode="laptop", experiment_config_path=test_cfg)
        
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
    
    def test_demographic_data_present(self, laptop_dataset_path):
        """Test that demographic columns are present in dataset."""
        from neuro_stylometry.data_engine.schemas import get_demographic_columns
        
        dataset = SOBRDataset(arrow_path=laptop_dataset_path, seed=42)
        table = dataset.table
        
        # Check demographic columns exist
        demographic_cols = get_demographic_columns()
        for col in demographic_cols:
            assert col in table.column_names, f"Demographic column '{col}' missing"
        
        # Check values are valid
        female_col = table.column("female")
        assert female_col.null_count < len(table), "All female values are null"
