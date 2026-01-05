# tests/unit/test_leace.py
"""
Unit tests for LEACE projection computation.

These tests validate the mathematical correctness of the LEACE algorithm:
1. Idempotence property (P^2 ≈ P)
2. Numerical stability with imbalanced classes
3. Batch accumulation equivalence to full-batch computation
4. FP32/FP64 precision consistency
5. CPU/GPU equivalence

Implements: Testing Plan Step 4 (Numerical Correctness)
"""

import pytest
import torch
import numpy as np

from neuro_stylometry.pollution_guard.leace import LEACEComputer


@pytest.mark.unit
class TestLEACEMathematicalProperties:
    """Test mathematical properties of LEACE projection."""
    
    @pytest.mark.determinism
    def test_leace_idempotence_property(self, device):
        """Verify P^2 ≈ P for computed projection matrix."""
        # Generate synthetic embeddings and labels
        torch.manual_seed(42)
        embeddings = torch.randn(100, 768, device=device)
        labels = torch.randint(0, 7, (100,), device=device)
        
        # Compute projection
        computer = LEACEComputer(
            embedding_dim=768, 
            device=str(device),
            force_cpu=(device == "cpu"),
        )
        P = computer.compute_projection(embeddings, labels)
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        # Tolerance depends on device: CUDA has slightly lower precision
        tolerance = 1e-3 if device == "cuda" else 1e-5
        assert rel_error < tolerance, f"Idempotence failed: ||P^2 - P||/||P|| = {rel_error:.2e}"
    
    @pytest.mark.cpu
    def test_leace_condition_number_sparse_demographics(self):
        """LEACE remains numerically stable with imbalanced classes."""
        torch.manual_seed(42)
        
        # 95% female, 5% male - highly imbalanced
        embeddings = torch.randn(200, 768)
        labels = torch.tensor([0]*190 + [1]*10)  # Highly imbalanced
        
        computer = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P = computer.compute_projection(embeddings, labels)
        
        # Verify projection is still valid (idempotent)
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        assert rel_error < 1e-4, (
            f"Idempotence failed on imbalanced data: {rel_error:.2e}"
        )
        
        # Verify no NaN or Inf
        assert torch.isfinite(P).all(), "Projection contains non-finite values"
    
    @pytest.mark.cpu
    def test_leace_batch_accumulation_equals_full_batch(self):
        """Mini-batch accumulation produces same P as full-batch."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(1000, 768)
        # Use concept vectors instead of labels for accumulation
        concepts = torch.randn(1000, 8)  # 8 demographic dimensions
        
        # Full-batch computation
        computer_full = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P_full = computer_full.compute_projection_from_concepts(embeddings, concepts)
        
        # Mini-batch accumulation (batch_size=100)
        computer_mini = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        accumulated_stats = None
        for i in range(0, 1000, 100):
            accumulated_stats = computer_mini.accumulate_batch_concepts(
                embeddings[i:i+100],
                concepts[i:i+100],
                accumulated_stats,
            )
        P_mini = computer_mini.compute_projection_from_concept_stats(accumulated_stats)
        
        # Assert < 0.1% relative error
        rel_error = torch.norm(P_full - P_mini) / torch.norm(P_full)
        
        assert rel_error < 1e-3, (
            f"Batch accumulation error: ||P_full - P_mini||/||P_full|| = {rel_error:.2e}"
        )
    
    @pytest.mark.cpu
    def test_leace_numerical_precision_across_dtypes(self):
        """Projection matrix consistent across FP32/FP64."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(500, 768)
        labels = torch.randint(0, 7, (500,))
        
        # FP32 computation
        computer32 = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P32 = computer32.compute_projection(embeddings.float(), labels)
        
        # FP64 computation - create new computer with FP64 embeddings
        computer64 = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P64 = computer64.compute_projection(embeddings.double(), labels)
        
        # Relative difference should be small
        rel_diff = torch.norm(P32 - P64.float()) / torch.norm(P32)
        
        assert rel_diff < 1e-3, (
            f"FP32/FP64 precision mismatch: rel_diff = {rel_diff:.2e}"
        )
    
    @pytest.mark.cuda
    def test_leace_cpu_gpu_equivalence(self):
        """CPU and GPU projections match within tolerance."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(500, 768)
        labels = torch.randint(0, 7, (500,))
        
        # CPU computation
        computer_cpu = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P_cpu = computer_cpu.compute_projection(embeddings, labels)
        
        # GPU computation
        computer_gpu = LEACEComputer(embedding_dim=768, device="cuda", force_cpu=False)
        P_gpu = computer_gpu.compute_projection(embeddings.cuda(), labels.cuda()).cpu()
        
        # Allow higher tolerance for GPU float32 vs CPU float64
        assert torch.allclose(P_cpu, P_gpu, atol=1e-3, rtol=1e-2), (
            f"CPU-GPU mismatch: max_diff = {torch.max(torch.abs(P_cpu - P_gpu)):.2e}"
        )


@pytest.mark.unit
class TestLEACEEdgeCases:
    """Test edge cases and boundary conditions."""
    
    @pytest.mark.cpu
    def test_leace_single_class(self):
        """LEACE handles single-class data (edge case)."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(100, 768)
        labels = torch.zeros(100, dtype=torch.long)  # All same class
        
        computer = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        
        # Should handle gracefully - projection should be identity-like
        P = computer.compute_projection(embeddings, labels)
        
        assert torch.isfinite(P).all(), "Single-class case produced non-finite values"
    
    @pytest.mark.cpu
    def test_leace_two_samples_per_class(self):
        """LEACE handles minimal samples per class."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(6, 768)  # 3 classes, 2 samples each
        labels = torch.tensor([0, 0, 1, 1, 2, 2])
        
        computer = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P = computer.compute_projection(embeddings, labels)
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        assert rel_error < 1e-4, f"Minimal samples idempotence failed: {rel_error:.2e}"
    
    @pytest.mark.cpu
    def test_leace_high_dimensional_concepts(self):
        """LEACE handles high-dimensional concept vectors."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(500, 768)
        concepts = torch.randn(500, 50)  # 50 demographic dimensions
        
        computer = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P = computer.compute_projection_from_concepts(embeddings, concepts)
        
        # Check basic properties
        assert P.shape == (768, 768), f"Unexpected shape: {P.shape}"
        assert torch.isfinite(P).all(), "High-dim concepts produced non-finite values"
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        assert rel_error < 1e-4, f"High-dim concepts idempotence failed: {rel_error:.2e}"
    
    @pytest.mark.cpu
    @pytest.mark.parametrize("embedding_dim", [128, 768, 1024])
    def test_leace_various_embedding_dims(self, embedding_dim):
        """LEACE works with various embedding dimensions."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(200, embedding_dim)
        labels = torch.randint(0, 5, (200,))
        
        computer = LEACEComputer(embedding_dim=embedding_dim, device="cpu", force_cpu=True)
        P = computer.compute_projection(embeddings, labels)
        
        assert P.shape == (embedding_dim, embedding_dim)
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        assert rel_error < 1e-4, f"dim={embedding_dim}: idempotence failed {rel_error:.2e}"
    
    @pytest.mark.cpu
    @pytest.mark.parametrize("num_classes", [2, 5, 10, 20])
    def test_leace_various_label_cardinalities(self, num_classes):
        """LEACE works with various label cardinalities."""
        torch.manual_seed(42)
        
        embeddings = torch.randn(500, 768)
        labels = torch.randint(0, num_classes, (500,))
        
        computer = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
        P = computer.compute_projection(embeddings, labels)
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        assert rel_error < 1e-4, (
            f"num_classes={num_classes}: idempotence failed {rel_error:.2e}"
        )


@pytest.mark.unit
class TestLEACERegularization:
    """Test regularization effects on numerical stability."""
    
    @pytest.mark.cpu
    @pytest.mark.parametrize("regularization", [1e-7, 1e-5, 1e-3])
    def test_leace_regularization_effect(self, regularization):
        """Regularization improves numerical stability."""
        torch.manual_seed(42)
        
        # Near-collinear embeddings (ill-conditioned)
        base = torch.randn(1, 768)
        noise = torch.randn(100, 768) * 0.01
        embeddings = base + noise  # Nearly identical embeddings
        labels = torch.randint(0, 5, (100,))
        
        computer = LEACEComputer(
            embedding_dim=768, 
            device="cpu", 
            force_cpu=True,
            regularization=regularization,
        )
        P = computer.compute_projection(embeddings, labels)
        
        # Should produce valid projection
        assert torch.isfinite(P).all(), (
            f"reg={regularization}: Produced non-finite values"
        )
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        # Higher regularization should improve numerical stability
        tolerance = 1e-3 if regularization >= 1e-5 else 1e-2
        assert rel_error < tolerance, (
            f"reg={regularization}: idempotence error {rel_error:.2e}"
        )


@pytest.mark.unit
class TestLEACEInputValidation:
    """Test input validation and error handling."""
    
    @pytest.mark.cpu
    def test_leace_rejects_mismatched_shapes(self):
        """LEACE raises ValueError on shape mismatch."""
        embeddings = torch.randn(100, 768)
        labels = torch.randint(0, 5, (50,))  # Wrong size
        
        computer = LEACEComputer(embedding_dim=768, device="cpu")
        
        with pytest.raises(ValueError, match="same number of samples"):
            computer.compute_projection(embeddings, labels)
    
    @pytest.mark.cpu
    def test_leace_rejects_wrong_embedding_dim(self):
        """LEACE raises ValueError on wrong embedding dimension."""
        embeddings = torch.randn(100, 512)  # Wrong dim
        labels = torch.randint(0, 5, (100,))
        
        computer = LEACEComputer(embedding_dim=768, device="cpu")
        
        with pytest.raises(ValueError, match="Embedding dimension"):
            computer.compute_projection(embeddings, labels)
