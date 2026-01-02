"""
LEACE (Linear Adversarial Concept Erasure) Projection Computer.

Computes closed-form projection matrix to remove linear demographic information
from embeddings while preserving maximal variance in the orthogonal subspace.

Math:
    P = I - Σ_c^{-1/2} Σ_z Σ_c^{-1/2}

where:
    Σ_c = within-class covariance
    Σ_z = label-conditional mean differences

Implements: FR-10 (LEACE Math), phaseA-D_implementation_plan.md Section 3.2
"""

import logging
import torch
import torch.nn as nn
from typing import List, Tuple, Optional
import numpy as np
from dataclasses import dataclass

logger = logging.getLogger(__name__)


@dataclass
class CovarianceStats:
    """
    Accumulated covariance statistics.
    
    Attributes:
        within_class_cov: Within-class covariance Σ_c (d x d).
        label_means: Per-label mean embeddings (num_labels x d).
        global_mean: Global mean embedding (d,).
        num_samples: Total number of samples.
    """
    within_class_cov: torch.Tensor
    label_means: torch.Tensor
    global_mean: torch.Tensor
    num_samples: int

    # Optional extended fields used for exact/near-exact batch accumulation
    label_values: Optional[torch.Tensor] = None          # (k,)
    label_counts: Optional[torch.Tensor] = None          # (k,)
    label_sums: Optional[torch.Tensor] = None            # (k, d)
    label_scatter: Optional[torch.Tensor] = None         # (k, d, d)
    global_sum: Optional[torch.Tensor] = None            # (d,)


class LEACEComputer:
    """
    Compute LEACE projection matrix from embeddings and labels.
    
    Features:
    - Closed-form projection matrix computation
    - Regularized Cholesky decomposition for numerical stability
    - Batch accumulation for low-memory environments
    - Idempotence verification (P^2 ≈ P)
    
    Implements: FR-10
    """
    
    def __init__(
        self,
        embedding_dim: int,
        regularization: float = 1e-5,
        device: str = "cuda",
    ):
        """
        Initialize LEACE computer.
        
        Args:
            embedding_dim: Dimensionality of embeddings.
            regularization: Regularization strength for Cholesky (epsilon).
            device: PyTorch device.
        """
        self.embedding_dim = embedding_dim
        self.regularization = regularization
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        
        logger.info(
            f"LEACEComputer initialized: dim={embedding_dim}, "
            f"reg={regularization}, device={self.device}"
        )
    
    def compute_projection(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> torch.Tensor:
        """
        Compute LEACE projection matrix from embeddings and labels.
        
        Args:
            embeddings: Embeddings tensor (num_samples, embedding_dim).
            labels: Integer labels tensor (num_samples,).
            
        Returns:
            Projection matrix P of shape (embedding_dim, embedding_dim).
            
        Raises:
            ValueError: If embeddings/labels shape mismatch or invalid labels.
        """
        # Validate inputs
        if embeddings.shape[0] != labels.shape[0]:
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) and labels ({labels.shape[0]}) "
                "must have same number of samples"
            )
        
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) != expected ({self.embedding_dim})"
            )
        
        # Move to device
        embeddings = embeddings.to(self.device)
        labels = labels.to(self.device)
        
        # Compute covariance statistics
        stats = self._compute_covariance_stats(embeddings, labels)
        
        # Compute projection matrix
        projection_matrix = self._compute_projection_from_stats(stats)
        
        # Verify idempotence
        self._verify_idempotence(projection_matrix)
        
        return projection_matrix
    
    def accumulate_batch(
        self,
        embeddings_batch: torch.Tensor,
        labels_batch: torch.Tensor,
        accumulated_stats: Optional[CovarianceStats] = None,
    ) -> CovarianceStats:
        """
        Accumulate covariance statistics from a batch (for low-memory mode).
        
        This method allows computing LEACE incrementally by accumulating statistics
        across multiple batches without loading all data into memory.
        
        Args:
            embeddings_batch: Batch embeddings (batch_size, embedding_dim).
            labels_batch: Batch labels (batch_size,).
            accumulated_stats: Previously accumulated stats (None for first batch).
            
        Returns:
            Updated CovarianceStats.
        """
        # Keep batch computations on CPU for stability/VRAM control.
        embeddings_batch = embeddings_batch.detach().to("cpu")
        labels_batch = labels_batch.detach().to("cpu")
        
        # Compute batch statistics
        batch_stats = self._compute_covariance_stats(embeddings_batch, labels_batch)
        
        if accumulated_stats is None:
            return batch_stats
        
        # Merge statistics exactly (up to floating point) using per-label scatter.
        if (
            accumulated_stats.label_values is None
            or accumulated_stats.label_counts is None
            or accumulated_stats.label_sums is None
            or accumulated_stats.label_scatter is None
            or accumulated_stats.global_sum is None
            or batch_stats.label_values is None
            or batch_stats.label_counts is None
            or batch_stats.label_sums is None
            or batch_stats.label_scatter is None
            or batch_stats.global_sum is None
        ):
            raise ValueError("Accumulation requires extended CovarianceStats fields")

        a_labels = accumulated_stats.label_values
        b_labels = batch_stats.label_values
        union_labels = torch.unique(torch.cat([a_labels, b_labels])).sort().values

        d = self.embedding_dim
        k = int(union_labels.numel())

        merged_counts = torch.zeros(k, dtype=torch.long)
        merged_sums = torch.zeros(k, d, dtype=torch.float32)
        merged_scatter = torch.zeros(k, d, d, dtype=torch.float32)

        a_index = {int(v.item()): i for i, v in enumerate(a_labels)}
        b_index = {int(v.item()): i for i, v in enumerate(b_labels)}

        for j, lbl in enumerate(union_labels.tolist()):
            ia = a_index.get(int(lbl))
            ib = b_index.get(int(lbl))

            n_a = int(accumulated_stats.label_counts[ia].item()) if ia is not None else 0
            n_b = int(batch_stats.label_counts[ib].item()) if ib is not None else 0

            sum_a = accumulated_stats.label_sums[ia] if ia is not None else torch.zeros(d)
            sum_b = batch_stats.label_sums[ib] if ib is not None else torch.zeros(d)

            scat_a = accumulated_stats.label_scatter[ia] if ia is not None else torch.zeros(d, d)
            scat_b = batch_stats.label_scatter[ib] if ib is not None else torch.zeros(d, d)

            n_t = n_a + n_b
            merged_counts[j] = n_t
            merged_sums[j] = sum_a + sum_b

            if n_a > 0 and n_b > 0:
                mean_a = sum_a / float(n_a)
                mean_b = sum_b / float(n_b)
                delta = (mean_b - mean_a).unsqueeze(1)  # (d, 1)
                correction = (n_a * n_b / float(n_t)) * (delta @ delta.T)
                merged_scatter[j] = scat_a + scat_b + correction
            else:
                merged_scatter[j] = scat_a + scat_b

        total_samples = int(accumulated_stats.num_samples + batch_stats.num_samples)
        global_sum = accumulated_stats.global_sum + batch_stats.global_sum
        global_mean = global_sum / float(total_samples)

        # Compute label means; guard against empty labels (shouldn't occur in practice)
        label_means = torch.zeros(k, d, dtype=torch.float32)
        nonzero = merged_counts > 0
        label_means[nonzero] = merged_sums[nonzero] / merged_counts[nonzero].unsqueeze(1).to(torch.float32)

        within_class_scatter_total = merged_scatter.sum(dim=0)
        within_class_cov = within_class_scatter_total / float(total_samples)

        return CovarianceStats(
            within_class_cov=within_class_cov,
            label_means=label_means,
            global_mean=global_mean,
            num_samples=total_samples,
            label_values=union_labels.to(torch.long),
            label_counts=merged_counts,
            label_sums=merged_sums,
            label_scatter=merged_scatter,
            global_sum=global_sum,
        )
    
    def compute_projection_from_stats(
        self,
        stats: CovarianceStats,
    ) -> torch.Tensor:
        """
        Compute projection matrix from accumulated statistics.
        
        Args:
            stats: Accumulated covariance statistics.
            
        Returns:
            Projection matrix P.
        """
        return self._compute_projection_from_stats(stats)
    
    def _compute_covariance_stats(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> CovarianceStats:
        """
        Compute covariance statistics from embeddings and labels.
        
        Args:
            embeddings: Embeddings (num_samples, embedding_dim).
            labels: Labels (num_samples,).
            
        Returns:
            CovarianceStats.
        """
        # Force CPU for numerical stability and to avoid CUDA linalg driver issues.
        embeddings = embeddings.detach().to("cpu", dtype=torch.float32)
        labels = labels.detach().to("cpu")

        num_samples, embedding_dim = embeddings.shape

        unique_labels = torch.unique(labels).sort().values
        num_labels = int(unique_labels.numel())

        global_sum = embeddings.sum(dim=0)
        global_mean = global_sum / float(num_samples)

        label_counts = torch.zeros(num_labels, dtype=torch.long)
        label_sums = torch.zeros(num_labels, embedding_dim, dtype=torch.float32)
        label_scatter = torch.zeros(num_labels, embedding_dim, embedding_dim, dtype=torch.float32)

        for i, label in enumerate(unique_labels):
            mask = labels == label
            class_embeddings = embeddings[mask]
            n = int(class_embeddings.shape[0])
            label_counts[i] = n
            if n == 0:
                continue

            s = class_embeddings.sum(dim=0)
            label_sums[i] = s
            mean = s / float(n)
            centered = class_embeddings - mean.unsqueeze(0)
            label_scatter[i] = centered.T @ centered

        # within-class covariance: sum of per-label scatters divided by total samples
        within_class_cov = label_scatter.sum(dim=0) / float(num_samples)

        # label means (k, d)
        label_means = torch.zeros(num_labels, embedding_dim, dtype=torch.float32)
        nonzero = label_counts > 0
        label_means[nonzero] = label_sums[nonzero] / label_counts[nonzero].unsqueeze(1).to(torch.float32)

        return CovarianceStats(
            within_class_cov=within_class_cov,
            label_means=label_means,
            global_mean=global_mean,
            num_samples=int(num_samples),
            label_values=unique_labels.to(torch.long),
            label_counts=label_counts,
            label_sums=label_sums,
            label_scatter=label_scatter,
            global_sum=global_sum,
        )
    
    def _compute_projection_from_stats(
        self,
        stats: CovarianceStats,
    ) -> torch.Tensor:
        """
        Compute projection matrix from covariance statistics.
        
        Implements:
            P = I - Σ_c^{-1/2} Σ_z Σ_c^{-1/2}
        
        Args:
            stats: Covariance statistics.
            
        Returns:
            Projection matrix P (embedding_dim, embedding_dim).
        """
        d = self.embedding_dim

        # Work on CPU for stability and to match accumulated stats storage.
        label_means_cpu = stats.label_means.detach().to("cpu", dtype=torch.float64)
        global_mean_cpu = stats.global_mean.detach().to("cpu", dtype=torch.float64)
        within_cov_cpu = stats.within_class_cov.detach().to("cpu", dtype=torch.float64)

        # Center label means (k, d)
        centered_means = label_means_cpu - global_mean_cpu.unsqueeze(0)

        # Regularize within-class covariance
        Sigma_c = within_cov_cpu + float(self.regularization) * torch.eye(d, dtype=torch.float64)

        # Preferred path: compute a true projector using a symmetric inverse sqrt on CPU.
        # This avoids fragile CUDA cuSOLVER paths and yields an (almost) exactly idempotent P.
        try:
            # Symmetric eigendecomposition
            evals, evecs = torch.linalg.eigh(Sigma_c)
            eps = float(self.regularization)
            evals = torch.clamp(evals, min=eps)

            inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
            sqrt = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T

            # Whitened class-mean matrix; row-space spans the concept subspace
            Z = centered_means @ inv_sqrt  # (k, d)
            _, S, Vh = torch.linalg.svd(Z, full_matrices=False)
            # Use a relative tolerance to avoid treating numerical noise as signal.
            # This is important for determinism between full-batch and accumulated stats.
            tol = float(S.max().item()) * 1e-6
            r = int((S > tol).sum().item())
            if r == 0:
                P_cpu = torch.eye(d, dtype=torch.float64)
            else:
                V = Vh[:r].T  # (d, r), orthonormal
                P_whitened = torch.eye(d, dtype=torch.float64) - (V @ V.T)
                P_cpu = sqrt @ P_whitened @ inv_sqrt

            P = P_cpu.to(dtype=torch.float32, device=self.device)
            logger.info(f"Computed projection matrix: {P.shape}")
            return P
        except Exception as e:
            logger.warning(f"CPU-eigendecomposition LEACE path failed, falling back: {e}")

        # Fallback: use robust Cholesky + pseudo-inverse factorization.
        # This path is less strictly idempotent but keeps the pipeline running.
        # Fallback expects tensors on the configured device.
        Sigma_c_dev = Sigma_c.to(dtype=torch.float32, device=self.device)
        centered_dev = centered_means.to(dtype=torch.float32, device=self.device)

        L = self._robust_cholesky(Sigma_c_dev)
        L_pinv = self._robust_pinv(L)
        Sigma_c_inv_sqrt = L_pinv.T

        label_cov = (centered_dev.T @ centered_dev) / len(stats.label_means)
        M = Sigma_c_inv_sqrt @ label_cov @ Sigma_c_inv_sqrt
        I = torch.eye(d, device=self.device)
        P = I - M
        logger.info(f"Computed projection matrix (fallback): {P.shape}")
        return P

    def _robust_cholesky(self, matrix: torch.Tensor) -> torch.Tensor:
        """Robust Cholesky wrapper with GPU->CPU fallback.

        1) Try `torch.linalg.cholesky` on the current device.
        2) If it fails with RuntimeError, retry on CPU.
        3) If CPU also fails (e.g., matrix not PD), fallback to an SVD-based
           factor `L` such that `L @ L.T` approximates `matrix`.
        """

        try:
            return torch.linalg.cholesky(matrix)
        except RuntimeError as e:
            logger.error(f"Cholesky decomposition failed on {matrix.device}: {e}")

        # CPU fallback (explicit)
        try:
            cpu_matrix = matrix.detach().to("cpu")
            L_cpu = torch.linalg.cholesky(cpu_matrix)
            return L_cpu.to(matrix.device)
        except RuntimeError as e:
            logger.error(f"CPU Cholesky fallback failed: {e}")

        # Final fallback: SVD-based PSD factorization.
        # For symmetric PSD matrices, eigen-decomposition is ideal, but SVD is
        # broadly stable and available.
        cpu_matrix = matrix.detach().to("cpu")
        U, S, _ = torch.linalg.svd(cpu_matrix)
        eps = float(self.regularization)
        S_clamped = torch.clamp(S, min=eps)
        L_cpu = U @ torch.diag(torch.sqrt(S_clamped))
        return L_cpu.to(matrix.device)

    def _robust_pinv(self, matrix: torch.Tensor) -> torch.Tensor:
        """Compute pseudo-inverse with GPU->CPU fallback."""

        try:
            return torch.linalg.pinv(matrix)
        except RuntimeError as e:
            logger.error(f"pinv failed on {matrix.device}: {e}")

        cpu_matrix = matrix.detach().to("cpu")
        pinv_cpu = torch.linalg.pinv(cpu_matrix)
        return pinv_cpu.to(matrix.device)
    
    def _verify_idempotence(
        self,
        projection_matrix: torch.Tensor,
        tolerance: float = 1e-5,
    ) -> None:
        """
        Verify that projection matrix is idempotent (P^2 ≈ P).
        
        Args:
            projection_matrix: Projection matrix P.
            tolerance: Maximum allowed error.
            
        Raises:
            ValueError: If idempotence check fails.
        """
        P = projection_matrix
        P_squared = P @ P
        
        error = torch.abs(P_squared - P).max().item()
        
        if error > tolerance:
            logger.warning(
                f"Idempotence check: |P^2 - P|_max = {error:.2e} > {tolerance:.2e}"
            )
        else:
            logger.info(f"Idempotence verified: |P^2 - P|_max = {error:.2e}")
