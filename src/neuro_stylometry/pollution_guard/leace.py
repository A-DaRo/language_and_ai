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


@dataclass
class ConceptCovarianceStats:
        """Accumulated covariance statistics for multi-dimensional concept vectors Z.

        This is the general LEACE case (Theorem 4.1/4.2): Z may be multi-label,
        continuous, or a concatenation of multiple demographic attributes.

        We accumulate raw moments to compute:
            Σ_XX = E[XX^T] - μ_X μ_X^T
            Σ_XZ = E[XZ^T] - μ_X μ_Z^T
        """

        num_samples: int
        sum_x: torch.Tensor  # (d,)
        sum_z: torch.Tensor  # (k,)
        sum_xx: torch.Tensor  # (d, d)
        sum_xz: torch.Tensor  # (d, k)


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
        force_cpu: bool = False,
        compute_dtype: str = "float64",
    ):
        """
        Initialize LEACE computer.
        
        Args:
            embedding_dim: Dimensionality of embeddings.
            regularization: Regularization strength for Cholesky (epsilon).
            device: PyTorch device for accumulation and computation.
            force_cpu: If True, always use CPU for accumulation (for VRAM-limited systems).
            compute_dtype: Data type for LEACE math ("float64" recommended for stability,
                           "float32" for speed on GPU with reduced precision).
        """
        self.embedding_dim = embedding_dim
        self.regularization = regularization
        self.device = torch.device(device if torch.cuda.is_available() else "cpu")
        self.force_cpu = force_cpu
        self.compute_dtype = getattr(torch, compute_dtype, torch.float64)
        
        # Resolve accumulation device based on force_cpu flag
        self._accum_device = torch.device("cpu") if self.force_cpu else self.device
        
        logger.info(
            f"LEACEComputer initialized: dim={embedding_dim}, "
            f"reg={regularization}, device={self.device}, force_cpu={self.force_cpu}, "
            f"compute_dtype={compute_dtype}, accum_device={self._accum_device}"
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
        compute_device = torch.device("cpu") if self.force_cpu else self.device
        embeddings = embeddings.to(compute_device, dtype=torch.float32)
        labels = labels.to(compute_device)
        
        # Compute covariance statistics
        stats = self._compute_covariance_stats(embeddings, labels, device=compute_device)
        
        # Compute projection matrix
        projection_matrix = self._compute_projection_from_stats(stats, device=compute_device)
        
        # Verify idempotence
        self._verify_idempotence(projection_matrix)
        
        return projection_matrix

    def compute_projection_from_concepts(
        self,
        embeddings: torch.Tensor,
        concepts: torch.Tensor,
    ) -> torch.Tensor:
        """Compute LEACE projection from embeddings and a concept design matrix Z.

        Args:
            embeddings: (n, d) float tensor.
            concepts: (n, k) float tensor.

        Returns:
            Projection matrix P of shape (d, d).
        """
        if embeddings.shape[0] != concepts.shape[0]:
            raise ValueError(
                f"Embeddings ({embeddings.shape[0]}) and concepts ({concepts.shape[0]}) "
                "must have same number of samples"
            )
        if embeddings.shape[1] != self.embedding_dim:
            raise ValueError(
                f"Embedding dimension ({embeddings.shape[1]}) != expected ({self.embedding_dim})"
            )

        compute_device = torch.device("cpu") if self.force_cpu else self.device
        X = embeddings.to(compute_device, dtype=torch.float32)
        Z = concepts.to(compute_device, dtype=torch.float32)

        stats = self._compute_concept_stats(X, Z, device=compute_device)
        P = self._compute_projection_from_concept_stats(stats, device=compute_device)
        self._verify_idempotence(P)
        return P

    def accumulate_batch_concepts(
        self,
        embeddings_batch: torch.Tensor,
        concepts_batch: torch.Tensor,
        accumulated_stats: Optional[ConceptCovarianceStats] = None,
    ) -> ConceptCovarianceStats:
        """Accumulate concept covariance stats from a batch.

        Accumulation device is determined by force_cpu flag:
        - force_cpu=True: CPU for streaming stability/VRAM control.
        - force_cpu=False: Use self.device (GPU for HPC throughput).
        
        Note:
            If force_cpu is True and embeddings are on CUDA, a warning is logged
            and embeddings are automatically moved to CPU.
        """
        # Check for GPU embeddings when force_cpu is enabled
        if self.force_cpu and embeddings_batch.device.type == "cuda":
            logger.warning(
                "force_cpu=True but embeddings are on CUDA. "
                "Auto-moving to CPU for LEACE accumulation."
            )

        # Use configured accumulation device (respects force_cpu)
        X = embeddings_batch.detach().to(self._accum_device, dtype=torch.float32)
        Z = concepts_batch.detach().to(self._accum_device, dtype=torch.float32)

        batch_stats = self._compute_concept_stats(X, Z, device=self._accum_device)
        if accumulated_stats is None:
            return batch_stats

        if accumulated_stats.sum_z.shape != batch_stats.sum_z.shape:
            raise ValueError(
                "Concept dimension mismatch during accumulation: "
                f"{accumulated_stats.sum_z.shape} vs {batch_stats.sum_z.shape}"
            )

        return ConceptCovarianceStats(
            num_samples=int(accumulated_stats.num_samples + batch_stats.num_samples),
            sum_x=accumulated_stats.sum_x + batch_stats.sum_x,
            sum_z=accumulated_stats.sum_z + batch_stats.sum_z,
            sum_xx=accumulated_stats.sum_xx + batch_stats.sum_xx,
            sum_xz=accumulated_stats.sum_xz + batch_stats.sum_xz,
        )

    def compute_projection_from_concept_stats(
        self,
        stats: ConceptCovarianceStats,
    ) -> torch.Tensor:
        return self._compute_projection_from_concept_stats(stats)
    
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
        
        Accumulation device is determined by force_cpu flag:
        - force_cpu=True: CPU for streaming stability/VRAM control.
        - force_cpu=False: Use self.device (GPU for HPC throughput).
        
        Args:
            embeddings_batch: Batch embeddings (batch_size, embedding_dim).
            labels_batch: Batch labels (batch_size,).
            accumulated_stats: Previously accumulated stats (None for first batch).
            
        Returns:
            Updated CovarianceStats.
        """
        # Use configured accumulation device (respects force_cpu)
        embeddings_batch = embeddings_batch.detach().to(self._accum_device)
        labels_batch = labels_batch.detach().to(self._accum_device)
        
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
        device: Optional[torch.device] = None,
    ) -> CovarianceStats:
        """
        Compute covariance statistics from embeddings and labels.
        
        Args:
            embeddings: Embeddings (num_samples, embedding_dim).
            labels: Labels (num_samples,).
            
        Returns:
            CovarianceStats.
        """
        compute_device = device or embeddings.device
        embeddings = embeddings.detach().to(compute_device, dtype=torch.float32)
        labels = labels.detach().to(compute_device)

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

    def _compute_concept_stats(
        self,
        embeddings: torch.Tensor,
        concepts: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> ConceptCovarianceStats:
        compute_device = device or embeddings.device
        X = embeddings.detach().to(compute_device, dtype=torch.float32)
        Z = concepts.detach().to(compute_device, dtype=torch.float32)

        if X.ndim != 2 or Z.ndim != 2:
            raise ValueError("Embeddings and concepts must be 2D tensors")
        if X.shape[0] != Z.shape[0]:
            raise ValueError("Embeddings and concepts must share the same first dimension")

        n = int(X.shape[0])
        d = int(X.shape[1])
        if d != self.embedding_dim:
            raise ValueError(f"Embedding dimension ({d}) != expected ({self.embedding_dim})")

        sum_x = X.sum(dim=0)
        sum_z = Z.sum(dim=0)
        sum_xx = X.T @ X
        sum_xz = X.T @ Z
        return ConceptCovarianceStats(
            num_samples=n,
            sum_x=sum_x,
            sum_z=sum_z,
            sum_xx=sum_xx,
            sum_xz=sum_xz,
        )
    
    def _compute_projection_from_stats(
        self,
        stats: CovarianceStats,
        device: Optional[torch.device] = None,
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

        compute_device = device or stats.within_class_cov.device
        output_device = torch.device("cpu") if self.force_cpu else self.device
        compute_dtype = torch.float64 if compute_device.type == "cpu" else torch.float32

        label_means = stats.label_means.detach().to(compute_device, dtype=compute_dtype)
        global_mean = stats.global_mean.detach().to(compute_device, dtype=compute_dtype)
        within_cov = stats.within_class_cov.detach().to(compute_device, dtype=compute_dtype)

        # Center label means (k, d)
        centered_means = label_means - global_mean.unsqueeze(0)

        # Regularize within-class covariance
        Sigma_c = within_cov + float(self.regularization) * torch.eye(
            d, dtype=compute_dtype, device=compute_device
        )

        self._log_condition_numbers(Sigma_c, centered_means, device=compute_device)

        # Preferred path: compute a true projector using a symmetric inverse sqrt.
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
                P_comp = torch.eye(d, dtype=compute_dtype, device=compute_device)
            else:
                V = Vh[:r].T  # (d, r), orthonormal
                P_whitened = torch.eye(d, dtype=compute_dtype, device=compute_device) - (V @ V.T)
                P_comp = sqrt @ P_whitened @ inv_sqrt

            P = P_comp.to(dtype=torch.float32, device=output_device)
            logger.info(f"Computed projection matrix: {P.shape}")
            return P
        except Exception as e:
            logger.warning(f"Eigendecomposition LEACE path failed, falling back: {e}")

        # Fallback: use robust Cholesky + pseudo-inverse factorization.
        # This path is less strictly idempotent but keeps the pipeline running.
        # Fallback expects tensors on the configured device.
        Sigma_c_dev = Sigma_c.to(dtype=torch.float32, device=output_device)
        centered_dev = centered_means.to(dtype=torch.float32, device=output_device)

        L = self._robust_cholesky(Sigma_c_dev)
        L_pinv = self._robust_pinv(L)
        Sigma_c_inv_sqrt = L_pinv.T

        label_cov = (centered_dev.T @ centered_dev) / len(stats.label_means)
        M = Sigma_c_inv_sqrt @ label_cov @ Sigma_c_inv_sqrt
        I = torch.eye(d, device=output_device)
        P = I - M
        logger.info(f"Computed projection matrix (fallback): {P.shape}")
        return P

    def _compute_projection_from_concept_stats(
        self,
        stats: ConceptCovarianceStats,
        device: Optional[torch.device] = None,
    ) -> torch.Tensor:
        """Compute LEACE projection using cross-covariance Σ_XZ.

        Implements the general LEACE solution for multi-dimensional Z.
        """

        d = self.embedding_dim
        compute_device = device or stats.sum_x.device
        output_device = torch.device("cpu") if self.force_cpu else self.device
        compute_dtype = torch.float64 if compute_device.type == "cpu" else torch.float32

        n = float(max(int(stats.num_samples), 1))

        sum_x = stats.sum_x.detach().to(compute_device, dtype=compute_dtype)
        sum_z = stats.sum_z.detach().to(compute_device, dtype=compute_dtype)
        sum_xx = stats.sum_xx.detach().to(compute_device, dtype=compute_dtype)
        sum_xz = stats.sum_xz.detach().to(compute_device, dtype=compute_dtype)

        mu_x = sum_x / n
        mu_z = sum_z / n

        Sigma_xx = (sum_xx / n) - (mu_x.unsqueeze(1) @ mu_x.unsqueeze(0))
        Sigma_xz = (sum_xz / n) - (mu_x.unsqueeze(1) @ mu_z.unsqueeze(0))

        Sigma_xx = Sigma_xx + float(self.regularization) * torch.eye(
            d, dtype=compute_dtype, device=compute_device
        )

        try:
            evals, evecs = torch.linalg.eigh(Sigma_xx)
            eps = float(self.regularization)
            evals = torch.clamp(evals, min=eps)

            inv_sqrt = evecs @ torch.diag(1.0 / torch.sqrt(evals)) @ evecs.T
            sqrt = evecs @ torch.diag(torch.sqrt(evals)) @ evecs.T

            M = inv_sqrt @ Sigma_xz  # (d, k)
            U, S, _ = torch.linalg.svd(M, full_matrices=False)
            if S.numel() == 0:
                P_comp = torch.eye(d, dtype=compute_dtype, device=compute_device)
            else:
                tol = float(S.max().item()) * 1e-6
                r = int((S > tol).sum().item())
                if r == 0:
                    P_comp = torch.eye(d, dtype=compute_dtype, device=compute_device)
                else:
                    U_r = U[:, :r]
                    P_white = torch.eye(d, dtype=compute_dtype, device=compute_device) - (U_r @ U_r.T)
                    P_comp = sqrt @ P_white @ inv_sqrt

            P = P_comp.to(dtype=torch.float32, device=output_device)
            logger.info(f"Computed projection matrix (concepts): {P.shape}")
            return P
        except Exception as e:
            logger.warning(f"Concept LEACE eigendecomposition path failed, falling back: {e}")

        # Fallback: GPU robust Cholesky + pseudo-inverse whitening.
        Sigma_xx_dev = Sigma_xx.to(dtype=torch.float32, device=output_device)
        Sigma_xz_dev = Sigma_xz.to(dtype=torch.float32, device=output_device)

        L = self._robust_cholesky(Sigma_xx_dev)
        L_pinv = self._robust_pinv(L)
        inv_sqrt = L_pinv.T

        M = inv_sqrt @ Sigma_xz_dev
        U, S, _ = torch.linalg.svd(M, full_matrices=False)
        if S.numel() == 0:
            return torch.eye(d, dtype=torch.float32, device=output_device)
        tol = float(S.max().item()) * 1e-6
        r = int((S > tol).sum().item())
        if r == 0:
            return torch.eye(d, dtype=torch.float32, device=output_device)
        U_r = U[:, :r]
        P_white = torch.eye(d, dtype=torch.float32, device=output_device) - (U_r @ U_r.T)
        # Approximate unwhitening using L as sqrt factor.
        P = L @ P_white @ inv_sqrt
        logger.info(f"Computed projection matrix (concepts fallback): {P.shape}")
        return P

    def _log_condition_numbers(
        self,
        Sigma_c: torch.Tensor,
        centered_means: torch.Tensor,
        device: Optional[torch.device] = None,
    ) -> None:
        """Log condition numbers for diagnostic stability checks."""
        try:
            cond_c = torch.linalg.cond(Sigma_c).item()
            logger.info(f"LEACE cond(Sigma_c): {cond_c:.2e}")
        except Exception as e:
            logger.warning(f"Failed to compute cond(Sigma_c): {e}")

        try:
            label_cov = (centered_means.T @ centered_means) / max(1, centered_means.shape[0])
            cond_z = torch.linalg.cond(label_cov).item()
            logger.info(f"LEACE cond(label_cov): {cond_z:.2e}")
        except Exception as e:
            logger.warning(f"Failed to compute cond(label_cov): {e}")

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
        
        diff = P_squared - P
        numerator = torch.linalg.norm(diff, ord="fro").item()
        denominator = torch.linalg.norm(P, ord="fro").item()
        rel_error = numerator / max(denominator, 1e-12)

        if rel_error > tolerance:
            logger.warning(
                f"Idempotence check: ||P^2 - P||_F / ||P||_F = {rel_error:.2e} "
                f"> {tolerance:.2e}"
            )
        else:
            logger.info(
                f"Idempotence verified: ||P^2 - P||_F / ||P||_F = {rel_error:.2e}"
            )
