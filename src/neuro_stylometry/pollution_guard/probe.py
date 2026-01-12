"""
Linear Probe for LEACE Evaluation.

Implements logistic regression probe to measure demographic information leakage
before and after LEACE projection.

Metric: Amnesic Drop = (Acc_before - Acc_after) / Acc_before

Target: Amnesic Drop > 30%

Features:
- Multi-backend support: sklearn (CPU), cuml (GPU), torch (GPU)
- Automatic backend detection based on hardware
- K-fold cross-validation with statistical significance testing
- Class imbalance handling via weighted sampling / pos_weight
- Comprehensive metrics: balanced accuracy, F1, per-class performance

Implements: phaseA-D_implementation_plan.md Section 9.2
"""

from __future__ import annotations

import logging
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
from typing import Dict, List, Optional, Tuple, Any

import numpy as np
import torch
from sklearn.metrics import (
    accuracy_score,
    balanced_accuracy_score,
    f1_score,
    confusion_matrix,
)

logger = logging.getLogger(__name__)


class ProbeBackend(Enum):
    """Supported probe backends."""
    SKLEARN = "sklearn"
    CUML = "cuml"
    TORCH = "torch"


@dataclass
class ProbeConfig:
    """Configuration for probe training and evaluation."""
    backend: str = "auto"  # "auto" | "sklearn" | "cuml" | "torch"
    max_iter: int = 1000
    random_state: int = 42
    # K-fold cross-validation settings
    n_folds: int = 5
    # Statistical significance
    pvalue_threshold: float = 0.05
    # Torch backend settings
    torch_lr: float = 0.01
    torch_epochs: int = 100
    torch_batch_size: int = 256
    torch_weight_decay: float = 1e-4
    # Class imbalance handling
    use_class_weights: bool = True
    # Solver benchmark (compare exact vs approximate)
    benchmark_solvers: bool = False


@dataclass
class ProbeResult:
    """Result from probe training/evaluation."""
    accuracy: float
    balanced_accuracy: float
    f1_macro: float
    f1_weighted: float
    confusion_matrix: Optional[np.ndarray] = None
    class_accuracies: Optional[Dict[int, float]] = None
    predictions: Optional[np.ndarray] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "accuracy": float(self.accuracy),
            "balanced_accuracy": float(self.balanced_accuracy),
            "f1_macro": float(self.f1_macro),
            "f1_weighted": float(self.f1_weighted),
            "class_accuracies": self.class_accuracies,
        }


@dataclass
class CrossValidationResult:
    """Result from k-fold cross-validation."""
    mean_accuracy: float
    std_accuracy: float
    mean_balanced_accuracy: float
    std_balanced_accuracy: float
    ci_95_lower: float
    ci_95_upper: float
    fold_accuracies: List[float] = field(default_factory=list)
    fold_balanced_accuracies: List[float] = field(default_factory=list)
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        return {
            "mean_accuracy": float(self.mean_accuracy),
            "std_accuracy": float(self.std_accuracy),
            "mean_balanced_accuracy": float(self.mean_balanced_accuracy),
            "std_balanced_accuracy": float(self.std_balanced_accuracy),
            "ci_95_lower": float(self.ci_95_lower),
            "ci_95_upper": float(self.ci_95_upper),
            "fold_accuracies": [float(a) for a in self.fold_accuracies],
        }


@dataclass
class AmnesicDropResult:
    """Comprehensive result from amnesic drop computation."""
    acc_before: float
    acc_after: float
    amnesic_drop: float
    # Balanced accuracy metrics (robust to class imbalance)
    balanced_acc_before: float
    balanced_acc_after: float
    balanced_amnesic_drop: float
    # Statistical significance (from k-fold)
    cv_before: Optional[CrossValidationResult] = None
    cv_after: Optional[CrossValidationResult] = None
    pvalue: Optional[float] = None
    is_significant: bool = False
    # Per-class metrics
    minority_class_f1_before: Optional[float] = None
    minority_class_f1_after: Optional[float] = None
    minority_class_f1_delta: Optional[float] = None
    # Majority baseline comparison
    majority_baseline: Optional[float] = None
    majority_baseline_gap_before: Optional[float] = None
    majority_baseline_gap_after: Optional[float] = None
    # Learning curves (torch backend only)
    learning_curve_before: Optional[List[float]] = None
    learning_curve_after: Optional[List[float]] = None
    # Solver benchmark metrics
    solver_accuracy_delta: Optional[float] = None
    weight_cosine_similarity: Optional[float] = None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert to JSON-serializable dict."""
        result = {
            "acc_before": float(self.acc_before),
            "acc_after": float(self.acc_after),
            "amnesic_drop": float(self.amnesic_drop),
            "balanced_acc_before": float(self.balanced_acc_before),
            "balanced_acc_after": float(self.balanced_acc_after),
            "balanced_amnesic_drop": float(self.balanced_amnesic_drop),
            "is_significant": self.is_significant,
        }
        if self.pvalue is not None:
            result["pvalue"] = float(self.pvalue)
        if self.cv_before is not None:
            result["cv_before"] = self.cv_before.to_dict()
        if self.cv_after is not None:
            result["cv_after"] = self.cv_after.to_dict()
        if self.minority_class_f1_delta is not None:
            result["minority_class_f1_delta"] = float(self.minority_class_f1_delta)
        if self.majority_baseline_gap_before is not None:
            result["majority_baseline_gap_before"] = float(self.majority_baseline_gap_before)
            result["majority_baseline_gap_after"] = float(self.majority_baseline_gap_after)
        if self.learning_curve_before is not None:
            result["learning_curve_before"] = [float(x) for x in self.learning_curve_before]
            result["learning_curve_after"] = [float(x) for x in self.learning_curve_after]
        if self.solver_accuracy_delta is not None:
            result["solver_accuracy_delta"] = float(self.solver_accuracy_delta)
            result["weight_cosine_similarity"] = float(self.weight_cosine_similarity)
        return result


def _detect_backend() -> ProbeBackend:
    """Auto-detect best available backend."""
    # Try cuML first (GPU-accelerated)
    if torch.cuda.is_available():
        try:
            import cuml  # noqa: F401
            logger.info("Detected cuML backend (GPU-accelerated)")
            return ProbeBackend.CUML
        except ImportError:
            pass
        # Fall back to torch on GPU
        logger.info("Using torch backend (GPU available, cuML not found)")
        return ProbeBackend.TORCH
    
    # CPU fallback
    logger.info("Using sklearn backend (CPU)")
    return ProbeBackend.SKLEARN


class BaseProbe(ABC):
    """Abstract base class for linear probes."""
    
    @abstractmethod
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:
        """Fit probe to training data."""
        pass
    
    @abstractmethod
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels for data."""
        pass
    
    @abstractmethod
    def get_weights(self) -> np.ndarray:
        """Get learned weight coefficients."""
        pass
    
    def evaluate(self, X: np.ndarray, y: np.ndarray) -> ProbeResult:
        """Evaluate probe on test data."""
        y_pred = self.predict(X)
        
        # Compute metrics
        accuracy = accuracy_score(y, y_pred)
        balanced_acc = balanced_accuracy_score(y, y_pred)
        f1_mac = f1_score(y, y_pred, average="macro", zero_division=0)
        f1_wgt = f1_score(y, y_pred, average="weighted", zero_division=0)
        cm = confusion_matrix(y, y_pred)
        
        # Per-class accuracy
        unique_labels = np.unique(y)
        class_accs = {}
        for label in unique_labels:
            mask = y == label
            if mask.sum() > 0:
                class_accs[int(label)] = float((y_pred[mask] == label).mean())
        
        return ProbeResult(
            accuracy=accuracy,
            balanced_accuracy=balanced_acc,
            f1_macro=f1_mac,
            f1_weighted=f1_wgt,
            confusion_matrix=cm,
            class_accuracies=class_accs,
            predictions=y_pred,
        )


class SklearnProbe(BaseProbe):
    """sklearn-based logistic regression probe (CPU)."""
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.classifier = None
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:
        """Fit sklearn logistic regression."""
        from sklearn.linear_model import LogisticRegression
        
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning("Less than 2 unique labels, creating dummy classifier")
            self.classifier = LogisticRegression(
                max_iter=self.config.max_iter,
                random_state=self.config.random_state,
            )
            if len(unique_labels) == 1:
                X_dummy = np.vstack([X[:1], X[:1]])
                y_dummy = np.array([unique_labels[0], unique_labels[0]])
                self.classifier.fit(X_dummy, y_dummy)
            return
        
        # Modern sklearn auto-selects solver and multi_class
        class_weight = "balanced" if self.config.use_class_weights else None
        self.classifier = LogisticRegression(
            max_iter=self.config.max_iter,
            random_state=self.config.random_state,
            class_weight=class_weight,
            n_jobs=-1,  # Use all cores
        )
        self.classifier.fit(X, y, sample_weight=sample_weights)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.classifier is None:
            raise ValueError("Probe not fitted")
        return self.classifier.predict(X)
    
    def get_weights(self) -> np.ndarray:
        """Get weight coefficients."""
        if self.classifier is None:
            raise ValueError("Probe not fitted")
        return self.classifier.coef_


class CuMLProbe(BaseProbe):
    """cuML-based logistic regression probe (GPU-accelerated)."""
    
    def __init__(self, config: ProbeConfig):
        self.config = config
        self.classifier = None
        self._fallback_sklearn = False
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:
        """Fit cuML logistic regression."""
        try:
            from cuml.linear_model import LogisticRegression as CuMLLogisticRegression
            import cupy as cp
        except ImportError as e:
            raise ImportError(
                "cuML not available. Install RAPIDS cuML: "
                "conda install -c rapidsai -c conda-forge -c nvidia cuml"
            ) from e
        
        unique_labels = np.unique(y)
        if len(unique_labels) < 2:
            logger.warning("Less than 2 unique labels, falling back to sklearn")
            sklearn_probe = SklearnProbe(self.config)
            sklearn_probe.fit(X, y, sample_weights)
            self.classifier = sklearn_probe.classifier
            self._fallback_sklearn = True
            return
        
        self._fallback_sklearn = False
        
        # Convert to cupy arrays
        X_gpu = cp.asarray(X, dtype=cp.float32)
        y_gpu = cp.asarray(y, dtype=cp.float32)
        
        # cuML LogisticRegression
        self.classifier = CuMLLogisticRegression(
            max_iter=self.config.max_iter,
            verbose=0,
        )
        
        # Note: cuML LR doesn't support sample_weight directly
        self.classifier.fit(X_gpu, y_gpu)
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.classifier is None:
            raise ValueError("Probe not fitted")
        
        if self._fallback_sklearn:
            return self.classifier.predict(X)
        
        import cupy as cp
        X_gpu = cp.asarray(X, dtype=cp.float32)
        return cp.asnumpy(self.classifier.predict(X_gpu)).astype(int)
    
    def get_weights(self) -> np.ndarray:
        """Get weight coefficients."""
        if self.classifier is None:
            raise ValueError("Probe not fitted")
        
        if self._fallback_sklearn:
            return self.classifier.coef_
        
        import cupy as cp
        return cp.asnumpy(self.classifier.coef_)


class TorchProbe(BaseProbe):
    """PyTorch-based logistic regression probe (GPU with BF16 autocast)."""
    
    def __init__(self, config: ProbeConfig, device: str = "auto"):
        self.config = config
        if device == "auto":
            self.device = "cuda" if torch.cuda.is_available() else "cpu"
        else:
            self.device = device
        self.model = None
        self.classes_ = None
        self.learning_curve: List[float] = []
        self._single_class = 0
        self._label_map = {}
        self._reverse_label_map = {}
    
    def fit(self, X: np.ndarray, y: np.ndarray, sample_weights: Optional[np.ndarray] = None) -> None:
        """Fit PyTorch logistic regression with SGD/Adam."""
        unique_labels = np.unique(y)
        self.classes_ = unique_labels
        n_classes = len(unique_labels)
        
        if n_classes < 2:
            logger.warning("Less than 2 unique labels, creating dummy model")
            self.model = None
            self._single_class = unique_labels[0] if len(unique_labels) == 1 else 0
            return
        
        # Map labels to contiguous indices
        label_map = {label: idx for idx, label in enumerate(unique_labels)}
        y_mapped = np.array([label_map[label] for label in y])
        
        # Convert to tensors
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        y_t = torch.tensor(y_mapped, dtype=torch.long, device=self.device)
        
        n_features = X.shape[1]
        
        # Binary vs multiclass
        if n_classes == 2:
            self.model = torch.nn.Linear(n_features, 1).to(self.device)
            criterion = torch.nn.BCEWithLogitsLoss(reduction="none")
            y_t_float = y_t.float()
        else:
            self.model = torch.nn.Linear(n_features, n_classes).to(self.device)
            criterion = torch.nn.CrossEntropyLoss(reduction="none")
            y_t_float = None
        
        # Compute class weights for imbalance handling
        if self.config.use_class_weights:
            class_counts = np.bincount(y_mapped, minlength=n_classes)
            class_weights = len(y_mapped) / (n_classes * class_counts + 1e-8)
            class_weights_t = torch.tensor(class_weights, dtype=torch.float32, device=self.device)
        else:
            class_weights_t = None
        
        # Sample weights
        if sample_weights is not None:
            sample_weights_t = torch.tensor(sample_weights, dtype=torch.float32, device=self.device)
        else:
            sample_weights_t = torch.ones(len(X_t), device=self.device)
        
        # Optimizer
        optimizer = torch.optim.AdamW(
            self.model.parameters(),
            lr=self.config.torch_lr,
            weight_decay=self.config.torch_weight_decay,
        )
        
        # Learning rate scheduler
        scheduler = torch.optim.lr_scheduler.CosineAnnealingLR(
            optimizer, T_max=self.config.torch_epochs
        )
        
        batch_size = self.config.torch_batch_size
        n_samples = len(X_t)
        self.learning_curve = []
        
        # Use BF16 autocast if available
        use_bf16 = (
            self.device == "cuda" 
            and torch.cuda.is_available() 
            and torch.cuda.is_bf16_supported()
        )
        autocast_dtype = torch.bfloat16 if use_bf16 else torch.float32
        
        self.model.train()
        for epoch in range(self.config.torch_epochs):
            # Shuffle indices
            perm = torch.randperm(n_samples, device=self.device)
            epoch_loss = 0.0
            n_batches = 0
            
            for i in range(0, n_samples, batch_size):
                batch_idx = perm[i:i + batch_size]
                X_batch = X_t[batch_idx]
                w_batch = sample_weights_t[batch_idx]
                
                optimizer.zero_grad()
                
                with torch.autocast(device_type=self.device.split(":")[0], dtype=autocast_dtype):
                    logits = self.model(X_batch)
                    
                    if n_classes == 2:
                        y_batch = y_t_float[batch_idx]
                        loss_per_sample = criterion(logits.squeeze(), y_batch)
                    else:
                        y_batch = y_t[batch_idx]
                        loss_per_sample = criterion(logits, y_batch)
                    
                    # Apply sample weights
                    loss = (loss_per_sample * w_batch).mean()
                    
                    # Apply class weights (for binary)
                    if class_weights_t is not None and n_classes == 2:
                        weight_factor = class_weights_t[y_t[batch_idx]]
                        loss = (loss_per_sample * weight_factor * w_batch).mean()
                
                loss.backward()
                optimizer.step()
                
                epoch_loss += loss.item()
                n_batches += 1
            
            scheduler.step()
            
            # Record learning curve
            avg_loss = epoch_loss / max(n_batches, 1)
            self.learning_curve.append(avg_loss)
            
            if (epoch + 1) % 20 == 0:
                logger.debug(f"Epoch {epoch + 1}/{self.config.torch_epochs}, Loss: {avg_loss:.4f}")
        
        self.model.eval()
        self._label_map = label_map
        self._reverse_label_map = {v: k for k, v in label_map.items()}
    
    def predict(self, X: np.ndarray) -> np.ndarray:
        """Predict labels."""
        if self.model is None:
            # Dummy classifier
            return np.full(len(X), self._single_class)
        
        X_t = torch.tensor(X, dtype=torch.float32, device=self.device)
        
        self.model.eval()
        with torch.no_grad():
            logits = self.model(X_t)
            
            if len(self.classes_) == 2:
                preds_idx = (torch.sigmoid(logits.squeeze()) > 0.5).long()
            else:
                preds_idx = logits.argmax(dim=1)
        
        # Map back to original labels
        preds_np = preds_idx.cpu().numpy()
        return np.array([self._reverse_label_map[int(p)] for p in preds_np])
    
    def get_weights(self) -> np.ndarray:
        """Get weight coefficients."""
        if self.model is None:
            raise ValueError("Probe not fitted (dummy classifier)")
        return self.model.weight.detach().cpu().numpy()
    
    def get_learning_curve(self) -> List[float]:
        """Get training loss curve."""
        return self.learning_curve


def create_probe(config: ProbeConfig, device: str = "auto") -> BaseProbe:
    """Factory function to create probe based on config."""
    backend_str = config.backend.lower()
    
    if backend_str == "auto":
        backend = _detect_backend()
    else:
        backend = ProbeBackend(backend_str)
    
    if backend == ProbeBackend.SKLEARN:
        return SklearnProbe(config)
    elif backend == ProbeBackend.CUML:
        return CuMLProbe(config)
    elif backend == ProbeBackend.TORCH:
        return TorchProbe(config, device=device)
    else:
        raise ValueError(f"Unknown backend: {backend}")


# =============================================================================
# Legacy LinearProbe class for backward compatibility
# =============================================================================

class LinearProbe:
    """
    Linear probe for evaluating demographic information leakage.
    
    Uses logistic regression to measure how much demographic information
    is linearly accessible in embeddings.
    
    This class maintains backward compatibility while using the new backend system.
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
        backend: str = "auto",
    ):
        """
        Initialize linear probe.
        
        Args:
            max_iter: Maximum iterations for logistic regression.
            random_state: Random seed for reproducibility.
            backend: Probe backend ("auto", "sklearn", "cuml", "torch").
        """
        self.config = ProbeConfig(
            backend=backend,
            max_iter=max_iter,
            random_state=random_state,
        )
        self._probe: Optional[BaseProbe] = None
    
    def train(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Train probe on embeddings and labels.
        
        Args:
            embeddings: Embeddings tensor (num_samples, embedding_dim).
            labels: Integer labels tensor (num_samples,).
            
        Returns:
            Training accuracy.
        """
        # Convert to numpy
        X = embeddings.cpu().numpy()
        y = labels.cpu().numpy()
        
        # Filter out null labels (-1)
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        if len(X) == 0:
            logger.warning("No valid samples after filtering null labels")
            return 0.0
        
        self._probe = create_probe(self.config)
        self._probe.fit(X, y)
        
        # Compute training accuracy
        result = self._probe.evaluate(X, y)
        logger.info(f"Probe training accuracy: {result.accuracy:.2%}")
        return result.accuracy
    
    def evaluate(
        self,
        embeddings: torch.Tensor,
        labels: torch.Tensor,
    ) -> float:
        """
        Evaluate probe on embeddings and labels.
        
        Args:
            embeddings: Embeddings tensor (num_samples, embedding_dim).
            labels: Integer labels tensor (num_samples,).
            
        Returns:
            Evaluation accuracy.
        """
        if self._probe is None:
            raise ValueError("Probe not trained, call train() first")
        
        # Convert to numpy
        X = embeddings.cpu().numpy()
        y = labels.cpu().numpy()
        
        # Filter out null labels
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        result = self._probe.evaluate(X, y)
        logger.info(f"Probe evaluation accuracy: {result.accuracy:.2%}")
        return result.accuracy


# =============================================================================
# K-Fold Cross-Validation
# =============================================================================

def _compute_cv_result(
    X: np.ndarray,
    y: np.ndarray,
    config: ProbeConfig,
    device: str = "auto",
) -> CrossValidationResult:
    """Perform stratified k-fold cross-validation."""
    from sklearn.model_selection import StratifiedKFold
    
    skf = StratifiedKFold(
        n_splits=config.n_folds,
        shuffle=True,
        random_state=config.random_state,
    )
    
    fold_accs = []
    fold_balanced_accs = []
    
    for fold_idx, (train_idx, test_idx) in enumerate(skf.split(X, y)):
        X_train, X_test = X[train_idx], X[test_idx]
        y_train, y_test = y[train_idx], y[test_idx]
        
        probe = create_probe(config, device=device)
        probe.fit(X_train, y_train)
        result = probe.evaluate(X_test, y_test)
        
        fold_accs.append(result.accuracy)
        fold_balanced_accs.append(result.balanced_accuracy)
    
    mean_acc = np.mean(fold_accs)
    std_acc = np.std(fold_accs, ddof=1)
    mean_balanced = np.mean(fold_balanced_accs)
    std_balanced = np.std(fold_balanced_accs, ddof=1)
    
    # 95% CI using t-distribution
    from scipy import stats
    n = len(fold_accs)
    t_crit = stats.t.ppf(0.975, df=n - 1)
    se = std_acc / np.sqrt(n)
    ci_lower = mean_acc - t_crit * se
    ci_upper = mean_acc + t_crit * se
    
    return CrossValidationResult(
        mean_accuracy=mean_acc,
        std_accuracy=std_acc,
        mean_balanced_accuracy=mean_balanced,
        std_balanced_accuracy=std_balanced,
        ci_95_lower=ci_lower,
        ci_95_upper=ci_upper,
        fold_accuracies=fold_accs,
        fold_balanced_accuracies=fold_balanced_accs,
    )


def _compute_pvalue(
    fold_accs_before: List[float],
    fold_accs_after: List[float],
) -> float:
    """Compute p-value using paired t-test."""
    from scipy import stats
    
    if len(fold_accs_before) != len(fold_accs_after):
        raise ValueError("Fold counts must match for paired t-test")
    
    # Paired t-test (one-sided: before > after)
    t_stat, p_value = stats.ttest_rel(fold_accs_before, fold_accs_after)
    
    # One-sided p-value (we expect before > after)
    return p_value / 2 if t_stat > 0 else 1 - p_value / 2


# =============================================================================
# Main compute_amnesic_drop function
# =============================================================================

def compute_amnesic_drop(
    embeddings_before: torch.Tensor,
    embeddings_after: torch.Tensor,
    labels: torch.Tensor,
    train_split: float = 0.8,
    random_state: int = 42,
    config: Optional[ProbeConfig] = None,
    use_kfold: bool = False,
    device: str = "auto",
) -> Tuple[float, float, float]:
    """
    Compute amnesic drop metric (legacy signature for backward compatibility).
    
    Trains probes on embeddings before and after LEACE projection
    and measures the drop in accuracy.
    
    Args:
        embeddings_before: Embeddings before projection (num_samples, dim).
        embeddings_after: Embeddings after projection (num_samples, dim).
        labels: Demographic labels (num_samples,).
        train_split: Fraction of data for training probe (ignored if use_kfold=True).
        random_state: Random seed.
        config: Optional ProbeConfig for advanced settings.
        use_kfold: Whether to use k-fold cross-validation.
        device: Device for torch backend.
        
    Returns:
        Tuple of (acc_before, acc_after, amnesic_drop).
    """
    result = compute_amnesic_drop_extended(
        embeddings_before=embeddings_before,
        embeddings_after=embeddings_after,
        labels=labels,
        train_split=train_split,
        random_state=random_state,
        config=config,
        use_kfold=use_kfold,
        device=device,
    )
    return result.acc_before, result.acc_after, result.amnesic_drop


def compute_amnesic_drop_extended(
    embeddings_before: torch.Tensor,
    embeddings_after: torch.Tensor,
    labels: torch.Tensor,
    train_split: float = 0.8,
    random_state: int = 42,
    config: Optional[ProbeConfig] = None,
    use_kfold: bool = False,
    device: str = "auto",
) -> AmnesicDropResult:
    """
    Compute comprehensive amnesic drop metrics.
    
    Features:
    - K-fold cross-validation with 95% CI
    - Paired t-test for statistical significance
    - Balanced accuracy (robust to class imbalance)
    - Minority class F1 tracking
    - Learning curves (torch backend)
    
    Args:
        embeddings_before: Embeddings before projection (num_samples, dim).
        embeddings_after: Embeddings after projection (num_samples, dim).
        labels: Demographic labels (num_samples,).
        train_split: Fraction of data for training probe (used if use_kfold=False).
        random_state: Random seed.
        config: ProbeConfig for backend and hyperparameters.
        use_kfold: Whether to use k-fold cross-validation.
        device: Device for torch backend.
        
    Returns:
        AmnesicDropResult with comprehensive metrics.
    """
    if config is None:
        config = ProbeConfig(random_state=random_state)
    
    # Convert to numpy
    X_before = embeddings_before.cpu().numpy()
    X_after = embeddings_after.cpu().numpy()
    y = labels.cpu().numpy()
    
    # Filter out null labels (-1)
    valid_mask = y != -1
    X_before = X_before[valid_mask]
    X_after = X_after[valid_mask]
    y = y[valid_mask]
    
    if len(y) == 0:
        logger.warning("No valid samples after filtering null labels")
        return AmnesicDropResult(
            acc_before=0.0, acc_after=0.0, amnesic_drop=0.0,
            balanced_acc_before=0.0, balanced_acc_after=0.0, balanced_amnesic_drop=0.0,
        )
    
    # Compute majority baseline
    unique, counts = np.unique(y, return_counts=True)
    majority_class = unique[np.argmax(counts)]
    majority_baseline = counts.max() / len(y)
    minority_class = unique[np.argmin(counts)]
    
    learning_curve_before = None
    learning_curve_after = None
    cv_before = None
    cv_after = None
    pvalue = None
    is_significant = False
    
    if use_kfold:
        # K-fold cross-validation
        logger.info(f"Running {config.n_folds}-fold cross-validation...")
        
        cv_before = _compute_cv_result(X_before, y, config, device)
        cv_after = _compute_cv_result(X_after, y, config, device)
        
        acc_before = cv_before.mean_accuracy
        acc_after = cv_after.mean_accuracy
        balanced_acc_before = cv_before.mean_balanced_accuracy
        balanced_acc_after = cv_after.mean_balanced_accuracy
        
        # Paired t-test
        pvalue = _compute_pvalue(cv_before.fold_accuracies, cv_after.fold_accuracies)
        is_significant = pvalue < config.pvalue_threshold
        
        logger.info(
            f"CV Results - Before: {acc_before:.2%} ± {cv_before.std_accuracy:.2%}, "
            f"After: {acc_after:.2%} ± {cv_after.std_accuracy:.2%}, p={pvalue:.4f}"
        )
    else:
        # Single train/test split (legacy mode)
        num_samples = len(X_before)
        indices = np.arange(num_samples)
        np.random.seed(random_state)
        np.random.shuffle(indices)
        
        split_idx = int(num_samples * train_split)
        train_idx = indices[:split_idx]
        test_idx = indices[split_idx:]
        
        # Train probes
        probe_before = create_probe(config, device=device)
        probe_before.fit(X_before[train_idx], y[train_idx])
        result_before = probe_before.evaluate(X_before[test_idx], y[test_idx])
        
        probe_after = create_probe(config, device=device)
        probe_after.fit(X_after[train_idx], y[train_idx])
        result_after = probe_after.evaluate(X_after[test_idx], y[test_idx])
        
        acc_before = result_before.accuracy
        acc_after = result_after.accuracy
        balanced_acc_before = result_before.balanced_accuracy
        balanced_acc_after = result_after.balanced_accuracy
        
        # Get learning curves if torch backend
        if isinstance(probe_before, TorchProbe):
            learning_curve_before = probe_before.get_learning_curve()
        if isinstance(probe_after, TorchProbe):
            learning_curve_after = probe_after.get_learning_curve()
    
    # Compute amnesic drop
    if acc_before > 0:
        amnesic_drop = (acc_before - acc_after) / acc_before
    else:
        amnesic_drop = 0.0
    
    if balanced_acc_before > 0:
        balanced_amnesic_drop = (balanced_acc_before - balanced_acc_after) / balanced_acc_before
    else:
        balanced_amnesic_drop = 0.0
    
    # Compute minority class F1 (full dataset)
    probe_full_before = create_probe(config, device=device)
    probe_full_before.fit(X_before, y)
    y_pred_before = probe_full_before.predict(X_before)
    
    probe_full_after = create_probe(config, device=device)
    probe_full_after.fit(X_after, y)
    y_pred_after = probe_full_after.predict(X_after)
    
    # Per-class F1
    f1_per_class_before = f1_score(y, y_pred_before, average=None, zero_division=0)
    f1_per_class_after = f1_score(y, y_pred_after, average=None, zero_division=0)
    
    minority_idx = np.where(unique == minority_class)[0][0]
    minority_f1_before = float(f1_per_class_before[minority_idx]) if len(f1_per_class_before) > minority_idx else 0.0
    minority_f1_after = float(f1_per_class_after[minority_idx]) if len(f1_per_class_after) > minority_idx else 0.0
    minority_f1_delta = minority_f1_before - minority_f1_after
    
    # Majority baseline gap
    majority_gap_before = acc_before - majority_baseline
    majority_gap_after = acc_after - majority_baseline
    
    logger.info(f"Accuracy before LEACE: {acc_before:.2%}")
    logger.info(f"Accuracy after LEACE: {acc_after:.2%}")
    logger.info(f"Amnesic Drop: {amnesic_drop:.2%}")
    logger.info(f"Balanced Amnesic Drop: {balanced_amnesic_drop:.2%}")
    
    if amnesic_drop > 0.30:
        logger.info("✓ Amnesic drop > 30% (target met)")
    else:
        logger.warning(f"⚠ Amnesic drop {amnesic_drop:.2%} < 30% (target not met)")
    
    return AmnesicDropResult(
        acc_before=acc_before,
        acc_after=acc_after,
        amnesic_drop=amnesic_drop,
        balanced_acc_before=balanced_acc_before,
        balanced_acc_after=balanced_acc_after,
        balanced_amnesic_drop=balanced_amnesic_drop,
        cv_before=cv_before,
        cv_after=cv_after,
        pvalue=pvalue,
        is_significant=is_significant,
        minority_class_f1_before=minority_f1_before,
        minority_class_f1_after=minority_f1_after,
        minority_class_f1_delta=minority_f1_delta,
        majority_baseline=majority_baseline,
        majority_baseline_gap_before=majority_gap_before,
        majority_baseline_gap_after=majority_gap_after,
        learning_curve_before=learning_curve_before,
        learning_curve_after=learning_curve_after,
    )


def benchmark_solver_convergence(
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_test: np.ndarray,
    y_test: np.ndarray,
    config: ProbeConfig,
) -> Dict[str, Any]:
    """
    Benchmark exact solver (cuml/sklearn) vs approximate solver (torch).
    
    Returns metrics comparing the two approaches:
    - solver_accuracy_delta: |acc_exact - acc_torch|
    - weight_cosine_similarity: cosine(weights_exact, weights_torch)
    - time_to_convergence_ratio: time_torch / time_exact
    """
    import time
    
    results: Dict[str, Any] = {}
    
    # Exact solver (sklearn or cuml)
    exact_config = ProbeConfig(
        backend="sklearn",
        max_iter=config.max_iter,
        random_state=config.random_state,
    )
    
    t0 = time.perf_counter()
    exact_probe = create_probe(exact_config)
    exact_probe.fit(X_train, y_train)
    exact_result = exact_probe.evaluate(X_test, y_test)
    t_exact = time.perf_counter() - t0
    
    # Torch solver
    torch_config = ProbeConfig(
        backend="torch",
        max_iter=config.max_iter,
        random_state=config.random_state,
        torch_lr=config.torch_lr,
        torch_epochs=config.torch_epochs,
        torch_batch_size=config.torch_batch_size,
    )
    
    t0 = time.perf_counter()
    torch_probe = create_probe(torch_config)
    torch_probe.fit(X_train, y_train)
    torch_result = torch_probe.evaluate(X_test, y_test)
    t_torch = time.perf_counter() - t0
    
    # Accuracy delta
    results["solver_accuracy_delta"] = abs(exact_result.accuracy - torch_result.accuracy)
    
    # Weight cosine similarity
    try:
        w_exact = exact_probe.get_weights().flatten()
        w_torch = torch_probe.get_weights().flatten()
        
        if len(w_exact) == len(w_torch):
            cos_sim = np.dot(w_exact, w_torch) / (
                np.linalg.norm(w_exact) * np.linalg.norm(w_torch) + 1e-8
            )
            results["weight_cosine_similarity"] = float(cos_sim)
        else:
            results["weight_cosine_similarity"] = None
    except Exception as e:
        logger.warning(f"Could not compute weight cosine similarity: {e}")
        results["weight_cosine_similarity"] = None
    
    # Time ratio
    results["time_to_convergence_ratio"] = t_torch / max(t_exact, 1e-6)
    
    results["exact_accuracy"] = exact_result.accuracy
    results["torch_accuracy"] = torch_result.accuracy
    results["exact_time_s"] = t_exact
    results["torch_time_s"] = t_torch
    
    # Warning if delta > 1%
    if results["solver_accuracy_delta"] > 0.01:
        logger.warning(
            f"Solver accuracy delta {results['solver_accuracy_delta']:.2%} > 1%: "
            "torch probe may not have converged properly"
        )
    
    return results
