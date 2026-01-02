"""
Linear Probe for LEACE Evaluation.

Implements logistic regression probe to measure demographic information leakage
before and after LEACE projection.

Metric: Amnesic Drop = (Acc_before - Acc_after) / Acc_before

Target: Amnesic Drop > 30%

Implements: phaseA-D_implementation_plan.md Section 9.2
"""

import logging
import torch
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import accuracy_score
from typing import Tuple

logger = logging.getLogger(__name__)


class LinearProbe:
    """
    Linear probe for evaluating demographic information leakage.
    
    Uses logistic regression to measure how much demographic information
    is linearly accessible in embeddings.
    """
    
    def __init__(
        self,
        max_iter: int = 1000,
        random_state: int = 42,
    ):
        """
        Initialize linear probe.
        
        Args:
            max_iter: Maximum iterations for logistic regression.
            random_state: Random seed for reproducibility.
        """
        self.max_iter = max_iter
        self.random_state = random_state
        self.classifier = None
    
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
        
        if len(np.unique(y)) < 2:
            logger.warning("Less than 2 unique labels, cannot train probe")
            return 0.0
        
        # Train logistic regression
        self.classifier = LogisticRegression(
            max_iter=self.max_iter,
            random_state=self.random_state,
            multi_class='multinomial' if len(np.unique(y)) > 2 else 'ovr',
        )
        
        self.classifier.fit(X, y)
        
        # Compute training accuracy
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"Probe training accuracy: {accuracy:.2%}")
        return accuracy
    
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
        if self.classifier is None:
            raise ValueError("Probe not trained, call train() first")
        
        # Convert to numpy
        X = embeddings.cpu().numpy()
        y = labels.cpu().numpy()
        
        # Filter out null labels
        valid_mask = y != -1
        X = X[valid_mask]
        y = y[valid_mask]
        
        # Predict
        y_pred = self.classifier.predict(X)
        accuracy = accuracy_score(y, y_pred)
        
        logger.info(f"Probe evaluation accuracy: {accuracy:.2%}")
        return accuracy


def compute_amnesic_drop(
    embeddings_before: torch.Tensor,
    embeddings_after: torch.Tensor,
    labels: torch.Tensor,
    train_split: float = 0.8,
    random_state: int = 42,
) -> Tuple[float, float, float]:
    """
    Compute amnesic drop metric.
    
    Trains probes on embeddings before and after LEACE projection
    and measures the drop in accuracy.
    
    Args:
        embeddings_before: Embeddings before projection (num_samples, dim).
        embeddings_after: Embeddings after projection (num_samples, dim).
        labels: Demographic labels (num_samples,).
        train_split: Fraction of data for training probe.
        random_state: Random seed.
        
    Returns:
        Tuple of (acc_before, acc_after, amnesic_drop).
    """
    # Split data
    num_samples = len(embeddings_before)
    indices = np.arange(num_samples)
    np.random.seed(random_state)
    np.random.shuffle(indices)
    
    split_idx = int(num_samples * train_split)
    train_indices = indices[:split_idx]
    test_indices = indices[split_idx:]
    
    # Train probe on "before" embeddings
    probe_before = LinearProbe(random_state=random_state)
    probe_before.train(
        embeddings_before[train_indices],
        labels[train_indices],
    )
    
    acc_before = probe_before.evaluate(
        embeddings_before[test_indices],
        labels[test_indices],
    )
    
    # Train probe on "after" embeddings
    probe_after = LinearProbe(random_state=random_state)
    probe_after.train(
        embeddings_after[train_indices],
        labels[train_indices],
    )
    
    acc_after = probe_after.evaluate(
        embeddings_after[test_indices],
        labels[test_indices],
    )
    
    # Compute amnesic drop
    if acc_before > 0:
        amnesic_drop = (acc_before - acc_after) / acc_before
    else:
        amnesic_drop = 0.0
    
    logger.info(f"Accuracy before LEACE: {acc_before:.2%}")
    logger.info(f"Accuracy after LEACE: {acc_after:.2%}")
    logger.info(f"Amnesic Drop: {amnesic_drop:.2%}")
    
    if amnesic_drop > 0.30:
        logger.info("✓ Amnesic drop > 30% (target met)")
    else:
        logger.warning(f"⚠ Amnesic drop {amnesic_drop:.2%} < 30% (target not met)")
    
    return acc_before, acc_after, amnesic_drop
# LinearProbe - Train linear probes for self-evaluation
