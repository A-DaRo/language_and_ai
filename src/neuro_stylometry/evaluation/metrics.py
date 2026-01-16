# src/neuro_stylometry/evaluation/metrics.py
"""
Phase A Metrics Computation.

Aggregates dataset statistics, GLiNER detection metrics, LEACE projection
quality metrics, probe accuracy, embedding separability, and class imbalance
metrics into a unified JSON-serializable dict.

Output: phase_a_metrics.json
"""

import hashlib
import json
import logging
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import pyarrow as pa

logger = logging.getLogger(__name__)


def compute_phase_a_metrics(
    clean_table: pa.Table,
    logs_table: pa.Table,
    projection_matrix: Optional["torch.Tensor"] = None,
    strategy_metadata: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
    use_only_labels: Optional[List[str]] = None,
) -> Dict[str, Any]:
    """
    Compute unified Phase A metrics for JSON export.
    
    Args:
        clean_table: Cleaned dataset Arrow table with post_masked column.
        logs_table: Pollution detection logs Arrow table.
        projection_matrix: LEACE projection matrix (768x768).
        strategy_metadata: Metadata returned by strategy.execute().
        config: Pipeline configuration dict (for config_hash).
        use_only_labels: Optional list of demographic labels being filtered.
            If provided, records the label_filter for single-label mode tracking.
        
    Returns:
        Dict suitable for JSON serialization to phase_a_metrics.json.
    """
    import torch
    
    metrics: Dict[str, Any] = {}
    
    # Timestamp
    metrics["timestamp"] = datetime.now(timezone.utc).isoformat()
    
    # Config hash (for experiment tracking)
    if config is not None:
        config_str = json.dumps(config, sort_keys=True, default=str)
        metrics["config_hash"] = hashlib.sha256(config_str.encode()).hexdigest()[:16]
    else:
        metrics["config_hash"] = None
    
    # -------------------------------------------------------------------------
    # Dataset Statistics
    # -------------------------------------------------------------------------
    dataset_stats = _compute_dataset_stats(clean_table)
    metrics["dataset"] = dataset_stats
    
    # -------------------------------------------------------------------------
    # GLiNER Detection Statistics
    # -------------------------------------------------------------------------
    gliner_stats = _compute_gliner_stats(logs_table)
    metrics["gliner"] = gliner_stats
    
    # -------------------------------------------------------------------------
    # LEACE Projection Statistics
    # -------------------------------------------------------------------------
    if projection_matrix is not None:
        leace_stats = _compute_leace_stats(projection_matrix)
        metrics["leace"] = leace_stats
    else:
        metrics["leace"] = None
    
    # -------------------------------------------------------------------------
    # Merge Strategy Metadata (probe accuracy, recall, timing, etc.)
    # -------------------------------------------------------------------------
    if strategy_metadata:
        # Explicit recall
        if "explicit_recall" in strategy_metadata:
            metrics["explicit_recall"] = strategy_metadata["explicit_recall"]
        
        # Probe accuracy (before/after LEACE)
        if "probe" in strategy_metadata:
            metrics["probe"] = strategy_metadata["probe"]
        
        # Timing breakdown (if available)
        if "timing" in strategy_metadata:
            metrics["timing"] = strategy_metadata["timing"]
        
        # Execution mode
        metrics["execution"] = {
            "mode": strategy_metadata.get("mode", "unknown"),
            "staged_execution": strategy_metadata.get("staged_execution", False),
            "num_samples": strategy_metadata.get("num_samples", len(clean_table)),
            "label_filter": use_only_labels or strategy_metadata.get("label_filter"),
        }
    else:
        metrics["execution"] = {
            "mode": "unknown",
            "staged_execution": False,
            "num_samples": len(clean_table),
            "label_filter": use_only_labels,
        }
    
    return metrics


def _compute_dataset_stats(clean_table: pa.Table) -> Dict[str, Any]:
    """Compute dataset-level statistics."""
    n_samples = len(clean_table)
    
    # Check for post_masked column
    if "post_masked" in clean_table.column_names and "post" in clean_table.column_names:
        post_col = clean_table["post"].to_pylist()
        post_masked_col = clean_table["post_masked"].to_pylist()
        
        # Count samples with masks applied
        n_masked = sum(
            1 for orig, masked in zip(post_col, post_masked_col)
            if orig != masked
        )
        
        # Count mask token occurrences
        mask_token_counts: Dict[str, int] = {}
        for masked_text in post_masked_col:
            if masked_text and "[MASK:" in masked_text:
                # Extract mask tokens
                import re
                tokens = re.findall(r"\[MASK:[A-Z_]+\]", masked_text)
                for token in tokens:
                    mask_token_counts[token] = mask_token_counts.get(token, 0) + 1
        
        # Text length statistics
        text_lengths = [len(t) if t else 0 for t in post_col]
        masked_lengths = [len(t) if t else 0 for t in post_masked_col]
    else:
        n_masked = 0
        mask_token_counts = {}
        text_lengths = []
        masked_lengths = []
    
    # Demographic column coverage
    demographic_cols = [
        "birth_year", "female", "nationality", "political_leaning",
        "extrovert", "sensing", "feeling", "judging"
    ]
    demographic_coverage: Dict[str, float] = {}
    for col in demographic_cols:
        if col in clean_table.column_names:
            col_data = clean_table[col]
            n_valid = n_samples - col_data.null_count
            demographic_coverage[col] = n_valid / n_samples if n_samples > 0 else 0.0
    
    return {
        "num_samples": n_samples,
        "num_with_masks": n_masked,
        "mask_rate": n_masked / n_samples if n_samples > 0 else 0.0,
        "mask_token_counts": mask_token_counts,
        "text_length": {
            "mean": float(np.mean(text_lengths)) if text_lengths else 0.0,
            "std": float(np.std(text_lengths)) if text_lengths else 0.0,
            "min": int(min(text_lengths)) if text_lengths else 0,
            "max": int(max(text_lengths)) if text_lengths else 0,
        },
        "masked_text_length": {
            "mean": float(np.mean(masked_lengths)) if masked_lengths else 0.0,
            "std": float(np.std(masked_lengths)) if masked_lengths else 0.0,
        },
        "demographic_coverage": demographic_coverage,
    }


def _compute_gliner_stats(logs_table: pa.Table) -> Dict[str, Any]:
    """Compute GLiNER detection statistics from pollution logs."""
    n_spans = len(logs_table)
    
    if n_spans == 0:
        return {
            "num_spans_detected": 0,
            "spans_per_sample": {"mean": 0.0, "std": 0.0, "max": 0},
            "confidence": {"mean": 0.0, "std": 0.0, "min": 0.0, "max": 0.0},
            "by_entity_type": {},
            "span_length": {"mean": 0.0, "std": 0.0, "max": 0},
        }
    
    # Entity type breakdown
    entity_types = logs_table["entity_type"].to_pylist()
    entity_counts: Dict[str, int] = {}
    for etype in entity_types:
        entity_counts[etype] = entity_counts.get(etype, 0) + 1
    
    # Confidence statistics
    confidences = logs_table["confidence"].to_pylist()
    confidences = [c for c in confidences if c is not None]
    
    # Confidence by entity type
    confidence_by_type: Dict[str, Dict[str, float]] = {}
    if "entity_type" in logs_table.column_names and "confidence" in logs_table.column_names:
        for etype, conf in zip(entity_types, logs_table["confidence"].to_pylist()):
            if etype not in confidence_by_type:
                confidence_by_type[etype] = {"values": []}
            if conf is not None:
                confidence_by_type[etype]["values"].append(conf)
        
        for etype in confidence_by_type:
            vals = confidence_by_type[etype]["values"]
            confidence_by_type[etype] = {
                "mean": float(np.mean(vals)) if vals else 0.0,
                "std": float(np.std(vals)) if vals else 0.0,
                "count": len(vals),
            }
    
    # Span length statistics
    if "span_start" in logs_table.column_names and "span_end" in logs_table.column_names:
        starts = logs_table["span_start"].to_pylist()
        ends = logs_table["span_end"].to_pylist()
        span_lengths = [e - s for s, e in zip(starts, ends) if s is not None and e is not None]
    else:
        span_lengths = []
    
    # Spans per unique post_id
    if "post_id" in logs_table.column_names:
        post_ids = logs_table["post_id"].to_pylist()
        spans_per_post: Dict[str, int] = {}
        for pid in post_ids:
            spans_per_post[pid] = spans_per_post.get(pid, 0) + 1
        spans_counts = list(spans_per_post.values())
    else:
        spans_counts = []
    
    return {
        "num_spans_detected": n_spans,
        "num_unique_posts_with_spans": len(set(post_ids)) if "post_id" in logs_table.column_names else 0,
        "spans_per_sample": {
            "mean": float(np.mean(spans_counts)) if spans_counts else 0.0,
            "std": float(np.std(spans_counts)) if spans_counts else 0.0,
            "max": int(max(spans_counts)) if spans_counts else 0,
        },
        "confidence": {
            "mean": float(np.mean(confidences)) if confidences else 0.0,
            "std": float(np.std(confidences)) if confidences else 0.0,
            "min": float(min(confidences)) if confidences else 0.0,
            "max": float(max(confidences)) if confidences else 0.0,
        },
        "by_entity_type": {
            etype: {
                "count": entity_counts.get(etype, 0),
                **confidence_by_type.get(etype, {"mean": 0.0, "std": 0.0}),
            }
            for etype in sorted(entity_counts.keys())
        },
        "span_length": {
            "mean": float(np.mean(span_lengths)) if span_lengths else 0.0,
            "std": float(np.std(span_lengths)) if span_lengths else 0.0,
            "max": int(max(span_lengths)) if span_lengths else 0,
        },
    }


def _compute_leace_stats(projection_matrix: "torch.Tensor") -> Dict[str, Any]:
    """Compute LEACE projection matrix statistics."""
    import torch
    
    P = projection_matrix
    
    # Shape
    shape = list(P.shape)
    
    # Dtype
    dtype = str(P.dtype)
    
    # Idempotence check: ||P² - P||_F / ||P||_F
    P2 = P @ P
    idempotence_error = float(torch.norm(P2 - P) / torch.norm(P))
    
    # Singular value analysis
    try:
        U, S, Vh = torch.linalg.svd(P)
        singular_values = S.cpu().numpy().tolist()
        
        # Effective rank (number of singular values > 1e-5)
        effective_rank = int((S > 1e-5).sum().item())
        
        # Projection dimensionality loss
        # For an ideal LEACE projection, some singular values should be 0 or near 0
        # indicating directions erased
        n_erased = int((S < 1e-3).sum().item())
    except Exception as e:
        logger.warning(f"SVD computation failed: {e}")
        singular_values = []
        effective_rank = -1
        n_erased = -1
    
    # Frobenius norm
    frobenius_norm = float(torch.norm(P, p="fro"))
    
    # Check for finite values
    is_finite = bool(torch.isfinite(P).all())
    
    return {
        "shape": shape,
        "dtype": dtype,
        "idempotence_error": idempotence_error,
        "idempotence_satisfied": idempotence_error < 1e-5,
        "frobenius_norm": frobenius_norm,
        "is_finite": is_finite,
        "singular_values": {
            "top_10": singular_values[:10] if len(singular_values) >= 10 else singular_values,
            "bottom_10": singular_values[-10:] if len(singular_values) >= 10 else singular_values,
            "effective_rank": effective_rank,
            "n_erased_directions": n_erased,
        },
    }


def save_metrics(metrics: Dict[str, Any], output_path: Path) -> None:
    """Save metrics dict to JSON file."""
    output_path = Path(output_path)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    
    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(metrics, f, indent=2, default=str)
    
    logger.info(f"Saved Phase A metrics to {output_path}")


# =============================================================================
# Extended Phase A Metrics
# =============================================================================

def compute_masking_rate_histogram(
    clean_table: pa.Table,
    n_bins: int = 10,
) -> Dict[str, Any]:
    """
    Compute per-document masking rate histogram.
    
    Masking rate = (number of mask tokens) / (original text length)
    
    Returns:
        Dict with histogram bins, counts, and statistics.
    """
    import re
    
    if "post" not in clean_table.column_names or "post_masked" not in clean_table.column_names:
        return {"error": "Missing post or post_masked columns"}
    
    post_col = clean_table["post"].to_pylist()
    post_masked_col = clean_table["post_masked"].to_pylist()
    
    masking_rates = []
    for orig, masked in zip(post_col, post_masked_col):
        if not orig or not masked:
            masking_rates.append(0.0)
            continue
        
        # Count mask tokens
        n_masks = len(re.findall(r"\[MASK:[A-Z_]+\]", masked))
        # Masking rate as ratio of mask tokens to original words
        orig_words = len(orig.split())
        rate = n_masks / max(orig_words, 1)
        masking_rates.append(rate)
    
    masking_rates = np.array(masking_rates)
    
    # Compute histogram
    hist, bin_edges = np.histogram(masking_rates, bins=n_bins, range=(0, 1))
    
    return {
        "histogram": {
            "counts": hist.tolist(),
            "bin_edges": bin_edges.tolist(),
        },
        "statistics": {
            "mean": float(np.mean(masking_rates)),
            "std": float(np.std(masking_rates)),
            "median": float(np.median(masking_rates)),
            "min": float(np.min(masking_rates)),
            "max": float(np.max(masking_rates)),
            "zero_mask_rate_pct": float((masking_rates == 0).sum() / len(masking_rates) * 100),
        },
    }


def compute_embedding_separability(
    embeddings: np.ndarray,
    labels: np.ndarray,
    sample_size: int = 5000,
) -> Dict[str, Any]:
    """
    Compute embedding separability scores using clustering metrics.
    
    Metrics:
    - Silhouette score: How similar each point is to its own cluster vs other clusters
    - Davies-Bouldin index: Ratio of within-cluster to between-cluster distances
    
    Higher silhouette (closer to 1) = better separation
    Lower Davies-Bouldin = better separation
    
    Args:
        embeddings: Embedding matrix (n_samples, n_features)
        labels: Class labels (n_samples,)
        sample_size: Max samples for computation (performance)
        
    Returns:
        Dict with separability metrics
    """
    from sklearn.metrics import silhouette_score, davies_bouldin_score
    
    # Filter out invalid labels
    valid_mask = labels != -1
    embeddings = embeddings[valid_mask]
    labels = labels[valid_mask]
    
    if len(embeddings) == 0:
        return {"error": "No valid samples"}
    
    # Sample if too large
    if len(embeddings) > sample_size:
        indices = np.random.choice(len(embeddings), sample_size, replace=False)
        embeddings = embeddings[indices]
        labels = labels[indices]
    
    unique_labels = np.unique(labels)
    if len(unique_labels) < 2:
        return {"error": "Need at least 2 classes for separability metrics"}
    
    try:
        silhouette = silhouette_score(embeddings, labels)
        davies_bouldin = davies_bouldin_score(embeddings, labels)
    except Exception as e:
        logger.warning(f"Separability computation failed: {e}")
        return {"error": str(e)}
    
    return {
        "silhouette_score": float(silhouette),
        "davies_bouldin_index": float(davies_bouldin),
        "n_samples": len(embeddings),
        "n_classes": len(unique_labels),
    }


def compute_class_imbalance_metrics(
    labels: np.ndarray,
) -> Dict[str, Any]:
    """
    Compute class imbalance metrics for probe evaluation.
    
    Metrics:
    - Class distribution
    - Imbalance ratio (majority / minority)
    - Majority baseline accuracy (ZeroR)
    """
    # Filter invalid
    valid_labels = labels[labels != -1]
    
    if len(valid_labels) == 0:
        return {"error": "No valid labels"}
    
    unique, counts = np.unique(valid_labels, return_counts=True)
    total = len(valid_labels)
    
    # Class distribution
    distribution = {int(label): int(count) for label, count in zip(unique, counts)}
    proportions = {int(label): float(count / total) for label, count in zip(unique, counts)}
    
    # Imbalance metrics
    majority_count = counts.max()
    minority_count = counts.min()
    imbalance_ratio = majority_count / max(minority_count, 1)
    majority_baseline = majority_count / total
    
    return {
        "distribution": distribution,
        "proportions": proportions,
        "imbalance_ratio": float(imbalance_ratio),
        "majority_baseline": float(majority_baseline),
        "majority_class": int(unique[np.argmax(counts)]),
        "minority_class": int(unique[np.argmin(counts)]),
        "n_classes": len(unique),
    }


def compute_control_probe_metrics(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    target_labels: np.ndarray,
    control_labels: np.ndarray,
    control_name: str = "control",
) -> Dict[str, Any]:
    """
    Compute control probe metrics to verify masking specificity.
    
    A control probe targets a feature that SHOULD NOT change after masking
    (e.g., text length, sentiment). If both target and control drop significantly,
    the masking is destroying general embedding quality rather than specifically
    removing the target pollution.
    
    Metrics:
    - Control stability score: 1 - (acc_before - acc_after) for control
    - Specificity ratio: target_drop / (control_drop + epsilon)
    
    Args:
        embeddings_before: Embeddings before masking/LEACE
        embeddings_after: Embeddings after masking/LEACE
        target_labels: Labels for the target attribute (what we want to remove)
        control_labels: Labels for the control attribute (what should stay)
        control_name: Name for logging
        
    Returns:
        Dict with control probe metrics
    """
    from ..pollution_guard.probe import ProbeConfig, create_probe
    
    config = ProbeConfig(backend="sklearn", max_iter=1000)
    
    # Filter valid samples (both labels must be valid)
    valid_mask = (target_labels != -1) & (control_labels != -1)
    X_before = embeddings_before[valid_mask]
    X_after = embeddings_after[valid_mask]
    y_target = target_labels[valid_mask]
    y_control = control_labels[valid_mask]
    
    if len(X_before) < 100:
        return {"error": "Insufficient samples for control probe"}
    
    # Simple train/test split
    n = len(X_before)
    train_idx = np.arange(int(n * 0.8))
    test_idx = np.arange(int(n * 0.8), n)
    
    results = {}
    
    # Target probe
    target_probe_before = create_probe(config)
    target_probe_before.fit(X_before[train_idx], y_target[train_idx])
    target_acc_before = target_probe_before.evaluate(X_before[test_idx], y_target[test_idx]).accuracy
    
    target_probe_after = create_probe(config)
    target_probe_after.fit(X_after[train_idx], y_target[train_idx])
    target_acc_after = target_probe_after.evaluate(X_after[test_idx], y_target[test_idx]).accuracy
    
    target_drop = target_acc_before - target_acc_after
    
    # Control probe
    control_probe_before = create_probe(config)
    control_probe_before.fit(X_before[train_idx], y_control[train_idx])
    control_acc_before = control_probe_before.evaluate(X_before[test_idx], y_control[test_idx]).accuracy
    
    control_probe_after = create_probe(config)
    control_probe_after.fit(X_after[train_idx], y_control[train_idx])
    control_acc_after = control_probe_after.evaluate(X_after[test_idx], y_control[test_idx]).accuracy
    
    control_drop = control_acc_before - control_acc_after
    
    # Metrics
    control_stability = 1.0 - max(0, control_drop)
    specificity_ratio = target_drop / (control_drop + 1e-6) if control_drop > 0 else float('inf')
    
    results = {
        "target": {
            "acc_before": float(target_acc_before),
            "acc_after": float(target_acc_after),
            "drop": float(target_drop),
        },
        f"control_{control_name}": {
            "acc_before": float(control_acc_before),
            "acc_after": float(control_acc_after),
            "drop": float(control_drop),
        },
        "control_stability_score": float(control_stability),
        "specificity_ratio": float(min(specificity_ratio, 100.0)),  # Cap for JSON
        "is_specific": specificity_ratio > 1.5,  # Heuristic threshold
    }
    
    return results


def compute_per_column_amnesic_drop(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels_dict: Dict[str, np.ndarray],
    config: Optional[Any] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Compute amnesic drop for each demographic column with cross-validation.
    
    Args:
        embeddings_before: Embeddings before LEACE
        embeddings_after: Embeddings after LEACE
        labels_dict: Dict mapping column names to label arrays
        config: Optional ProbeConfig
        
    Returns:
        Dict mapping column names to amnesic drop results with CI
    """
    import torch
    from ..pollution_guard.probe import compute_amnesic_drop_extended, ProbeConfig
    
    if config is None:
        config = ProbeConfig(use_kfold=True, n_folds=5)
    
    results = {}
    
    for col_name, labels in labels_dict.items():
        # Convert to torch if needed
        if isinstance(embeddings_before, np.ndarray):
            emb_before_t = torch.tensor(embeddings_before, dtype=torch.float32)
            emb_after_t = torch.tensor(embeddings_after, dtype=torch.float32)
        else:
            emb_before_t = embeddings_before
            emb_after_t = embeddings_after
        
        if isinstance(labels, np.ndarray):
            labels_t = torch.tensor(labels, dtype=torch.long)
        else:
            labels_t = labels
        
        try:
            result = compute_amnesic_drop_extended(
                embeddings_before=emb_before_t,
                embeddings_after=emb_after_t,
                labels=labels_t,
                config=config,
                use_kfold=True,
            )
            results[col_name] = result.to_dict()
        except Exception as e:
            logger.warning(f"Amnesic drop computation failed for {col_name}: {e}")
            results[col_name] = {"error": str(e)}
    
    return results
