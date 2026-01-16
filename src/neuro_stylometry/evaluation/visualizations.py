"""Plotting utilities for Phase D evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np

try:
    import seaborn as sns
    HAS_SEABORN = True
except ImportError:
    HAS_SEABORN = False


def _safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _read_jsonl(path: Path) -> List[Dict]:
    entries: List[Dict] = []
    if not path.exists():
        return entries
    with path.open("r", encoding="utf-8") as handle:
        for line in handle:
            line = line.strip()
            if not line:
                continue
            try:
                entries.append(json.loads(line))
            except json.JSONDecodeError:
                continue
    return entries


def plot_loss_curve(log_path: Path, output_path: Path, title: str) -> Optional[Path]:
    """Plot training loss over steps from a JSONL log."""
    records = _read_jsonl(log_path)
    if not records:
        return None

    steps = [rec.get("step") for rec in records if "step" in rec]
    losses = [rec.get("loss") for rec in records if "loss" in rec]
    if not steps or not losses:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(8, 4))
    plt.plot(steps, losses, color="#2a6f97", linewidth=1.5)
    plt.title(title)
    plt.xlabel("Step")
    plt.ylabel("Loss")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_task_bars(
    metrics: Dict[str, Dict[str, float]],
    output_path: Path,
    *,
    metric_key: str,
    title: str,
) -> Optional[Path]:
    """Plot a bar chart for per-task metrics."""
    if not metrics:
        return None

    tasks: List[str] = []
    values: List[float] = []
    for task, task_metrics in metrics.items():
        if metric_key in task_metrics:
            tasks.append(task)
            values.append(float(task_metrics[metric_key]))

    if not tasks:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(10, 4))
    plt.bar(tasks, values, color="#588157")
    plt.title(title)
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.xticks(rotation=30, ha="right")
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def plot_comparison_bars(
    baseline: Dict[str, Dict[str, float]],
    constrained: Dict[str, Dict[str, float]],
    output_path: Path,
    *,
    metric_key: str,
    title: str,
) -> Optional[Path]:
    """Plot grouped bars comparing baseline vs constrained per task."""
    tasks = sorted(set(baseline.keys()) | set(constrained.keys()))
    if not tasks:
        return None

    baseline_vals: List[float] = []
    constrained_vals: List[float] = []
    for task in tasks:
        baseline_vals.append(float(baseline.get(task, {}).get(metric_key, 0.0)))
        constrained_vals.append(float(constrained.get(task, {}).get(metric_key, 0.0)))

    output_path.parent.mkdir(parents=True, exist_ok=True)
    x = range(len(tasks))
    width = 0.35

    plt.figure(figsize=(10, 4))
    plt.bar([i - width / 2 for i in x], baseline_vals, width, label="Baseline", color="#335c67")
    plt.bar([i + width / 2 for i in x], constrained_vals, width, label="Constrained", color="#e09f3e")
    plt.title(title)
    plt.ylabel(metric_key.replace("_", " ").title())
    plt.xticks(list(x), tasks, rotation=30, ha="right")
    plt.legend()
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


def load_phase_d_metrics(phase_d_dir: Path) -> Tuple[Dict, Dict]:
    """Load baseline and constrained metrics from Phase D output dirs."""
    baseline_metrics = _safe_read_json(phase_d_dir / "baseline" / "phase_d_metrics.json") or {}
    constrained_metrics = _safe_read_json(phase_d_dir / "constrained" / "phase_d_metrics.json") or {}
    return baseline_metrics, constrained_metrics


def plot_confusion_matrix(
    matrix: List[List[int]],
    output_path: Path,
    *,
    title: str,
) -> Optional[Path]:
    if not matrix:
        return None
    output_path.parent.mkdir(parents=True, exist_ok=True)
    plt.figure(figsize=(4, 4))
    plt.imshow(matrix, cmap="Blues")
    plt.title(title)
    plt.xlabel("Predicted")
    plt.ylabel("True")
    plt.colorbar(fraction=0.046, pad=0.04)
    plt.tight_layout()
    plt.savefig(output_path, dpi=150)
    plt.close()
    return output_path


# =============================================================================
# New Multi-Class Visualization Functions
# =============================================================================


def plot_per_class_f1_bars(
    per_class: Dict[str, Dict[str, float]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    title: str = "F1 Score by Class",
    top_n: Optional[int] = None,
    dpi: int = 150,
) -> Optional[Path]:
    """
    Horizontal bar chart of F1 per class, sorted descending.

    Args:
        per_class: Dict mapping class index (str) to metrics dict with 'f1' key.
        label_map: Dict mapping class name to index, e.g. {"Albania": 0, ...}.
        output_path: Path to save the plot.
        title: Plot title.
        top_n: If set, show only top N classes by F1.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not per_class:
        return None

    # Invert label_map: {0: "Albania", 1: "Argentina", ...}
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Build (class_name, f1, support) tuples
    data = []
    for class_idx_str, metrics in per_class.items():
        try:
            class_idx = int(class_idx_str)
        except ValueError:
            class_idx = -1
        class_name = idx_to_name.get(class_idx, f"Class {class_idx_str}")
        f1 = metrics.get("f1", 0.0)
        support = metrics.get("support", 0)
        data.append((class_name, f1, support))

    # Filter out classes with zero support
    data = [(n, f, s) for n, f, s in data if s > 0]
    if not data:
        return None

    # Sort by F1 descending
    data.sort(key=lambda x: x[1], reverse=True)

    if top_n and len(data) > top_n:
        data = data[:top_n]

    names, f1s, _ = zip(*data)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_height = max(6, len(data) * 0.35)
    fig, ax = plt.subplots(figsize=(10, fig_height))

    # Color gradient: red (low) to green (high)
    colors = plt.cm.RdYlGn(np.array(f1s))
    y_pos = np.arange(len(names))
    ax.barh(y_pos, f1s, color=colors, edgecolor="black", linewidth=0.3)

    ax.set_yticks(y_pos)
    ax.set_yticklabels(names, fontsize=9)
    ax.set_xlabel("F1 Score")
    ax.set_title(title)
    ax.set_xlim(0, 1)
    ax.invert_yaxis()  # Highest at top
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, (name, f1, _) in enumerate(data):
        ax.text(f1 + 0.01, i, f"{f1:.2f}", va="center", fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_topn_confusion_heatmap(
    confusion_matrix: List[List[int]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    top_n: int = 15,
    title: str = "Confusion Matrix (Top Classes)",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Confusion matrix zoomed to top-N most frequent classes.

    Args:
        confusion_matrix: Full NxN confusion matrix as nested lists.
        label_map: Dict mapping class name to index.
        output_path: Path to save the plot.
        top_n: Number of top classes to show.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not confusion_matrix:
        return None

    cm = np.array(confusion_matrix)
    if cm.size == 0:
        return None

    # Invert label_map
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Find top-N classes by total samples (row sums)
    class_totals = cm.sum(axis=1)
    n_classes = min(top_n, len(class_totals))
    top_indices = np.argsort(class_totals)[-n_classes:][::-1]

    # Extract sub-matrix
    sub_cm = cm[np.ix_(top_indices, top_indices)]
    labels = [idx_to_name.get(int(i), f"{i}") for i in top_indices]

    # Truncate long names
    labels = [name[:15] + "..." if len(name) > 18 else name for name in labels]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig_size = max(8, n_classes * 0.6)
    fig, ax = plt.subplots(figsize=(fig_size, fig_size))

    if HAS_SEABORN:
        sns.heatmap(
            sub_cm,
            annot=True,
            fmt="d",
            cmap="Blues",
            xticklabels=labels,
            yticklabels=labels,
            ax=ax,
            cbar_kws={"shrink": 0.8},
        )
    else:
        im = ax.imshow(sub_cm, cmap="Blues")
        ax.set_xticks(np.arange(len(labels)))
        ax.set_yticks(np.arange(len(labels)))
        ax.set_xticklabels(labels)
        ax.set_yticklabels(labels)
        plt.colorbar(im, ax=ax, shrink=0.8)
        # Add annotations
        for i in range(len(labels)):
            for j in range(len(labels)):
                val = sub_cm[i, j]
                if val > 0:
                    ax.text(j, i, str(val), ha="center", va="center", fontsize=7)

    ax.set_xlabel("Predicted")
    ax.set_ylabel("True")
    ax.set_title(title)
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_confusion_delta(
    baseline_cm: List[List[int]],
    constrained_cm: List[List[int]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    top_n: int = 10,
    title: str = "Top Confusion Changes (Baseline → Constrained)",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Show pairs where confusion changed most between models.

    Positive delta = more confusion in constrained (worse).
    Negative delta = less confusion in constrained (better).

    Args:
        baseline_cm: Baseline confusion matrix.
        constrained_cm: Constrained confusion matrix.
        label_map: Dict mapping class name to index.
        output_path: Path to save the plot.
        top_n: Number of top changes to show.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not baseline_cm or not constrained_cm:
        return None

    cm_b = np.array(baseline_cm)
    cm_c = np.array(constrained_cm)

    if cm_b.shape != cm_c.shape:
        return None

    delta = cm_c - cm_b  # Positive = more confusion in constrained

    # Invert label_map
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Find biggest absolute changes (off-diagonal only)
    n = delta.shape[0]
    changes = []
    for i in range(n):
        for j in range(n):
            if i != j and abs(delta[i, j]) > 0:
                changes.append((i, j, int(delta[i, j]), int(cm_b[i, j]), int(cm_c[i, j])))

    if not changes:
        return None

    # Sort by absolute delta
    changes.sort(key=lambda x: abs(x[2]), reverse=True)
    top_changes = changes[:top_n]

    # Build labels and values
    pair_labels = []
    deltas = []
    for i, j, d, b, c in top_changes:
        name_i = idx_to_name.get(i, f"{i}")
        name_j = idx_to_name.get(j, f"{j}")
        # Truncate names
        name_i = name_i[:12] + ".." if len(name_i) > 14 else name_i
        name_j = name_j[:12] + ".." if len(name_j) > 14 else name_j
        pair_labels.append(f"{name_i} → {name_j}")
        deltas.append(d)

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, max(5, len(top_changes) * 0.4)))

    # Diverging bar chart: green = improved, red = degraded
    colors = ["#2d6a4f" if d < 0 else "#ae2012" for d in deltas]
    y_pos = np.arange(len(pair_labels))
    ax.barh(y_pos, deltas, color=colors, edgecolor="black", linewidth=0.3)

    ax.axvline(x=0, color="black", linewidth=0.8)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(pair_labels, fontsize=9)
    ax.set_xlabel("Change in Confusion Count (- = improved, + = degraded)")
    ax.set_title(title)
    ax.invert_yaxis()
    ax.grid(axis="x", alpha=0.3)

    # Add value labels
    for i, d in enumerate(deltas):
        offset = 0.5 if d >= 0 else -0.5
        ha = "left" if d >= 0 else "right"
        ax.text(d + offset, i, f"{d:+d}", va="center", ha=ha, fontsize=8)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_class_distribution(
    per_class: Dict[str, Dict[str, float]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    title: str = "Class Distribution (Sample Counts)",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Histogram of class sample counts to show imbalance.

    Args:
        per_class: Dict mapping class index to metrics with 'support' key.
        label_map: Dict mapping class name to index.
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not per_class:
        return None

    # Extract support values
    supports = []
    for class_idx_str, metrics in per_class.items():
        support = metrics.get("support", 0)
        if support > 0:
            supports.append(support)

    if not supports:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    ax.hist(supports, bins=30, color="#457b9d", edgecolor="black", alpha=0.8)
    ax.axvline(x=np.mean(supports), color="red", linestyle="--", linewidth=2,
               label=f"Mean: {np.mean(supports):.0f}")
    ax.axvline(x=np.median(supports), color="orange", linestyle="--", linewidth=2,
               label=f"Median: {np.median(supports):.0f}")

    ax.set_xlabel("Sample Count per Class")
    ax.set_ylabel("Number of Classes")
    ax.set_title(title)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add stats annotation
    stats_text = (
        f"Classes: {len(supports)}\n"
        f"Total: {sum(supports):,}\n"
        f"Min: {min(supports)}\n"
        f"Max: {max(supports)}"
    )
    ax.text(0.95, 0.95, stats_text, transform=ax.transAxes, fontsize=9,
            verticalalignment="top", horizontalalignment="right",
            bbox=dict(boxstyle="round", facecolor="white", alpha=0.8))

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_performance_vs_support(
    per_class: Dict[str, Dict[str, float]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    title: str = "F1 Score vs Sample Count",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Scatter plot showing relationship between class size and performance.

    Args:
        per_class: Dict mapping class index to metrics with 'f1' and 'support'.
        label_map: Dict mapping class name to index.
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not per_class:
        return None

    # Invert label_map
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Extract data points
    supports = []
    f1s = []
    names = []
    for class_idx_str, metrics in per_class.items():
        support = metrics.get("support", 0)
        f1 = metrics.get("f1", 0.0)
        if support > 0:
            supports.append(support)
            f1s.append(f1)
            try:
                class_idx = int(class_idx_str)
            except ValueError:
                class_idx = -1
            names.append(idx_to_name.get(class_idx, f"Class {class_idx_str}"))

    if not supports:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 6))

    # Scatter with color based on F1
    scatter = ax.scatter(supports, f1s, c=f1s, cmap="RdYlGn", s=60,
                         edgecolor="black", linewidth=0.5, alpha=0.8)
    plt.colorbar(scatter, ax=ax, label="F1 Score")

    ax.set_xlabel("Sample Count (Support)")
    ax.set_ylabel("F1 Score")
    ax.set_title(title)
    ax.set_ylim(0, 1)
    ax.grid(alpha=0.3)

    # Add trend line
    if len(supports) > 2:
        z = np.polyfit(supports, f1s, 1)
        p = np.poly1d(z)
        x_line = np.linspace(min(supports), max(supports), 100)
        ax.plot(x_line, p(x_line), "r--", alpha=0.5, label="Trend")
        ax.legend()

    # Annotate outliers (best and worst performers)
    if len(supports) >= 5:
        top_idx = np.argsort(f1s)[-3:]  # Top 3
        bottom_idx = np.argsort(f1s)[:3]  # Bottom 3
        for idx in list(top_idx) + list(bottom_idx):
            name = names[idx][:10]
            ax.annotate(name, (supports[idx], f1s[idx]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_loss_comparison(
    baseline_log_path: Path,
    constrained_log_path: Path,
    output_path: Path,
    *,
    title: str = "Training Loss: Baseline vs Constrained",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Overlay baseline and constrained loss curves on same plot.

    Args:
        baseline_log_path: Path to baseline training_log.jsonl.
        constrained_log_path: Path to constrained training_log.jsonl.
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    baseline_records = _read_jsonl(baseline_log_path)
    constrained_records = _read_jsonl(constrained_log_path)

    if not baseline_records and not constrained_records:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(12, 5))

    if baseline_records:
        steps_b = [r.get("step", i) for i, r in enumerate(baseline_records) if "loss" in r]
        losses_b = [r["loss"] for r in baseline_records if "loss" in r]
        if steps_b:
            ax.plot(steps_b, losses_b, color="#335c67", linewidth=1.5,
                    label="Baseline", alpha=0.8)

    if constrained_records:
        steps_c = [r.get("step", i) for i, r in enumerate(constrained_records) if "loss" in r]
        losses_c = [r["loss"] for r in constrained_records if "loss" in r]
        if steps_c:
            ax.plot(steps_c, losses_c, color="#e09f3e", linewidth=1.5,
                    label="Constrained", alpha=0.8)

    ax.set_xlabel("Step")
    ax.set_ylabel("Loss")
    ax.set_title(title)
    ax.legend()
    ax.grid(alpha=0.3)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_precision_recall_scatter(
    per_class: Dict[str, Dict[str, float]],
    label_map: Optional[Dict[str, int]],
    output_path: Path,
    *,
    title: str = "Precision vs Recall by Class",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Scatter plot of precision vs recall for each class.

    Args:
        per_class: Dict mapping class index to metrics with 'precision' and 'recall'.
        label_map: Dict mapping class name to index.
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not per_class:
        return None

    # Invert label_map
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Extract data
    precisions = []
    recalls = []
    names = []
    supports = []
    for class_idx_str, metrics in per_class.items():
        p = metrics.get("precision", 0.0)
        r = metrics.get("recall", 0.0)
        s = metrics.get("support", 0)
        if s > 0:
            precisions.append(p)
            recalls.append(r)
            supports.append(s)
            try:
                class_idx = int(class_idx_str)
            except ValueError:
                class_idx = -1
            names.append(idx_to_name.get(class_idx, f"Class {class_idx_str}"))

    if not precisions:
        return None

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 8))

    # Size based on support
    sizes = np.array(supports)
    sizes = 20 + 80 * (sizes - sizes.min()) / (sizes.max() - sizes.min() + 1)

    scatter = ax.scatter(recalls, precisions, s=sizes, c=supports, cmap="viridis",
                         edgecolor="black", linewidth=0.5, alpha=0.7)
    plt.colorbar(scatter, ax=ax, label="Sample Count")

    # Add diagonal (P=R line)
    ax.plot([0, 1], [0, 1], "r--", alpha=0.5, label="P = R")

    ax.set_xlabel("Recall")
    ax.set_ylabel("Precision")
    ax.set_title(title)
    ax.set_xlim(0, 1.05)
    ax.set_ylim(0, 1.05)
    ax.legend()
    ax.grid(alpha=0.3)

    # Annotate some outliers
    if len(precisions) >= 5:
        # Find points far from diagonal
        distances = [abs(p - r) for p, r in zip(precisions, recalls)]
        outlier_idx = np.argsort(distances)[-5:]
        for idx in outlier_idx:
            name = names[idx][:10]
            ax.annotate(name, (recalls[idx], precisions[idx]), fontsize=7,
                        xytext=(5, 5), textcoords="offset points")

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path


def plot_head_classification_comparison(
    baseline_summary: Dict[str, int],
    constrained_summary: Dict[str, int],
    output_path: Path,
    *,
    title: str = "Attention Head Classification: Baseline vs Constrained",
    dpi: int = 150,
) -> Optional[Path]:
    """
    Grouped bar chart comparing CHG head classification between models.

    Args:
        baseline_summary: Dict with 'facilitating', 'irrelevant', 'neutral' counts.
        constrained_summary: Dict with same keys.
        output_path: Path to save the plot.
        title: Plot title.
        dpi: Figure DPI.

    Returns:
        Path to saved figure, or None if no data.
    """
    if not baseline_summary or not constrained_summary:
        return None

    categories = ["Facilitating", "Irrelevant", "Neutral"]
    baseline_vals = [
        baseline_summary.get("facilitating", 0),
        baseline_summary.get("irrelevant", 0),
        baseline_summary.get("neutral", 0),
    ]
    constrained_vals = [
        constrained_summary.get("facilitating", 0),
        constrained_summary.get("irrelevant", 0),
        constrained_summary.get("neutral", 0),
    ]

    output_path.parent.mkdir(parents=True, exist_ok=True)
    fig, ax = plt.subplots(figsize=(10, 5))

    x = np.arange(len(categories))
    width = 0.35

    bars1 = ax.bar(x - width / 2, baseline_vals, width, label="Baseline",
                   color="#335c67", edgecolor="black")
    bars2 = ax.bar(x + width / 2, constrained_vals, width, label="Constrained",
                   color="#e09f3e", edgecolor="black")

    ax.set_ylabel("Number of Attention Heads")
    ax.set_title(title)
    ax.set_xticks(x)
    ax.set_xticklabels(categories)
    ax.legend()
    ax.grid(axis="y", alpha=0.3)

    # Add value labels
    for bars in [bars1, bars2]:
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{int(height)}",
                        xy=(bar.get_x() + bar.get_width() / 2, height),
                        xytext=(0, 3), textcoords="offset points",
                        ha="center", va="bottom", fontsize=10)

    # Add delta annotations
    deltas = [c - b for b, c in zip(baseline_vals, constrained_vals)]
    for i, (cat, delta) in enumerate(zip(categories, deltas)):
        if delta != 0:
            color = "#2d6a4f" if (cat == "Irrelevant" and delta > 0) else "#ae2012"
            if cat == "Facilitating" and delta < 0:
                color = "#2d6a4f"
            ax.text(i, max(baseline_vals[i], constrained_vals[i]) + 5,
                    f"Δ: {delta:+d}", ha="center", fontsize=9, color=color)

    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    return output_path
