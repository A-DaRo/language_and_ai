# src/neuro_stylometry/evaluation/visualizations_phase_a.py
"""
Phase A Visualization Module.

Generates all Phase A visualizations as PNG files:
1. PCA scatter plots (before/after LEACE) colored by demographic
2. Singular value spectrum of projection matrix
3. Embedding norm distributions
4. Amnesic drop bar charts
5. GLiNER confidence histograms
6. Detection count by entity type
7. Explicit recall per-column bars
8. Entity co-occurrence heatmap
9. Demographic score scatter matrix
10. Masking coverage statistics

All plots are saved to the reports subdirectory.
"""

import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import seaborn as sns

logger = logging.getLogger(__name__)

# Matplotlib backend for headless operation
plt.switch_backend("Agg")

# Default plot style
plt.style.use("seaborn-v0_8-whitegrid")

# Demographic columns for visualization
DEMOGRAPHIC_COLS = [
    "birth_year", "female", "nationality", "political_leaning",
    "extrovert", "sensing", "feeling", "judging"
]

# Friendly display names
DEMOGRAPHIC_DISPLAY_NAMES = {
    "birth_year": "Birth Year",
    "female": "Gender",
    "nationality": "Nationality",
    "political_leaning": "Political Leaning",
    "extrovert": "E/I (MBTI)",
    "sensing": "S/N (MBTI)",
    "feeling": "F/T (MBTI)",
    "judging": "J/P (MBTI)",
}


def generate_phase_a_plots(
    clean_df_sample: pd.DataFrame,
    logs_df: pd.DataFrame,
    projection_matrix: "torch.Tensor",
    embedder: "FrozenEmbedder",
    output_dir: Path,
    metrics: Optional[Dict[str, Any]] = None,
    config: Optional[Dict[str, Any]] = None,
) -> None:
    """
    Generate all Phase A visualizations and save to output_dir.
    
    Args:
        clean_df_sample: Sampled clean dataset DataFrame (≤2000 rows recommended).
        logs_df: Pollution logs DataFrame.
        projection_matrix: LEACE projection matrix (768×768).
        embedder: FrozenEmbedder for computing embeddings.
        output_dir: Directory to save PNG files.
        metrics: Optional metrics dict (for amnesic drop plot).
        config: Optional config dict (for visualization settings).
    """
    import torch
    
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    
    # Extract visualization config
    viz_config = (config or {}).get("visualization", {})
    dpi = viz_config.get("figure_dpi", 150)
    palette = viz_config.get("color_palette", "husl")
    pca_max_samples = viz_config.get("pca_max_samples", 1000)
    batch_size = viz_config.get("embedding_batch_size", 32)
    plot_toggles = viz_config.get("plots", {})
    
    logger.info(f"Generating Phase A visualizations in {output_dir}")
    
    # -------------------------------------------------------------------------
    # Compute embeddings for PCA and score matrix plots
    # Sample to pca_max_samples if needed
    # -------------------------------------------------------------------------
    embeddings_before = None
    embeddings_after = None
    pca_sample_df = clean_df_sample
    
    if len(clean_df_sample) > pca_max_samples:
        pca_sample_df = clean_df_sample.sample(n=pca_max_samples, random_state=42)
        logger.info(f"Sampled {pca_max_samples} rows for embedding-based visualizations")
    
    # Check if we need embeddings (for PCA, norms, scores)
    needs_embeddings = any([
        plot_toggles.get("pca_before_after", True),
        plot_toggles.get("embedding_norms", True),
        plot_toggles.get("demographic_scores", True),
    ])
    
    if needs_embeddings and "post_masked" in pca_sample_df.columns:
        logger.info("Computing embeddings for visualization...")
        texts = pca_sample_df["post_masked"].fillna("").tolist()
        
        try:
            embeddings_before = embedder.embed_texts(texts, batch_size=batch_size)
            embeddings_after = embeddings_before @ projection_matrix.T.to(embeddings_before.device)
            logger.info(f"Computed embeddings: {embeddings_before.shape}")
        except Exception as e:
            logger.warning(f"Embedding computation failed: {e}")
            embeddings_before = None
            embeddings_after = None
    
    # -------------------------------------------------------------------------
    # Generate individual plots
    # -------------------------------------------------------------------------
    
    # 1. PCA before/after LEACE (per demographic)
    if plot_toggles.get("pca_before_after", True) and embeddings_before is not None:
        try:
            plot_pca_before_after(
                embeddings_before=embeddings_before.cpu().numpy(),
                embeddings_after=embeddings_after.cpu().numpy(),
                labels_df=pca_sample_df,
                output_dir=output_dir,
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"PCA plot failed: {e}")
    
    # 2. Singular value spectrum
    if plot_toggles.get("singular_values", True):
        try:
            plot_singular_values(
                projection_matrix=projection_matrix,
                output_path=output_dir / "singular_values.png",
                dpi=dpi,
            )
        except Exception as e:
            logger.warning(f"Singular values plot failed: {e}")
    
    # 3. Embedding norm distributions
    if plot_toggles.get("embedding_norms", True) and embeddings_before is not None:
        try:
            plot_embedding_norm_distribution(
                embeddings_before=embeddings_before.cpu().numpy(),
                embeddings_after=embeddings_after.cpu().numpy(),
                output_path=output_dir / "embedding_norms.png",
                dpi=dpi,
            )
        except Exception as e:
            logger.warning(f"Embedding norms plot failed: {e}")
    
    # 4. Amnesic drop bar chart (requires metrics)
    if plot_toggles.get("amnesic_drop_bars", True) and metrics is not None:
        try:
            plot_amnesic_drop_bars(
                metrics=metrics,
                output_path=output_dir / "amnesic_drop_bars.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"Amnesic drop plot failed: {e}")
    
    # 5. GLiNER confidence histogram
    if plot_toggles.get("gliner_confidence", True) and len(logs_df) > 0:
        try:
            plot_gliner_confidence_histogram(
                logs_df=logs_df,
                output_path=output_dir / "gliner_confidence.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"GLiNER confidence plot failed: {e}")
    
    # 6. Detection count by entity type
    if plot_toggles.get("detection_counts", True) and len(logs_df) > 0:
        try:
            plot_detection_count_by_type(
                logs_df=logs_df,
                output_path=output_dir / "detection_counts.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"Detection counts plot failed: {e}")
    
    # 7. Explicit recall bar chart (requires metrics)
    if plot_toggles.get("explicit_recall_bars", True) and metrics is not None:
        try:
            plot_explicit_recall_bars(
                metrics=metrics,
                output_path=output_dir / "explicit_recall_bars.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"Explicit recall plot failed: {e}")
    
    # 8. Entity co-occurrence heatmap
    if plot_toggles.get("entity_cooccurrence", True) and len(logs_df) > 0:
        try:
            plot_entity_cooccurrence(
                logs_df=logs_df,
                output_path=output_dir / "entity_cooccurrence.png",
                dpi=dpi,
            )
        except Exception as e:
            logger.warning(f"Entity co-occurrence plot failed: {e}")
    
    # 9. Demographic score scatter matrix
    if plot_toggles.get("demographic_scores", True) and embeddings_before is not None:
        try:
            plot_demographic_score_matrix(
                embeddings=embeddings_before.cpu().numpy(),
                labels_df=pca_sample_df,
                output_path=output_dir / "demographic_scores.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"Demographic scores plot failed: {e}")
    
    # 10. Masking coverage statistics
    if plot_toggles.get("masking_coverage", True):
        try:
            plot_masking_coverage(
                clean_df=clean_df_sample,
                logs_df=logs_df,
                output_path=output_dir / "masking_coverage.png",
                dpi=dpi,
                palette=palette,
            )
        except Exception as e:
            logger.warning(f"Masking coverage plot failed: {e}")
    
    logger.info(f"Phase A visualizations saved to {output_dir}")


# =============================================================================
# Individual Plot Functions
# =============================================================================

def plot_pca_before_after(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    labels_df: pd.DataFrame,
    output_dir: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """
    Generate PCA scatter plots comparing embeddings before and after LEACE.
    
    Creates one plot per demographic column with non-null values.
    Each plot has two subplots: before projection (left) and after (right).
    """
    from sklearn.decomposition import PCA
    
    # Fit PCA on combined embeddings for consistent axes
    all_embeddings = np.vstack([embeddings_before, embeddings_after])
    pca = PCA(n_components=2, random_state=42)
    pca.fit(all_embeddings)
    
    pca_before = pca.transform(embeddings_before)
    pca_after = pca.transform(embeddings_after)
    
    for col in DEMOGRAPHIC_COLS:
        if col not in labels_df.columns:
            continue
        
        # Filter to non-null labels
        mask = labels_df[col].notna()
        if mask.sum() < 10:  # Skip if too few samples
            continue
        
        labels = labels_df.loc[mask, col].values
        pca_b = pca_before[mask.values]
        pca_a = pca_after[mask.values]
        
        # Handle continuous vs categorical
        if col == "birth_year":
            # Bin birth years for color coding
            bins = [1940, 1960, 1980, 1990, 2000, 2010]
            labels_binned = pd.cut(labels, bins=bins, labels=["<1960", "1960s-70s", "1980s", "1990s", "2000s"])
            labels_binned = labels_binned.astype(str)
        else:
            labels_binned = labels.astype(str)
        
        # Create figure
        fig, axes = plt.subplots(1, 2, figsize=(14, 6))
        
        # Unique labels for consistent coloring
        unique_labels = sorted(set(labels_binned))
        colors = sns.color_palette(palette, n_colors=len(unique_labels))
        color_map = {label: colors[i] for i, label in enumerate(unique_labels)}
        
        # Before LEACE
        for label in unique_labels:
            idx = labels_binned == label
            axes[0].scatter(
                pca_b[idx, 0], pca_b[idx, 1],
                c=[color_map[label]], label=label, alpha=0.6, s=20
            )
        axes[0].set_title(f"Before LEACE Projection")
        axes[0].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        axes[0].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        axes[0].legend(title=DEMOGRAPHIC_DISPLAY_NAMES.get(col, col), loc="best", fontsize=8)
        
        # After LEACE
        for label in unique_labels:
            idx = labels_binned == label
            axes[1].scatter(
                pca_a[idx, 0], pca_a[idx, 1],
                c=[color_map[label]], label=label, alpha=0.6, s=20
            )
        axes[1].set_title(f"After LEACE Projection")
        axes[1].set_xlabel(f"PC1 ({pca.explained_variance_ratio_[0]:.1%} var)")
        axes[1].set_ylabel(f"PC2 ({pca.explained_variance_ratio_[1]:.1%} var)")
        axes[1].legend(title=DEMOGRAPHIC_DISPLAY_NAMES.get(col, col), loc="best", fontsize=8)
        
        fig.suptitle(f"PCA of Embeddings by {DEMOGRAPHIC_DISPLAY_NAMES.get(col, col)}", fontsize=14)
        plt.tight_layout()
        
        output_path = output_dir / f"pca_{col}.png"
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path}")


def plot_singular_values(
    projection_matrix: "torch.Tensor",
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot singular value spectrum of the LEACE projection matrix."""
    import torch
    
    P = projection_matrix.cpu()
    _, S, _ = torch.linalg.svd(P)
    singular_values = S.numpy()
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Full spectrum
    axes[0].plot(singular_values, "b-", linewidth=1.5)
    axes[0].axhline(y=1.0, color="r", linestyle="--", alpha=0.7, label="σ=1 (preserved)")
    axes[0].axhline(y=0.0, color="g", linestyle="--", alpha=0.7, label="σ=0 (erased)")
    axes[0].set_xlabel("Index")
    axes[0].set_ylabel("Singular Value")
    axes[0].set_title("Full Singular Value Spectrum")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Log scale for detail
    axes[1].semilogy(singular_values + 1e-10, "b-", linewidth=1.5)
    axes[1].axhline(y=1.0, color="r", linestyle="--", alpha=0.7, label="σ=1")
    axes[1].axhline(y=1e-5, color="orange", linestyle="--", alpha=0.7, label="σ=1e-5 (threshold)")
    axes[1].set_xlabel("Index")
    axes[1].set_ylabel("Singular Value (log scale)")
    axes[1].set_title("Log-Scale Singular Values")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    # Summary statistics in title
    n_preserved = (singular_values > 0.99).sum()
    n_erased = (singular_values < 1e-3).sum()
    fig.suptitle(
        f"LEACE Projection Matrix SVD\n"
        f"Preserved directions (σ≈1): {n_preserved}, Erased directions (σ≈0): {n_erased}",
        fontsize=12
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_embedding_norm_distribution(
    embeddings_before: np.ndarray,
    embeddings_after: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot histogram comparing embedding norms before and after LEACE."""
    norms_before = np.linalg.norm(embeddings_before, axis=1)
    norms_after = np.linalg.norm(embeddings_after, axis=1)
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    # Overlapping histograms
    axes[0].hist(norms_before, bins=50, alpha=0.6, label="Before LEACE", color="blue")
    axes[0].hist(norms_after, bins=50, alpha=0.6, label="After LEACE", color="orange")
    axes[0].set_xlabel("Embedding L2 Norm")
    axes[0].set_ylabel("Count")
    axes[0].set_title("Distribution of Embedding Norms")
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Norm ratio distribution
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_ratio = norms_after / norms_before
        norm_ratio = norm_ratio[np.isfinite(norm_ratio)]
    
    axes[1].hist(norm_ratio, bins=50, alpha=0.7, color="green")
    axes[1].axvline(x=1.0, color="red", linestyle="--", label="Ratio=1 (unchanged)")
    axes[1].axvline(x=np.mean(norm_ratio), color="blue", linestyle="-", label=f"Mean={np.mean(norm_ratio):.3f}")
    axes[1].set_xlabel("Norm Ratio (After / Before)")
    axes[1].set_ylabel("Count")
    axes[1].set_title("Norm Shrinkage Distribution")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    fig.suptitle(
        f"Embedding Norm Analysis\n"
        f"Before: μ={norms_before.mean():.2f}, σ={norms_before.std():.2f} | "
        f"After: μ={norms_after.mean():.2f}, σ={norms_after.std():.2f}",
        fontsize=11
    )
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_amnesic_drop_bars(
    metrics: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """Plot bar chart of probe accuracy before/after LEACE and amnesic drop."""
    probe = metrics.get("probe", {})
    
    if not probe:
        logger.warning("No probe data in metrics, skipping amnesic drop plot")
        return
    
    # Extract per-column data
    per_column = probe.get("per_column_drop", {})
    
    if not per_column:
        # Fallback to overall metrics only
        acc_before = probe.get("accuracy_before", 0.5)
        acc_after = probe.get("accuracy_after", 0.5)
        drop = probe.get("amnesic_drop", acc_before - acc_after)
        
        fig, ax = plt.subplots(figsize=(8, 5))
        x = np.arange(2)
        bars = ax.bar(x, [acc_before, acc_after], color=["steelblue", "coral"])
        ax.set_xticks(x)
        ax.set_xticklabels(["Before LEACE", "After LEACE"])
        ax.set_ylabel("Probe Accuracy")
        ax.set_title(f"Amnesic Probing: Overall Drop = {drop:.1%}")
        ax.set_ylim(0, 1)
        ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
        ax.legend()
        
        # Add value labels
        for bar in bars:
            height = bar.get_height()
            ax.annotate(f"{height:.1%}",
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 3), textcoords="offset points",
                ha="center", va="bottom", fontsize=10)
        
        plt.tight_layout()
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path}")
        return
    
    # Multi-column plot
    columns = list(per_column.keys())
    acc_before = [per_column[c].get("accuracy_before", 0.5) for c in columns]
    acc_after = [per_column[c].get("accuracy_after", 0.5) for c in columns]
    drops = [per_column[c].get("drop", 0) for c in columns]
    
    display_names = [DEMOGRAPHIC_DISPLAY_NAMES.get(c, c) for c in columns]
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 6))
    
    x = np.arange(len(columns))
    width = 0.35
    
    # Before/After comparison
    bars1 = axes[0].bar(x - width/2, acc_before, width, label="Before LEACE", color="steelblue")
    bars2 = axes[0].bar(x + width/2, acc_after, width, label="After LEACE", color="coral")
    axes[0].set_ylabel("Probe Accuracy")
    axes[0].set_xlabel("Demographic Attribute")
    axes[0].set_title("Probe Accuracy Before vs After LEACE")
    axes[0].set_xticks(x)
    axes[0].set_xticklabels(display_names, rotation=45, ha="right")
    axes[0].legend()
    axes[0].set_ylim(0, 1)
    axes[0].axhline(y=0.5, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Amnesic drop bars
    colors = sns.color_palette(palette, n_colors=len(columns))
    bars3 = axes[1].bar(x, drops, color=colors)
    axes[1].set_ylabel("Amnesic Drop (Before - After)")
    axes[1].set_xlabel("Demographic Attribute")
    axes[1].set_title("Amnesic Drop by Attribute")
    axes[1].set_xticks(x)
    axes[1].set_xticklabels(display_names, rotation=45, ha="right")
    axes[1].axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Threshold (0.3)")
    axes[1].legend()
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Add value labels on drop bars
    for bar, drop in zip(bars3, drops):
        height = bar.get_height()
        axes[1].annotate(f"{drop:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_gliner_confidence_histogram(
    logs_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """Plot confidence score distribution per entity type."""
    if "confidence" not in logs_df.columns or "entity_type" not in logs_df.columns:
        logger.warning("Missing columns for confidence histogram")
        return
    
    entity_types = logs_df["entity_type"].unique()
    n_types = len(entity_types)
    
    if n_types == 0:
        return
    
    # Create subplots grid
    n_cols = min(3, n_types)
    n_rows = (n_types + n_cols - 1) // n_cols
    
    fig, axes = plt.subplots(n_rows, n_cols, figsize=(5 * n_cols, 4 * n_rows))
    if n_types == 1:
        axes = np.array([[axes]])
    elif n_rows == 1:
        axes = axes.reshape(1, -1)
    
    colors = sns.color_palette(palette, n_colors=n_types)
    
    for idx, entity_type in enumerate(sorted(entity_types)):
        row, col = idx // n_cols, idx % n_cols
        ax = axes[row, col]
        
        conf_values = logs_df[logs_df["entity_type"] == entity_type]["confidence"].dropna()
        
        ax.hist(conf_values, bins=30, alpha=0.7, color=colors[idx], edgecolor="black")
        ax.axvline(x=conf_values.mean(), color="red", linestyle="--", 
                   label=f"Mean: {conf_values.mean():.2f}")
        ax.set_xlabel("Confidence Score")
        ax.set_ylabel("Count")
        ax.set_title(f"{entity_type} (n={len(conf_values)})")
        ax.legend(fontsize=8)
        ax.set_xlim(0, 1)
        ax.grid(True, alpha=0.3)
    
    # Hide unused subplots
    for idx in range(n_types, n_rows * n_cols):
        row, col = idx // n_cols, idx % n_cols
        axes[row, col].set_visible(False)
    
    fig.suptitle("GLiNER Confidence Distribution by Entity Type", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_detection_count_by_type(
    logs_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """Plot horizontal bar chart of detection counts by entity type."""
    if "entity_type" not in logs_df.columns:
        return
    
    counts = logs_df["entity_type"].value_counts().sort_values()
    
    fig, ax = plt.subplots(figsize=(10, max(6, len(counts) * 0.5)))
    
    colors = sns.color_palette(palette, n_colors=len(counts))
    bars = ax.barh(counts.index, counts.values, color=colors)
    
    ax.set_xlabel("Number of Detected Spans")
    ax.set_ylabel("Entity Type")
    ax.set_title(f"GLiNER Detection Counts by Entity Type (Total: {counts.sum():,})")
    ax.grid(True, alpha=0.3, axis="x")
    
    # Add count labels
    for bar, count in zip(bars, counts.values):
        ax.annotate(f"{count:,}",
            xy=(count, bar.get_y() + bar.get_height() / 2),
            xytext=(5, 0), textcoords="offset points",
            ha="left", va="center", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_explicit_recall_bars(
    metrics: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """Plot per-column explicit recall bar chart."""
    recall_data = metrics.get("explicit_recall", {})
    
    if not recall_data:
        logger.warning("No explicit recall data in metrics")
        return
    
    # Extract per-column recall
    per_column = {k: v for k, v in recall_data.items() if k != "overall"}
    overall = recall_data.get("overall", 0)
    
    if not per_column:
        return
    
    columns = list(per_column.keys())
    values = [per_column[c] if isinstance(per_column[c], (int, float)) else per_column[c].get("recall", 0) 
              for c in columns]
    display_names = [DEMOGRAPHIC_DISPLAY_NAMES.get(c, c) for c in columns]
    
    fig, ax = plt.subplots(figsize=(10, 6))
    
    colors = sns.color_palette(palette, n_colors=len(columns))
    x = np.arange(len(columns))
    bars = ax.bar(x, values, color=colors)
    
    ax.axhline(y=overall, color="red", linestyle="--", linewidth=2, label=f"Overall: {overall:.1%}")
    ax.axhline(y=0.8, color="orange", linestyle=":", alpha=0.7, label="Threshold (80%)")
    
    ax.set_ylabel("Explicit Recall")
    ax.set_xlabel("Demographic Column")
    ax.set_title("Explicit Recall by Demographic Column")
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylim(0, 1)
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, val in zip(bars, values):
        height = bar.get_height()
        ax.annotate(f"{val:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_entity_cooccurrence(
    logs_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """Plot heatmap of entity type co-occurrence within posts."""
    if "entity_type" not in logs_df.columns or "post_id" not in logs_df.columns:
        return
    
    # Build co-occurrence matrix
    entity_types = sorted(logs_df["entity_type"].unique())
    n_types = len(entity_types)
    
    if n_types < 2:
        logger.warning("Too few entity types for co-occurrence matrix")
        return
    
    cooccurrence = np.zeros((n_types, n_types), dtype=int)
    type_to_idx = {t: i for i, t in enumerate(entity_types)}
    
    for post_id, group in logs_df.groupby("post_id"):
        types_in_post = group["entity_type"].unique()
        for t1 in types_in_post:
            for t2 in types_in_post:
                i, j = type_to_idx[t1], type_to_idx[t2]
                cooccurrence[i, j] += 1
    
    fig, ax = plt.subplots(figsize=(10, 8))
    
    # Create mask for upper triangle (show lower + diagonal only)
    mask = np.triu(np.ones_like(cooccurrence, dtype=bool), k=1)
    
    sns.heatmap(
        cooccurrence,
        mask=mask,
        annot=True,
        fmt="d",
        cmap="YlOrRd",
        xticklabels=entity_types,
        yticklabels=entity_types,
        ax=ax,
        cbar_kws={"label": "Co-occurrence Count"}
    )
    
    ax.set_title("Entity Type Co-occurrence Matrix\n(Diagonal = self count)")
    plt.xticks(rotation=45, ha="right")
    plt.yticks(rotation=0)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_demographic_score_matrix(
    embeddings: np.ndarray,
    labels_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """
    Plot lower-triangle scatter matrix of demographic score axes.
    
    For each demographic with non-null values, compute a "score direction"
    (mean embedding of class 1 - mean embedding of class 0) and project embeddings.
    """
    # Compute score directions for binary demographics
    binary_cols = ["female", "extrovert", "sensing", "feeling", "judging"]
    score_directions = {}
    
    for col in binary_cols:
        if col not in labels_df.columns:
            continue
        
        mask = labels_df[col].notna()
        if mask.sum() < 20:
            continue
        
        labels = labels_df.loc[mask, col].values
        emb = embeddings[mask.values]
        
        # Get mean embeddings for each class
        class_0 = emb[labels == 0]
        class_1 = emb[labels == 1]
        
        if len(class_0) < 5 or len(class_1) < 5:
            continue
        
        # Direction: class_1 mean - class_0 mean
        direction = class_1.mean(axis=0) - class_0.mean(axis=0)
        direction = direction / (np.linalg.norm(direction) + 1e-10)
        score_directions[col] = direction
    
    if len(score_directions) < 2:
        logger.warning("Not enough binary demographics for score matrix")
        return
    
    # Project all embeddings onto score directions
    cols = list(score_directions.keys())
    projections = {}
    for col in cols:
        projections[col] = embeddings @ score_directions[col]
    
    # Create pairplot-style lower triangle
    n_cols = len(cols)
    fig, axes = plt.subplots(n_cols, n_cols, figsize=(3 * n_cols, 3 * n_cols))
    
    for i in range(n_cols):
        for j in range(n_cols):
            ax = axes[i, j]
            
            if i < j:  # Upper triangle - hide
                ax.set_visible(False)
            elif i == j:  # Diagonal - histogram
                ax.hist(projections[cols[i]], bins=30, alpha=0.7, color="steelblue")
                ax.set_ylabel("Count" if j == 0 else "")
            else:  # Lower triangle - scatter
                ax.scatter(projections[cols[j]], projections[cols[i]], alpha=0.3, s=5)
            
            # Labels
            if i == n_cols - 1:
                ax.set_xlabel(DEMOGRAPHIC_DISPLAY_NAMES.get(cols[j], cols[j]), fontsize=9)
            if j == 0:
                ax.set_ylabel(DEMOGRAPHIC_DISPLAY_NAMES.get(cols[i], cols[i]), fontsize=9)
            
            ax.tick_params(labelsize=7)
    
    fig.suptitle("Demographic Score Axes (Embedding Projections)", fontsize=14, y=1.02)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_masking_coverage(
    clean_df: pd.DataFrame,
    logs_df: pd.DataFrame,
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """Plot masking coverage statistics."""
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    # 1. Distribution of masks per post
    if "post_id" in logs_df.columns and len(logs_df) > 0:
        masks_per_post = logs_df.groupby("post_id").size()
        axes[0].hist(masks_per_post, bins=30, alpha=0.7, color="steelblue", edgecolor="black")
        axes[0].axvline(x=masks_per_post.mean(), color="red", linestyle="--", 
                       label=f"Mean: {masks_per_post.mean():.1f}")
        axes[0].set_xlabel("Number of Masks per Post")
        axes[0].set_ylabel("Count")
        axes[0].set_title("Distribution of Mask Counts")
        axes[0].legend()
        axes[0].grid(True, alpha=0.3)
    else:
        axes[0].text(0.5, 0.5, "No masking data available", ha="center", va="center")
        axes[0].set_title("Distribution of Mask Counts")
    
    # 2. Masked vs Unmasked posts
    if "post" in clean_df.columns and "post_masked" in clean_df.columns:
        n_total = len(clean_df)
        n_masked = (clean_df["post"] != clean_df["post_masked"]).sum()
        n_unmasked = n_total - n_masked
        
        colors = sns.color_palette(palette, n_colors=2)
        wedges, texts, autotexts = axes[1].pie(
            [n_masked, n_unmasked],
            labels=["Masked", "Unmasked"],
            autopct="%1.1f%%",
            colors=colors,
            explode=(0.05, 0),
        )
        axes[1].set_title(f"Posts with Masks Applied\n(n={n_total:,})")
    else:
        axes[1].text(0.5, 0.5, "No post data available", ha="center", va="center")
        axes[1].set_title("Posts with Masks Applied")
    
    # 3. Span length distribution
    if "span_start" in logs_df.columns and "span_end" in logs_df.columns and len(logs_df) > 0:
        span_lengths = logs_df["span_end"] - logs_df["span_start"]
        span_lengths = span_lengths[span_lengths > 0]
        
        if len(span_lengths) > 0:
            axes[2].hist(span_lengths, bins=30, alpha=0.7, color="coral", edgecolor="black")
            axes[2].axvline(x=span_lengths.mean(), color="red", linestyle="--",
                           label=f"Mean: {span_lengths.mean():.1f}")
            axes[2].set_xlabel("Span Length (characters)")
            axes[2].set_ylabel("Count")
            axes[2].set_title("Distribution of Detected Span Lengths")
            axes[2].legend()
            axes[2].grid(True, alpha=0.3)
        else:
            axes[2].text(0.5, 0.5, "No span data available", ha="center", va="center")
            axes[2].set_title("Distribution of Detected Span Lengths")
    else:
        axes[2].text(0.5, 0.5, "No span data available", ha="center", va="center")
        axes[2].set_title("Distribution of Detected Span Lengths")
    
    fig.suptitle("Masking Coverage Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


# =============================================================================
# New Visualizations for Enhanced Phase A Metrics
# =============================================================================

def plot_amnesic_drop_with_ci(
    per_column_results: Dict[str, Dict[str, Any]],
    output_path: Path,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """
    Plot amnesic drop bar chart with 95% confidence interval error bars.
    
    Args:
        per_column_results: Dict from compute_per_column_amnesic_drop()
        output_path: Where to save the plot
        dpi: Figure DPI
        palette: Seaborn color palette
    """
    columns = [c for c in per_column_results if "error" not in per_column_results[c]]
    if not columns:
        logger.warning("No valid column results for amnesic drop plot")
        return
    
    fig, ax = plt.subplots(figsize=(12, 6))
    
    x = np.arange(len(columns))
    drops = []
    ci_lowers = []
    ci_uppers = []
    
    for col in columns:
        result = per_column_results[col]
        drop = result.get("amnesic_drop", 0)
        drops.append(drop)
        
        # Extract CI from cross-validation results
        cv_before = result.get("cv_before", {})
        cv_after = result.get("cv_after", {})
        
        if cv_before and cv_after:
            # Approximate CI for the drop
            ci_before = cv_before.get("ci_95_upper", result.get("acc_before", 0.5)) - cv_before.get("ci_95_lower", result.get("acc_before", 0.5))
            ci_after = cv_after.get("ci_95_upper", result.get("acc_after", 0.5)) - cv_after.get("ci_95_lower", result.get("acc_after", 0.5))
            ci_drop = np.sqrt(ci_before**2 + ci_after**2) / 2
        else:
            ci_drop = 0
        
        ci_lowers.append(ci_drop)
        ci_uppers.append(ci_drop)
    
    display_names = [DEMOGRAPHIC_DISPLAY_NAMES.get(c, c) for c in columns]
    colors = sns.color_palette(palette, n_colors=len(columns))
    
    bars = ax.bar(x, drops, color=colors, edgecolor="black", linewidth=0.5)
    ax.errorbar(x, drops, yerr=[ci_lowers, ci_uppers], fmt="none", color="black", capsize=5)
    
    # Threshold line
    ax.axhline(y=0.3, color="red", linestyle="--", alpha=0.7, label="Target threshold (30%)")
    
    ax.set_xticks(x)
    ax.set_xticklabels(display_names, rotation=45, ha="right")
    ax.set_ylabel("Amnesic Drop")
    ax.set_xlabel("Demographic Attribute")
    ax.set_title("Amnesic Drop by Attribute with 95% Confidence Intervals")
    ax.legend()
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar, drop in zip(bars, drops):
        height = bar.get_height()
        ax.annotate(f"{drop:.1%}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=9)
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_embedding_separability(
    separability_before: Dict[str, Any],
    separability_after: Dict[str, Any],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Plot embedding separability comparison (silhouette score) before/after LEACE.
    
    Args:
        separability_before: Dict from compute_embedding_separability() before LEACE
        separability_after: Dict from compute_embedding_separability() after LEACE
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, axes = plt.subplots(1, 2, figsize=(12, 5))
    
    # Silhouette comparison
    sil_before = separability_before.get("silhouette_score", 0)
    sil_after = separability_after.get("silhouette_score", 0)
    
    bars = axes[0].bar(["Before LEACE", "After LEACE"], [sil_before, sil_after],
                       color=["steelblue", "coral"], edgecolor="black")
    axes[0].set_ylabel("Silhouette Score")
    axes[0].set_title("Embedding Separability (Silhouette Score)")
    axes[0].set_ylim(-1, 1)
    axes[0].axhline(y=0, color="gray", linestyle="--", alpha=0.5)
    axes[0].grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[0].annotate(f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3 if height >= 0 else -12),
            textcoords="offset points",
            ha="center", va="bottom" if height >= 0 else "top", fontsize=11)
    
    # Davies-Bouldin comparison (lower is better)
    db_before = separability_before.get("davies_bouldin_index", 0)
    db_after = separability_after.get("davies_bouldin_index", 0)
    
    bars = axes[1].bar(["Before LEACE", "After LEACE"], [db_before, db_after],
                       color=["steelblue", "coral"], edgecolor="black")
    axes[1].set_ylabel("Davies-Bouldin Index")
    axes[1].set_title("Embedding Separability (Davies-Bouldin Index)\n(Lower = better separation)")
    axes[1].grid(True, alpha=0.3, axis="y")
    
    # Add value labels
    for bar in bars:
        height = bar.get_height()
        axes[1].annotate(f"{height:.3f}",
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3), textcoords="offset points",
            ha="center", va="bottom", fontsize=11)
    
    fig.suptitle("Embedding Separability Analysis", fontsize=14)
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_probe_learning_curves(
    learning_curve_before: List[float],
    learning_curve_after: List[float],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Plot training loss curves for torch probes before/after LEACE.
    
    Args:
        learning_curve_before: Loss values per epoch for before-LEACE probe
        learning_curve_after: Loss values per epoch for after-LEACE probe
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    epochs = np.arange(1, len(learning_curve_before) + 1)
    
    ax.plot(epochs, learning_curve_before, "b-", linewidth=2, label="Before LEACE", alpha=0.8)
    ax.plot(epochs, learning_curve_after, "r-", linewidth=2, label="After LEACE", alpha=0.8)
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Training Loss")
    ax.set_title("Probe Training Loss Curves")
    ax.legend()
    ax.grid(True, alpha=0.3)
    
    # Add final loss annotations
    ax.annotate(f"Final: {learning_curve_before[-1]:.4f}",
        xy=(epochs[-1], learning_curve_before[-1]),
        xytext=(5, 5), textcoords="offset points",
        fontsize=9, color="blue")
    ax.annotate(f"Final: {learning_curve_after[-1]:.4f}",
        xy=(epochs[-1], learning_curve_after[-1]),
        xytext=(5, -10), textcoords="offset points",
        fontsize=9, color="red")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_solver_convergence_benchmark(
    benchmark_results: Dict[str, Any],
    learning_curve_torch: List[float],
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Plot PyTorch probe convergence vs exact solver accuracy.
    
    Shows validation accuracy over epochs with exact solver as horizontal baseline.
    
    Args:
        benchmark_results: Dict from benchmark_solver_convergence()
        learning_curve_torch: Validation accuracies per epoch from torch probe
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    exact_acc = benchmark_results.get("exact_accuracy", 0)
    torch_acc = benchmark_results.get("torch_accuracy", 0)
    
    epochs = np.arange(1, len(learning_curve_torch) + 1)
    
    # Plot torch convergence
    ax.plot(epochs, learning_curve_torch, "b-", linewidth=2, label="PyTorch SGD Probe", alpha=0.8)
    
    # Exact solver baseline
    ax.axhline(y=exact_acc, color="red", linestyle="--", linewidth=2, 
               label=f"Exact Solver (sklearn): {exact_acc:.3f}")
    
    ax.set_xlabel("Epoch")
    ax.set_ylabel("Validation Accuracy")
    ax.set_title("Solver Convergence Benchmark: PyTorch SGD vs Exact Solution")
    ax.legend(loc="lower right")
    ax.grid(True, alpha=0.3)
    
    # Annotate delta
    delta = benchmark_results.get("solver_accuracy_delta", 0)
    ax.annotate(f"Δ = {delta:.3f}",
        xy=(epochs[-1], torch_acc),
        xytext=(10, 0), textcoords="offset points",
        fontsize=10, ha="left")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_weight_correlation_scatter(
    weights_exact: np.ndarray,
    weights_torch: np.ndarray,
    output_path: Path,
    dpi: int = 150,
) -> None:
    """
    Scatter plot of exact vs torch probe weights to visualize alignment.
    
    Points on y=x diagonal indicate perfect agreement between solvers.
    
    Args:
        weights_exact: Weight coefficients from exact solver
        weights_torch: Weight coefficients from torch solver
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(8, 8))
    
    # Flatten weights
    w_exact = weights_exact.flatten()
    w_torch = weights_torch.flatten()
    
    # Scatter plot
    ax.scatter(w_exact, w_torch, alpha=0.5, s=10, c="blue")
    
    # Perfect alignment line
    lims = [
        min(w_exact.min(), w_torch.min()),
        max(w_exact.max(), w_torch.max()),
    ]
    ax.plot(lims, lims, "r--", linewidth=2, label="Perfect alignment (y=x)")
    
    # Compute correlation
    correlation = np.corrcoef(w_exact, w_torch)[0, 1]
    
    ax.set_xlabel("Exact Solver Weights")
    ax.set_ylabel("PyTorch Solver Weights")
    ax.set_title(f"Weight Correlation: r = {correlation:.4f}")
    ax.legend()
    ax.grid(True, alpha=0.3)
    ax.set_aspect("equal", adjustable="box")
    
    plt.tight_layout()
    fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
    plt.close(fig)
    logger.info(f"Saved {output_path}")


def plot_stratified_confusion_matrices(
    confusion_before: np.ndarray,
    confusion_after: np.ndarray,
    class_names: Optional[List[str]] = None,
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Plot 2x2 grid of confusion matrices before/after LEACE.
    
    Grid layout:
    - Top-Left: Before (Standard counts)
    - Top-Right: Before (Normalized by true label - shows Recall)
    - Bottom-Left: After (Standard counts)
    - Bottom-Right: After (Normalized by true label)
    
    Args:
        confusion_before: Confusion matrix before LEACE
        confusion_after: Confusion matrix after LEACE
        class_names: Optional class label names
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, axes = plt.subplots(2, 2, figsize=(14, 12))
    
    # Normalize by row (true labels)
    with np.errstate(divide="ignore", invalid="ignore"):
        norm_before = confusion_before / confusion_before.sum(axis=1, keepdims=True)
        norm_before = np.nan_to_num(norm_before)
        norm_after = confusion_after / confusion_after.sum(axis=1, keepdims=True)
        norm_after = np.nan_to_num(norm_after)
    
    # Top-Left: Before (counts)
    sns.heatmap(confusion_before, annot=True, fmt="d", cmap="Blues", ax=axes[0, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[0, 0].set_title("Before LEACE (Counts)")
    axes[0, 0].set_xlabel("Predicted")
    axes[0, 0].set_ylabel("True")
    
    # Top-Right: Before (normalized)
    sns.heatmap(norm_before, annot=True, fmt=".2f", cmap="Blues", ax=axes[0, 1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[0, 1].set_title("Before LEACE (Recall per Class)")
    axes[0, 1].set_xlabel("Predicted")
    axes[0, 1].set_ylabel("True")
    
    # Bottom-Left: After (counts)
    sns.heatmap(confusion_after, annot=True, fmt="d", cmap="Oranges", ax=axes[1, 0],
                xticklabels=class_names, yticklabels=class_names)
    axes[1, 0].set_title("After LEACE (Counts)")
    axes[1, 0].set_xlabel("Predicted")
    axes[1, 0].set_ylabel("True")
    
    # Bottom-Right: After (normalized)
    sns.heatmap(norm_after, annot=True, fmt=".2f", cmap="Oranges", ax=axes[1, 1],
                xticklabels=class_names, yticklabels=class_names, vmin=0, vmax=1)
    axes[1, 1].set_title("After LEACE (Recall per Class)")
    axes[1, 1].set_xlabel("Predicted")
    axes[1, 1].set_ylabel("True")
    
    fig.suptitle("Stratified Confusion Matrices: Before vs After LEACE", fontsize=14)
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path}")
    else:
        plt.show()


def plot_specificity_gap(
    target_results: Dict[str, Any],
    control_results: Dict[str, Any],
    target_name: str = "Target (PII)",
    control_name: str = "Control",
    output_path: Path = None,
    dpi: int = 150,
    palette: str = "husl",
) -> None:
    """
    Plot grouped bar chart showing accuracy drop for target vs control probes.
    
    Visual goal: Large drop for target, minimal drop for control.
    
    Args:
        target_results: Dict with acc_before, acc_after for target attribute
        control_results: Dict with acc_before, acc_after for control attribute
        target_name: Display name for target attribute
        control_name: Display name for control attribute
        output_path: Where to save the plot
        dpi: Figure DPI
        palette: Color palette
    """
    fig, ax = plt.subplots(figsize=(10, 6))
    
    x = np.arange(2)
    width = 0.35
    
    target_before = target_results.get("acc_before", 0.5)
    target_after = target_results.get("acc_after", 0.5)
    control_before = control_results.get("acc_before", 0.5)
    control_after = control_results.get("acc_after", 0.5)
    
    # Before bars
    bars1 = ax.bar(x - width/2, [target_before, control_before], width, 
                   label="Before LEACE", color="steelblue", edgecolor="black")
    # After bars
    bars2 = ax.bar(x + width/2, [target_after, control_after], width,
                   label="After LEACE", color="coral", edgecolor="black")
    
    ax.set_ylabel("Probe Accuracy")
    ax.set_title("Specificity Check: Target vs Control Probe")
    ax.set_xticks(x)
    ax.set_xticklabels([target_name, control_name])
    ax.legend()
    ax.set_ylim(0, 1)
    ax.axhline(y=0.5, color="gray", linestyle="--", alpha=0.5, label="Random baseline")
    ax.grid(True, alpha=0.3, axis="y")
    
    # Add drop annotations
    target_drop = target_before - target_after
    control_drop = control_before - control_after
    
    ax.annotate(f"Drop: {target_drop:.1%}", xy=(0, max(target_before, target_after) + 0.05),
                ha="center", fontsize=10, color="darkblue")
    ax.annotate(f"Drop: {control_drop:.1%}", xy=(1, max(control_before, control_after) + 0.05),
                ha="center", fontsize=10, color="darkred")
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path}")
    else:
        plt.show()


def plot_selectivity_frontier(
    strategy_results: List[Dict[str, Any]],
    strategy_names: List[str],
    output_path: Path = None,
    dpi: int = 150,
) -> None:
    """
    Scatter plot of control task performance vs target task amnesia.
    
    Each point represents a different masking strategy or LEACE configuration.
    Optimal strategies are in the top-right corner (high utility, high amnesia).
    
    Args:
        strategy_results: List of dicts with 'control_accuracy' and 'target_amnesia'
        strategy_names: Names for each strategy
        output_path: Where to save the plot
        dpi: Figure DPI
    """
    fig, ax = plt.subplots(figsize=(10, 8))
    
    control_accs = [r.get("control_accuracy", 0.5) for r in strategy_results]
    target_amnesias = [r.get("target_amnesia", 0) for r in strategy_results]
    
    scatter = ax.scatter(control_accs, target_amnesias, s=100, c=np.arange(len(strategy_results)),
                         cmap="viridis", edgecolor="black", linewidth=0.5)
    
    # Label points
    for i, name in enumerate(strategy_names):
        ax.annotate(name, (control_accs[i], target_amnesias[i]),
                    xytext=(5, 5), textcoords="offset points", fontsize=9)
    
    ax.set_xlabel("Control Task Performance (General Utility)")
    ax.set_ylabel("Target Task Amnesia (Privacy/Safety)")
    ax.set_title("Selectivity Frontier: Utility vs Privacy Trade-off")
    ax.grid(True, alpha=0.3)
    
    # Mark ideal region
    ax.axhline(y=0.3, color="green", linestyle="--", alpha=0.5, label="Target amnesia threshold (0.3)")
    ax.axvline(x=0.7, color="blue", linestyle="--", alpha=0.5, label="Utility threshold (0.7)")
    
    # Shade optimal quadrant
    ax.fill_between([0.7, 1.0], 0.3, 1.0, alpha=0.1, color="green", label="Optimal region")
    
    ax.legend(loc="lower left")
    ax.set_xlim(0, 1)
    ax.set_ylim(0, 1)
    
    plt.tight_layout()
    
    if output_path:
        fig.savefig(output_path, dpi=dpi, bbox_inches="tight")
        plt.close(fig)
        logger.info(f"Saved {output_path}")
    else:
        plt.show()

