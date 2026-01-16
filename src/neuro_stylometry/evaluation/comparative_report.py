"""HTML report generator for Phase D baseline vs constrained runs."""

from __future__ import annotations

import json
import logging
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from .attention_analysis import load_svs_result, summarize_head_classification
from .visualizations import (
    plot_class_distribution,
    plot_comparison_bars,
    plot_confusion_delta,
    plot_confusion_matrix,
    plot_head_classification_comparison,
    plot_loss_comparison,
    plot_loss_curve,
    plot_per_class_f1_bars,
    plot_performance_vs_support,
    plot_precision_recall_scatter,
    plot_task_bars,
    plot_topn_confusion_heatmap,
)

logger = logging.getLogger(__name__)


def _safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _detect_tasks_from_metadata(metadata: Dict) -> Tuple[List[str], Dict[str, Dict[str, int]]]:
    """Extract task list and label maps from training_metadata.json."""
    tasks = []
    label_maps = {}

    # Check for use_only in label_filter
    label_filter = metadata.get("label_filter", {})
    use_only = label_filter.get("use_only", [])
    if use_only:
        tasks = list(use_only)

    # Extract label_maps if available
    label_maps = metadata.get("label_maps", {})

    # If no use_only, try to infer from label_maps
    if not tasks and label_maps:
        tasks = list(label_maps.keys())

    return tasks, label_maps


def _detect_tasks_from_metrics(metrics: Dict) -> List[str]:
    """Extract task list from metrics file structure."""
    return list(metrics.keys())


def _metric_table_rows(
    baseline: Dict[str, Dict[str, float]],
    constrained: Dict[str, Dict[str, float]],
    tasks: Optional[List[str]] = None,
) -> str:
    """Generate HTML table rows for metrics comparison."""
    if tasks is None:
        tasks = sorted(set(baseline.keys()) | set(constrained.keys()))

    rows = []
    for task in tasks:
        base_acc = baseline.get(task, {}).get("accuracy", 0.0)
        cons_acc = constrained.get(task, {}).get("accuracy", 0.0)
        base_f1 = baseline.get(task, {}).get("f1_macro", 0.0)
        cons_f1 = constrained.get(task, {}).get("f1_macro", 0.0)
        delta_acc = cons_acc - base_acc
        delta_f1 = cons_f1 - base_f1

        # Color code deltas
        delta_acc_class = "positive" if delta_acc >= 0 else "negative"
        delta_f1_class = "positive" if delta_f1 >= 0 else "negative"

        rows.append(
            "<tr>"
            f"<td>{task}</td>"
            f"<td>{base_acc:.4f}</td>"
            f"<td>{cons_acc:.4f}</td>"
            f"<td class='{delta_acc_class}'>{delta_acc:+.4f}</td>"
            f"<td>{base_f1:.4f}</td>"
            f"<td>{cons_f1:.4f}</td>"
            f"<td class='{delta_f1_class}'>{delta_f1:+.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else "<tr><td colspan='7'>No metrics found.</td></tr>"


def _per_class_table(
    task: str,
    details: Dict[str, Any],
    label_map: Optional[Dict[str, int]],
    *,
    title: str,
    max_rows: int = 20,
) -> str:
    """Generate HTML table for per-class metrics with proper class names."""
    per_class = details.get("per_class", {})
    if not per_class:
        return f"<p>No per-class stats for {title} ({task}).</p>"

    # Invert label_map for class names
    idx_to_name: Dict[int, str] = {}
    if label_map:
        idx_to_name = {v: k for k, v in label_map.items()}

    # Build rows with class names
    class_data = []
    for cls_idx_str, stats in per_class.items():
        try:
            cls_idx = int(cls_idx_str)
        except ValueError:
            cls_idx = -1
        class_name = idx_to_name.get(cls_idx, f"Class {cls_idx_str}")
        support = stats.get("support", 0)
        if support > 0:  # Only show classes with data
            class_data.append({
                "name": class_name,
                "precision": stats.get("precision", 0.0),
                "recall": stats.get("recall", 0.0),
                "f1": stats.get("f1", 0.0),
                "support": support,
            })

    if not class_data:
        return f"<p>No classes with data for {title} ({task}).</p>"

    # Sort by F1 descending and limit rows
    class_data.sort(key=lambda x: x["f1"], reverse=True)
    total_classes = len(class_data)
    if len(class_data) > max_rows:
        class_data = class_data[:max_rows]

    rows = []
    for cd in class_data:
        rows.append(
            "<tr>"
            f"<td>{cd['name']}</td>"
            f"<td>{cd['precision']:.4f}</td>"
            f"<td>{cd['recall']:.4f}</td>"
            f"<td>{cd['f1']:.4f}</td>"
            f"<td>{cd['support']}</td>"
            "</tr>"
        )

    truncation_note = ""
    if total_classes > max_rows:
        truncation_note = f"<p><em>Showing top {max_rows} of {total_classes} classes (sorted by F1).</em></p>"

    return (
        f"<h3>{title}: {task}</h3>"
        f"{truncation_note}"
        "<table>"
        "<thead><tr><th>Class</th><th>Precision</th><th>Recall</th><th>F1</th><th>Support</th></tr></thead>"
        "<tbody>"
        + "\n".join(rows)
        + "</tbody></table>"
    )


def _executive_summary(
    tasks: List[str],
    baseline_metrics: Dict,
    constrained_metrics: Dict,
    label_maps: Dict[str, Dict[str, int]],
    metadata: Dict,
) -> str:
    """Generate executive summary section."""
    task_count = len(tasks)
    task_names = ", ".join(tasks)

    total_classes = 0
    for task in tasks:
        if task in label_maps:
            total_classes += len(label_maps[task])

    # Calculate overall accuracy delta
    avg_baseline_acc = 0.0
    avg_constrained_acc = 0.0
    for task in tasks:
        avg_baseline_acc += baseline_metrics.get(task, {}).get("accuracy", 0.0)
        avg_constrained_acc += constrained_metrics.get(task, {}).get("accuracy", 0.0)
    if task_count > 0:
        avg_baseline_acc /= task_count
        avg_constrained_acc /= task_count
    delta = avg_constrained_acc - avg_baseline_acc

    epochs = metadata.get("num_epochs", "n/a")
    model_variant = metadata.get("model_variant", "n/a")

    return f"""
    <div class="summary-box">
        <h3>Executive Summary</h3>
        <div class="summary-grid">
            <div class="summary-item">
                <span class="label">Task(s)</span>
                <span class="value">{task_names}</span>
            </div>
            <div class="summary-item">
                <span class="label">Total Classes</span>
                <span class="value">{total_classes}</span>
            </div>
            <div class="summary-item">
                <span class="label">Training Epochs</span>
                <span class="value">{epochs}</span>
            </div>
            <div class="summary-item">
                <span class="label">Model Variant</span>
                <span class="value">{model_variant}</span>
            </div>
            <div class="summary-item">
                <span class="label">Avg Baseline Accuracy</span>
                <span class="value">{avg_baseline_acc:.2%}</span>
            </div>
            <div class="summary-item">
                <span class="label">Avg Constrained Accuracy</span>
                <span class="value">{avg_constrained_acc:.2%}</span>
            </div>
            <div class="summary-item highlight {'negative' if delta < 0 else 'positive'}">
                <span class="label">Accuracy Delta</span>
                <span class="value">{delta:+.2%}</span>
            </div>
        </div>
    </div>
    """


def _verification_interpretation(
    baseline_chg: Dict,
    constrained_chg: Dict,
    svs_delta: Optional[float],
) -> str:
    """Generate human-readable interpretation of verification results."""
    interpretations = []

    # CHG interpretation
    baseline_fac = baseline_chg.get("facilitating", 0)
    constrained_fac = constrained_chg.get("facilitating", 0)
    baseline_irr = baseline_chg.get("irrelevant", 0)
    constrained_irr = constrained_chg.get("irrelevant", 0)

    fac_delta = constrained_fac - baseline_fac
    irr_delta = constrained_irr - baseline_irr

    if fac_delta < 0:
        interpretations.append(
            f"The constrained model has <strong>{abs(fac_delta)} fewer facilitating heads</strong>, "
            "suggesting reduced reliance on demographic-sensitive attention patterns."
        )
    if irr_delta > 0:
        interpretations.append(
            f"The constrained model has <strong>{irr_delta} more irrelevant heads</strong>, "
            "indicating more attention heads have become uninformative for the task."
        )

    # SVS interpretation
    if svs_delta is not None:
        if svs_delta < -0.1:
            interpretations.append(
                f"The SVS delta of <strong>{svs_delta:.3f}</strong> indicates the constrained model "
                "retains significantly less stylometric information in its representations."
            )
        elif svs_delta < 0:
            interpretations.append(
                f"The SVS delta of <strong>{svs_delta:.3f}</strong> shows a modest reduction "
                "in stylometric information retention."
            )

    if not interpretations:
        return "<p>No significant differences detected in verification metrics.</p>"

    return "<ul>" + "".join(f"<li>{i}</li>" for i in interpretations) + "</ul>"


def generate_phase_d_report(
    *,
    phase_d_dir: Path,
    output_dir: Path,
    dataset_path: Optional[Path] = None,
    report_name: str = "phase_d_report.html",
    baseline_subdir: str = "baseline",
    constrained_subdir: str = "constrained",
    chg_subdir: Optional[str] = None,
    use_only: Optional[List[str]] = None,
) -> Path:
    """
    Generate an HTML report summarizing Phase D results.

    Args:
        phase_d_dir: Directory containing Phase D artifacts.
        output_dir: Directory to write report and assets.
        dataset_path: Optional path to dataset for metadata.
        report_name: Name of the HTML report file.
        baseline_subdir: Name of baseline subdirectory.
        constrained_subdir: Name of constrained subdirectory.
        chg_subdir: Name of CHG subdirectory (defaults to "chg" or auto-detected).
        use_only: Explicit list of tasks to report on (auto-detected if None).

    Returns:
        Path to the generated report.
    """
    phase_d_dir = Path(phase_d_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "phase_d_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = phase_d_dir / baseline_subdir
    constrained_dir = phase_d_dir / constrained_subdir

    # Determine CHG directory
    if chg_subdir:
        chg_dir = phase_d_dir / chg_subdir
    else:
        # Try common locations
        for candidate in ["chg", "chg-data"]:
            if (phase_d_dir / candidate).exists():
                chg_dir = phase_d_dir / candidate
                break
        else:
            chg_dir = phase_d_dir / "chg"

    # Load all data
    baseline_metadata = _safe_read_json(baseline_dir / "training_metadata.json") or {}
    constrained_metadata = _safe_read_json(constrained_dir / "training_metadata.json") or {}
    baseline_metrics = _safe_read_json(baseline_dir / "phase_d_metrics.json") or {}
    constrained_metrics = _safe_read_json(constrained_dir / "phase_d_metrics.json") or {}
    baseline_test = _safe_read_json(baseline_dir / "test_metrics.json") or {}
    constrained_test = _safe_read_json(constrained_dir / "test_metrics.json") or {}
    baseline_details = _safe_read_json(baseline_dir / "test_details.json") or {}
    constrained_details = _safe_read_json(constrained_dir / "test_details.json") or {}
    comparative = _safe_read_json(phase_d_dir / "phase_d_comparative_metrics.json") or {}

    # Try CHG directory for comparative metrics if not found
    if not comparative:
        comparative = _safe_read_json(chg_dir / "phase_d_comparative_metrics.json") or {}

    # Detect tasks and label maps
    tasks: List[str] = []
    label_maps: Dict[str, Dict[str, int]] = {}

    if use_only:
        tasks = list(use_only)
    else:
        # Auto-detect from metadata
        detected_tasks, detected_maps = _detect_tasks_from_metadata(baseline_metadata)
        if detected_tasks:
            tasks = detected_tasks
            label_maps = detected_maps
        else:
            # Fallback to metrics
            tasks = _detect_tasks_from_metrics(baseline_test or baseline_metrics)

    # Ensure label_maps has entries for detected tasks
    if not label_maps:
        label_maps = baseline_metadata.get("label_maps", {})
    if not label_maps:
        label_maps = constrained_metadata.get("label_maps", {})

    logger.info(f"Detected tasks: {tasks}")
    logger.info(f"Label maps available for: {list(label_maps.keys())}")

    # Use test metrics if available, otherwise phase_d_metrics
    metrics_baseline = baseline_test if baseline_test else baseline_metrics
    metrics_constrained = constrained_test if constrained_test else constrained_metrics

    # =========================================================================
    # Generate Visualizations
    # =========================================================================

    def _img_block(path: Optional[Path], title: str, css_class: str = "") -> str:
        if not path:
            return ""
        rel_path = path.relative_to(output_dir)
        class_attr = f' class="{css_class}"' if css_class else ""
        return (
            f"<figure{class_attr}><img src='{rel_path.as_posix()}' alt='{title}'/>"
            f"<figcaption>{title}</figcaption></figure>"
        )

    # Loss curves (combined)
    loss_comparison_plot = plot_loss_comparison(
        baseline_dir / "training_log.jsonl",
        constrained_dir / "training_log.jsonl",
        assets_dir / "loss_comparison.png",
        title="Training Loss: Baseline vs Constrained",
    )

    # Individual loss curves (fallback)
    baseline_loss_plot = plot_loss_curve(
        baseline_dir / "training_log.jsonl",
        assets_dir / "baseline_loss.png",
        "Baseline Training Loss",
    )
    constrained_loss_plot = plot_loss_curve(
        constrained_dir / "training_log.jsonl",
        assets_dir / "constrained_loss.png",
        "Constrained Training Loss",
    )

    # Comparison bars
    comparison_acc_plot = plot_comparison_bars(
        metrics_baseline,
        metrics_constrained,
        assets_dir / "accuracy_comparison.png",
        metric_key="accuracy",
        title="Accuracy: Baseline vs Constrained",
    )
    comparison_f1_plot = plot_comparison_bars(
        metrics_baseline,
        metrics_constrained,
        assets_dir / "f1_comparison.png",
        metric_key="f1_macro",
        title="Macro F1: Baseline vs Constrained",
    )

    # Task-specific visualizations
    per_task_plots: Dict[str, Dict[str, Optional[Path]]] = {}

    for task in tasks:
        task_label_map = label_maps.get(task)
        baseline_task_details = baseline_details.get(task, {})
        constrained_task_details = constrained_details.get(task, {})

        task_plots: Dict[str, Optional[Path]] = {}

        # Per-class F1 bars (baseline)
        if baseline_task_details.get("per_class"):
            task_plots["baseline_f1_bars"] = plot_per_class_f1_bars(
                baseline_task_details["per_class"],
                task_label_map,
                assets_dir / f"{task}_baseline_f1_bars.png",
                title=f"Baseline F1 by {task.title()}",
            )

        # Per-class F1 bars (constrained)
        if constrained_task_details.get("per_class"):
            task_plots["constrained_f1_bars"] = plot_per_class_f1_bars(
                constrained_task_details["per_class"],
                task_label_map,
                assets_dir / f"{task}_constrained_f1_bars.png",
                title=f"Constrained F1 by {task.title()}",
            )

        # Top-N confusion heatmaps
        baseline_cm = baseline_task_details.get("confusion_matrix")
        constrained_cm = constrained_task_details.get("confusion_matrix")

        if baseline_cm:
            task_plots["baseline_confusion_topn"] = plot_topn_confusion_heatmap(
                baseline_cm,
                task_label_map,
                assets_dir / f"{task}_baseline_confusion_topn.png",
                top_n=15,
                title=f"Baseline Confusion (Top 15 {task.title()})",
            )

        if constrained_cm:
            task_plots["constrained_confusion_topn"] = plot_topn_confusion_heatmap(
                constrained_cm,
                task_label_map,
                assets_dir / f"{task}_constrained_confusion_topn.png",
                top_n=15,
                title=f"Constrained Confusion (Top 15 {task.title()})",
            )

        # Confusion delta chart
        if baseline_cm and constrained_cm:
            task_plots["confusion_delta"] = plot_confusion_delta(
                baseline_cm,
                constrained_cm,
                task_label_map,
                assets_dir / f"{task}_confusion_delta.png",
                top_n=15,
                title=f"Confusion Changes: {task.title()}",
            )

        # Class distribution (using baseline details)
        if baseline_task_details.get("per_class"):
            task_plots["class_distribution"] = plot_class_distribution(
                baseline_task_details["per_class"],
                task_label_map,
                assets_dir / f"{task}_class_distribution.png",
                title=f"Class Distribution: {task.title()}",
            )

        # Performance vs support scatter
        if baseline_task_details.get("per_class"):
            task_plots["performance_vs_support"] = plot_performance_vs_support(
                baseline_task_details["per_class"],
                task_label_map,
                assets_dir / f"{task}_performance_vs_support.png",
                title=f"F1 vs Sample Count: {task.title()} (Baseline)",
            )

        # Precision-recall scatter
        if baseline_task_details.get("per_class"):
            task_plots["precision_recall"] = plot_precision_recall_scatter(
                baseline_task_details["per_class"],
                task_label_map,
                assets_dir / f"{task}_precision_recall.png",
                title=f"Precision vs Recall: {task.title()} (Baseline)",
            )

        per_task_plots[task] = task_plots

    # CHG/SVS Verification
    baseline_chg = summarize_head_classification(chg_dir / "head_classification_baseline.json")
    constrained_chg = summarize_head_classification(chg_dir / "head_classification_constrained.json")
    baseline_svs = load_svs_result(chg_dir / "svs_baseline.json")
    constrained_svs = load_svs_result(chg_dir / "svs_constrained.json")
    verification_summary = _safe_read_json(chg_dir / "verification_summary.json") or {}

    svs_delta = verification_summary.get("svs_delta")
    if svs_delta is None and baseline_svs and constrained_svs:
        svs_delta = constrained_svs["svs"] - baseline_svs["svs"]

    # Head classification comparison plot
    head_classification_plot = None
    if baseline_chg and constrained_chg:
        baseline_summary = {
            "facilitating": baseline_chg.get("facilitating", 0),
            "irrelevant": baseline_chg.get("irrelevant", 0),
            "neutral": baseline_chg.get("neutral", 0),
        }
        constrained_summary = {
            "facilitating": constrained_chg.get("facilitating", 0),
            "irrelevant": constrained_chg.get("irrelevant", 0),
            "neutral": constrained_chg.get("neutral", 0),
        }
        head_classification_plot = plot_head_classification_comparison(
            baseline_summary,
            constrained_summary,
            assets_dir / "head_classification_comparison.png",
        )

    # =========================================================================
    # Build HTML
    # =========================================================================

    dataset_note = str(dataset_path) if dataset_path else "n/a"

    # Build per-task sections
    task_sections = []
    for task in tasks:
        task_label_map = label_maps.get(task)
        plots = per_task_plots.get(task, {})

        section = f"""
        <div class="task-section">
            <h2>Task: {task.replace("_", " ").title()}</h2>

            <h3>Per-Class Performance</h3>
            <div class="grid">
                {_img_block(plots.get("baseline_f1_bars"), f"Baseline F1 by Class", "wide")}
                {_img_block(plots.get("constrained_f1_bars"), f"Constrained F1 by Class", "wide")}
            </div>

            <h3>Confusion Analysis</h3>
            <div class="grid">
                {_img_block(plots.get("baseline_confusion_topn"), "Baseline Confusion (Top Classes)")}
                {_img_block(plots.get("constrained_confusion_topn"), "Constrained Confusion (Top Classes)")}
            </div>
            {_img_block(plots.get("confusion_delta"), "Confusion Changes (Baseline -> Constrained)", "full-width")}

            <h3>Data Distribution & Analysis</h3>
            <div class="grid">
                {_img_block(plots.get("class_distribution"), "Class Sample Distribution")}
                {_img_block(plots.get("performance_vs_support"), "Performance vs Sample Count")}
                {_img_block(plots.get("precision_recall"), "Precision vs Recall Trade-off")}
            </div>

            <h3>Detailed Per-Class Metrics</h3>
            <details>
                <summary>Baseline Per-Class Metrics (click to expand)</summary>
                {_per_class_table(task, baseline_details.get(task, {}), task_label_map, title="Baseline")}
            </details>
            <details>
                <summary>Constrained Per-Class Metrics (click to expand)</summary>
                {_per_class_table(task, constrained_details.get(task, {}), task_label_map, title="Constrained")}
            </details>
        </div>
        """
        task_sections.append(section)

    # Format SVS values safely
    baseline_svs_str = f"{baseline_svs['svs']:.4f}" if baseline_svs else "n/a"
    constrained_svs_str = f"{constrained_svs['svs']:.4f}" if constrained_svs else "n/a"
    svs_delta_str = f"{svs_delta:.4f}" if svs_delta is not None else "n/a"

    html = f"""<!doctype html>
<html lang="en">
<head>
    <meta charset="utf-8"/>
    <title>Phase D Comparative Report</title>
    <style>
        :root {{
            --primary: #2c3e50;
            --secondary: #3498db;
            --success: #27ae60;
            --danger: #e74c3c;
            --light: #ecf0f1;
            --dark: #2c3e50;
        }}
        body {{
            font-family: "Segoe UI", -apple-system, BlinkMacSystemFont, Roboto, Arial, sans-serif;
            margin: 0;
            padding: 24px;
            color: #333;
            background: #fafafa;
            line-height: 1.6;
        }}
        h1 {{
            color: var(--primary);
            border-bottom: 3px solid var(--secondary);
            padding-bottom: 12px;
            margin-bottom: 24px;
        }}
        h2 {{
            color: var(--primary);
            margin-top: 32px;
            border-left: 4px solid var(--secondary);
            padding-left: 12px;
        }}
        h3 {{
            color: var(--dark);
            margin-top: 24px;
        }}
        .meta {{
            background: var(--light);
            padding: 16px 20px;
            border-radius: 8px;
            margin-bottom: 24px;
            border-left: 4px solid var(--secondary);
        }}
        .summary-box {{
            background: white;
            padding: 20px;
            border-radius: 8px;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
            margin-bottom: 24px;
        }}
        .summary-grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(200px, 1fr));
            gap: 16px;
            margin-top: 16px;
        }}
        .summary-item {{
            background: var(--light);
            padding: 12px 16px;
            border-radius: 6px;
            display: flex;
            flex-direction: column;
        }}
        .summary-item .label {{
            font-size: 0.85em;
            color: #666;
            margin-bottom: 4px;
        }}
        .summary-item .value {{
            font-size: 1.2em;
            font-weight: 600;
            color: var(--primary);
        }}
        .summary-item.highlight {{
            border: 2px solid var(--secondary);
        }}
        .summary-item.highlight.negative .value {{
            color: var(--danger);
        }}
        .summary-item.highlight.positive .value {{
            color: var(--success);
        }}
        table {{
            border-collapse: collapse;
            width: 100%;
            margin: 16px 0;
            background: white;
            border-radius: 8px;
            overflow: hidden;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}
        th, td {{
            border: 1px solid #ddd;
            padding: 10px 12px;
            text-align: left;
        }}
        th {{
            background: var(--primary);
            color: white;
            font-weight: 500;
        }}
        tr:nth-child(even) {{
            background: #f8f9fa;
        }}
        tr:hover {{
            background: #e8f4f8;
        }}
        td.positive {{
            color: var(--success);
            font-weight: 600;
        }}
        td.negative {{
            color: var(--danger);
            font-weight: 600;
        }}
        figure {{
            margin: 12px 0;
            background: white;
            padding: 12px;
            border-radius: 8px;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}
        figure.wide {{
            grid-column: span 2;
        }}
        figure.full-width {{
            grid-column: 1 / -1;
        }}
        img {{
            max-width: 100%;
            height: auto;
            border-radius: 4px;
        }}
        figcaption {{
            text-align: center;
            font-size: 0.9em;
            color: #666;
            margin-top: 8px;
            font-style: italic;
        }}
        .grid {{
            display: grid;
            grid-template-columns: repeat(auto-fit, minmax(400px, 1fr));
            gap: 16px;
            margin: 16px 0;
        }}
        code {{
            background: #f1f3f4;
            padding: 2px 6px;
            border-radius: 4px;
            font-family: "Fira Code", monospace;
            font-size: 0.9em;
        }}
        pre {{
            background: #2d2d2d;
            color: #f8f8f2;
            padding: 16px;
            border-radius: 8px;
            overflow-x: auto;
            font-size: 0.85em;
        }}
        details {{
            background: white;
            border-radius: 8px;
            padding: 12px 16px;
            margin: 12px 0;
            box-shadow: 0 1px 4px rgba(0,0,0,0.1);
        }}
        summary {{
            cursor: pointer;
            font-weight: 600;
            color: var(--primary);
        }}
        summary:hover {{
            color: var(--secondary);
        }}
        .task-section {{
            background: white;
            border-radius: 12px;
            padding: 24px;
            margin: 24px 0;
            box-shadow: 0 2px 8px rgba(0,0,0,0.1);
        }}
        .interpretation {{
            background: #fff8e1;
            border-left: 4px solid #ffc107;
            padding: 16px;
            border-radius: 4px;
            margin: 16px 0;
        }}
        .interpretation ul {{
            margin: 8px 0;
            padding-left: 24px;
        }}
        .interpretation li {{
            margin: 8px 0;
        }}
    </style>
</head>
<body>
    <h1>Phase D Comparative Report</h1>

    <div class="meta">
        <div><strong>Phase D directory:</strong> <code>{phase_d_dir}</code></div>
        <div><strong>Dataset:</strong> <code>{dataset_note}</code></div>
        <div><strong>Tasks analyzed:</strong> <code>{", ".join(tasks)}</code></div>
    </div>

    {_executive_summary(tasks, metrics_baseline, metrics_constrained, label_maps, baseline_metadata)}

    <h2>Training Overview</h2>
    <div class="grid">
        {_img_block(loss_comparison_plot, "Training Loss Comparison", "full-width") if loss_comparison_plot else ""}
        {_img_block(baseline_loss_plot, "Baseline Loss") if not loss_comparison_plot else ""}
        {_img_block(constrained_loss_plot, "Constrained Loss") if not loss_comparison_plot else ""}
    </div>

    <h2>Overall Metrics</h2>
    <table>
        <thead>
            <tr>
                <th>Task</th>
                <th>Baseline Acc</th>
                <th>Constrained Acc</th>
                <th>Acc Delta</th>
                <th>Baseline F1</th>
                <th>Constrained F1</th>
                <th>F1 Delta</th>
            </tr>
        </thead>
        <tbody>
            {_metric_table_rows(metrics_baseline, metrics_constrained, tasks)}
        </tbody>
    </table>

    <div class="grid">
        {_img_block(comparison_acc_plot, "Accuracy Comparison")}
        {_img_block(comparison_f1_plot, "F1 Comparison")}
    </div>

    {"".join(task_sections)}

    <h2>Verification Results (CHG + SVS)</h2>
    <table>
        <thead>
            <tr>
                <th>Model</th>
                <th>Facilitating Heads</th>
                <th>Irrelevant Heads</th>
                <th>Neutral Heads</th>
                <th>SVS Score</th>
            </tr>
        </thead>
        <tbody>
            <tr>
                <td>Baseline</td>
                <td>{baseline_chg.get("facilitating", "n/a")}</td>
                <td>{baseline_chg.get("irrelevant", "n/a")}</td>
                <td>{baseline_chg.get("neutral", "n/a")}</td>
                <td>{baseline_svs_str}</td>
            </tr>
            <tr>
                <td>Constrained</td>
                <td>{constrained_chg.get("facilitating", "n/a")}</td>
                <td>{constrained_chg.get("irrelevant", "n/a")}</td>
                <td>{constrained_chg.get("neutral", "n/a")}</td>
                <td>{constrained_svs_str}</td>
            </tr>
        </tbody>
    </table>
    <p><strong>SVS Delta (constrained - baseline):</strong> {svs_delta_str}</p>

    {_img_block(head_classification_plot, "Attention Head Classification Comparison", "full-width") if head_classification_plot else ""}

    <div class="interpretation">
        <h4>Interpretation</h4>
        {_verification_interpretation(baseline_chg, constrained_chg, svs_delta)}
    </div>

    <h2>Raw Metrics</h2>
    <details>
        <summary>Comparative Metrics JSON (click to expand)</summary>
        <pre>{json.dumps(comparative, indent=2)}</pre>
    </details>

</body>
</html>
"""

    report_path = output_dir / report_name
    report_path.write_text(html, encoding="utf-8")
    logger.info(f"Report generated: {report_path}")
    return report_path
