"""HTML report generator for Phase D baseline vs constrained runs."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Optional

from .attention_analysis import load_svs_result, summarize_head_classification
from .visualizations import (
    plot_comparison_bars,
    plot_loss_curve,
    plot_task_bars,
)


def _safe_read_json(path: Path) -> Optional[Dict]:
    try:
        with path.open("r", encoding="utf-8") as handle:
            return json.load(handle)
    except FileNotFoundError:
        return None


def _metric_table_rows(
    baseline: Dict[str, Dict[str, float]],
    constrained: Dict[str, Dict[str, float]],
) -> str:
    tasks = sorted(set(baseline.keys()) | set(constrained.keys()))
    rows = []
    for task in tasks:
        base_acc = baseline.get(task, {}).get("accuracy", 0.0)
        cons_acc = constrained.get(task, {}).get("accuracy", 0.0)
        base_f1 = baseline.get(task, {}).get("f1_macro", 0.0)
        cons_f1 = constrained.get(task, {}).get("f1_macro", 0.0)
        rows.append(
            "<tr>"
            f"<td>{task}</td>"
            f"<td>{base_acc:.4f}</td>"
            f"<td>{cons_acc:.4f}</td>"
            f"<td>{cons_acc - base_acc:+.4f}</td>"
            f"<td>{base_f1:.4f}</td>"
            f"<td>{cons_f1:.4f}</td>"
            "</tr>"
        )
    return "\n".join(rows) if rows else "<tr><td colspan='6'>No metrics found.</td></tr>"


def generate_phase_d_report(
    *,
    phase_d_dir: Path,
    output_dir: Path,
    dataset_path: Optional[Path] = None,
    report_name: str = "phase_d_report.html",
) -> Path:
    """Generate an HTML report summarizing Phase D results."""
    phase_d_dir = Path(phase_d_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    assets_dir = output_dir / "phase_d_assets"
    assets_dir.mkdir(parents=True, exist_ok=True)

    baseline_dir = phase_d_dir / "baseline"
    constrained_dir = phase_d_dir / "constrained"

    baseline_metrics = _safe_read_json(baseline_dir / "phase_d_metrics.json") or {}
    constrained_metrics = _safe_read_json(constrained_dir / "phase_d_metrics.json") or {}
    baseline_test = _safe_read_json(baseline_dir / "test_metrics.json") or {}
    constrained_test = _safe_read_json(constrained_dir / "test_metrics.json") or {}

    metrics_for_table = (baseline_test or baseline_metrics, constrained_test or constrained_metrics)

    comparative = _safe_read_json(phase_d_dir / "phase_d_comparative_metrics.json") or {}

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
    comparison_acc_plot = plot_comparison_bars(
        metrics_for_table[0],
        metrics_for_table[1],
        assets_dir / "accuracy_comparison.png",
        metric_key="accuracy",
        title="Accuracy: Baseline vs Constrained",
    )
    comparison_f1_plot = plot_comparison_bars(
        metrics_for_table[0],
        metrics_for_table[1],
        assets_dir / "f1_comparison.png",
        metric_key="f1_macro",
        title="Macro F1: Baseline vs Constrained",
    )

    baseline_acc_plot = plot_task_bars(
        metrics_for_table[0],
        assets_dir / "baseline_accuracy.png",
        metric_key="accuracy",
        title="Baseline Accuracy by Task",
    )
    constrained_acc_plot = plot_task_bars(
        metrics_for_table[1],
        assets_dir / "constrained_accuracy.png",
        metric_key="accuracy",
        title="Constrained Accuracy by Task",
    )

    chg_dir = phase_d_dir / "chg"
    baseline_chg = summarize_head_classification(chg_dir / "head_classification_baseline.json")
    constrained_chg = summarize_head_classification(chg_dir / "head_classification_constrained.json")
    baseline_svs = load_svs_result(chg_dir / "svs_baseline.json")
    constrained_svs = load_svs_result(chg_dir / "svs_constrained.json")

    report_path = output_dir / report_name
    dataset_note = str(dataset_path) if dataset_path else "n/a"

    def _img_block(path: Optional[Path], title: str) -> str:
        if not path:
            return ""
        rel_path = path.relative_to(output_dir)
        return (
            f"<figure><img src='{rel_path.as_posix()}' alt='{title}'/>"
            f"<figcaption>{title}</figcaption></figure>"
        )

    html = f"""<!doctype html>
<html lang="en">
<head>
  <meta charset="utf-8"/>
  <title>Phase D Report</title>
  <style>
    body {{ font-family: "Segoe UI", Arial, sans-serif; margin: 24px; color: #1c1c1c; }}
    h1, h2 {{ color: #2d2d2d; }}
    .meta {{ background: #f5f5f5; padding: 12px 16px; border-radius: 6px; }}
    table {{ border-collapse: collapse; width: 100%; margin-top: 12px; }}
    th, td {{ border: 1px solid #ddd; padding: 6px 8px; text-align: left; }}
    th {{ background: #f0f0f0; }}
    figure {{ margin: 12px 0; }}
    img {{ max-width: 100%; height: auto; border: 1px solid #ddd; }}
    .grid {{ display: grid; grid-template-columns: repeat(auto-fit, minmax(280px, 1fr)); gap: 12px; }}
    code {{ background: #f5f5f5; padding: 1px 4px; border-radius: 4px; }}
  </style>
</head>
<body>
  <h1>Phase D Comparative Report</h1>
  <div class="meta">
    <div><strong>Phase D directory:</strong> <code>{phase_d_dir}</code></div>
    <div><strong>Dataset:</strong> <code>{dataset_note}</code></div>
  </div>

  <h2>Metrics Summary</h2>
  <table>
    <thead>
      <tr>
        <th>Task</th>
        <th>Baseline Acc</th>
        <th>Constrained Acc</th>
        <th>Delta</th>
        <th>Baseline F1</th>
        <th>Constrained F1</th>
      </tr>
    </thead>
    <tbody>
      {_metric_table_rows(metrics_for_table[0], metrics_for_table[1])}
    </tbody>
  </table>

  <h2>Training Curves</h2>
  <div class="grid">
    {_img_block(baseline_loss_plot, "Baseline Loss")}
    {_img_block(constrained_loss_plot, "Constrained Loss")}
  </div>

  <h2>Accuracy & F1 Comparisons</h2>
  <div class="grid">
    {_img_block(comparison_acc_plot, "Accuracy Comparison")}
    {_img_block(comparison_f1_plot, "Macro F1 Comparison")}
    {_img_block(baseline_acc_plot, "Baseline Accuracy by Task")}
    {_img_block(constrained_acc_plot, "Constrained Accuracy by Task")}
  </div>

  <h2>Verification Snapshot (CHG + SVS)</h2>
  <table>
    <thead>
      <tr>
        <th>Run</th>
        <th>Facilitating Heads</th>
        <th>Irrelevant Heads</th>
        <th>Neutral Heads</th>
        <th>SVS</th>
      </tr>
    </thead>
    <tbody>
      <tr>
        <td>Baseline</td>
        <td>{baseline_chg.get("facilitating", 0)}</td>
        <td>{baseline_chg.get("irrelevant", 0)}</td>
        <td>{baseline_chg.get("neutral", 0)}</td>
        <td>{(baseline_svs or {}).get("svs", 0.0):.4f}</td>
      </tr>
      <tr>
        <td>Constrained</td>
        <td>{constrained_chg.get("facilitating", 0)}</td>
        <td>{constrained_chg.get("irrelevant", 0)}</td>
        <td>{constrained_chg.get("neutral", 0)}</td>
        <td>{(constrained_svs or {}).get("svs", 0.0):.4f}</td>
      </tr>
    </tbody>
  </table>

  <h2>Raw Comparative Metrics</h2>
  <pre>{json.dumps(comparative, indent=2)}</pre>
</body>
</html>
"""

    report_path.write_text(html, encoding="utf-8")
    return report_path
