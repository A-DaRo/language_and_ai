"""Plotting utilities for Phase D evaluation artifacts."""

from __future__ import annotations

import json
from pathlib import Path
from typing import Dict, Iterable, List, Optional, Tuple

import matplotlib.pyplot as plt


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
