from pathlib import Path

import pytest
import torch

from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType
from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
from neuro_stylometry.config import load_pipeline_config


def test_hpc_strategy_smoke(require_real_models, tmp_path: Path):
    profile = HardwareDetector.detect()
    if profile.profile_type != ProfileType.HPC:
        pytest.skip("HPC-only smoke test skipped on non-HPC profile")

    dataset_path = Path("artifacts/data/sobr.arrow")
    if not dataset_path.exists():
        pytest.skip(f"Full dataset not found at {dataset_path}")

    strategy = HPCFilterStrategy()
    test_cfg = Path("tests/fixtures/pipeline_small.yaml")
    config = load_pipeline_config(mode="hpc", experiment_config_path=test_cfg)

    out_clean = tmp_path / "clean.arrow"
    out_proj = tmp_path / "projection.pt"
    out_logs = tmp_path / "pollution_logs.arrow"

    metadata = strategy.execute(
        input_dataset_path=dataset_path,
        output_dataset_path=out_clean,
        projection_matrix_path=out_proj,
        pollution_logs_path=out_logs,
        config=config,
    )

    assert out_clean.exists()
    assert out_proj.exists()
    assert out_logs.exists()

    P = torch.load(out_proj, map_location="cpu")
    assert isinstance(P, torch.Tensor)
    assert P.shape == (768, 768)
    err = torch.abs(P @ P - P).max().item()
    assert err < 1e-4
    assert "num_samples" in metadata
