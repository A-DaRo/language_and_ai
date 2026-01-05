"""Integration Test: Phase A End-to-End (Laptop)

Validates the Phase A orchestrator + laptop strategy end-to-end on a small subset
to keep runtime and VRAM usage bounded.

Assertions:
- Produces clean dataset artifact
- Produces projection matrix artifact
- Projection matrix is loadable and has shape (768, 768)
"""

from pathlib import Path

import pytest
import torch

from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
from neuro_stylometry.config import load_pipeline_config
from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType


def test_phase_a_end_to_end_small_subset(require_real_models, tmp_path: Path):
    profile = HardwareDetector.detect()

    test_cfg = Path("tests/fixtures/pipeline_small.yaml")

    if profile.profile_type == ProfileType.HPC:
        dataset_path = Path("artifacts/data/sobr.arrow")
        if not dataset_path.exists():
            pytest.skip(f"Full dataset not found at {dataset_path}")
        strategy = HPCFilterStrategy()
        config = load_pipeline_config(mode="hpc", experiment_config_path=test_cfg)
    else:
        dataset_path = Path("artifacts/data/sobr_laptop.arrow")
        if not dataset_path.exists():
            pytest.skip(f"Laptop dataset not found at {dataset_path}")
        strategy = LaptopFilterStrategy()
        config = load_pipeline_config(mode="laptop", experiment_config_path=test_cfg)

    pipeline = PhaseAPipeline(strategy=strategy, config=config)
    artifacts = pipeline.run(input_dataset_path=dataset_path, output_dir=tmp_path)

    assert artifacts.clean_dataset_path.exists(), "clean_dataset.arrow not created"
    assert artifacts.projection_matrix_path.exists(), "projection_matrix.pt not created"

    P = torch.load(artifacts.projection_matrix_path, map_location="cpu")

    assert isinstance(P, torch.Tensor)
    assert P.shape == (768, 768)
