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


@pytest.fixture
def laptop_dataset_path() -> Path:
    path = Path("artifacts/data/sobr_laptop.arrow")
    if not path.exists():
        pytest.skip(f"Laptop dataset not found at {path}")
    return path


def test_phase_a_end_to_end_small_subset(laptop_dataset_path: Path, tmp_path: Path):
    strategy = LaptopFilterStrategy()

    config = {
        "seed": 42,
        "max_samples": 10,
        "gliner_threshold": 0.85,
        "embedder_model": "roberta-base",
        "leace_regularization": 1e-5,
        "leace_batch_size": 10,
        "projection_label": "nationality",
    }

    pipeline = PhaseAPipeline(strategy=strategy, config=config)
    artifacts = pipeline.run(input_dataset_path=laptop_dataset_path, output_dir=tmp_path)

    assert artifacts.clean_dataset_path.exists(), "clean_dataset.arrow not created"
    assert artifacts.projection_matrix_path.exists(), "projection_matrix.pt not created"

    P = torch.load(artifacts.projection_matrix_path, map_location="cpu")

    assert isinstance(P, torch.Tensor)
    assert P.shape == (768, 768)
