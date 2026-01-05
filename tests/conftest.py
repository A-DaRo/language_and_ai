"""Pytest fixtures shared across the suite.

This project runs heavyweight transformer models in integration tests.
On low-VRAM laptops (e.g., 4GB), memory fragmentation/leakage between tests
can cause spurious CUDA OOMs.
"""

import gc
import os
import random
from typing import Optional

import numpy as np
import pytest
import torch

from neuro_stylometry.hardware_ops.detection import HardwareDetector, HardwareProfile, ProfileType


# ==============================================================================
# Session-scoped Hardware Detection
# ==============================================================================

_cached_hardware_profile: Optional[HardwareProfile] = None


@pytest.fixture(scope="session")
def detected_hardware_profile() -> HardwareProfile:
    """Cache hardware detection result once per session.
    
    Returns:
        HardwareProfile with detected capabilities.
    """
    global _cached_hardware_profile
    if _cached_hardware_profile is None:
        _cached_hardware_profile = HardwareDetector.detect()
    return _cached_hardware_profile


@pytest.fixture
def mock_hardware_profile(monkeypatch):
    """Factory fixture allowing tests to override detected hardware.
    
    Usage:
        def test_something(mock_hardware_profile):
            mock_hardware_profile(ProfileType.LAPTOP, has_cuda=False)
    """
    def _mock(profile_type: ProfileType, has_cuda: bool = True):
        mock_profile = HardwareProfile(
            profile_type=profile_type,
            cpu_cores=8 if profile_type == ProfileType.LAPTOP else 64,
            ram_gb=16.0 if profile_type == ProfileType.LAPTOP else 512.0,
            has_cuda=has_cuda,
            gpu_name="Mock GPU" if has_cuda else None,
            gpu_vram_gb=8.0 if profile_type == ProfileType.LAPTOP and has_cuda else (
                80.0 if has_cuda else None
            ),
            compute_capability=(7, 5) if has_cuda else None,
        )
        monkeypatch.setattr(
            "neuro_stylometry.hardware_ops.detection.HardwareDetector.detect",
            lambda: mock_profile
        )
        return mock_profile
    return _mock


@pytest.fixture
def device_for_hardware(detected_hardware_profile) -> str:
    """Return appropriate device based on detected hardware profile.
    
    Returns:
        "cpu" for laptop mode, "cuda" for HPC (if available).
    """
    if detected_hardware_profile.profile_type == ProfileType.LAPTOP:
        return "cpu"
    return "cuda" if detected_hardware_profile.has_cuda else "cpu"


@pytest.fixture(params=["cpu", "cuda"])
def device(request) -> str:
    """Parametrized device fixture, skips CUDA tests if unavailable.
    
    Yields:
        Device string ("cpu" or "cuda").
    """
    if request.param == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA not available")
    return request.param


# ==============================================================================
# Seed Management for Determinism
# ==============================================================================

@pytest.fixture
def set_all_seeds():
    """Fixture setting NumPy, PyTorch CPU/CUDA, and Python random state.
    
    Usage:
        def test_deterministic(set_all_seeds):
            set_all_seeds(42)
    """
    def _set_seeds(seed: int = 42):
        random.seed(seed)
        np.random.seed(seed)
        torch.manual_seed(seed)
        if torch.cuda.is_available():
            torch.cuda.manual_seed_all(seed)
            # Enable deterministic operations where possible
            torch.backends.cudnn.deterministic = True
            torch.backends.cudnn.benchmark = False
    return _set_seeds


# ==============================================================================
# Strategy Fixtures
# ==============================================================================

@pytest.fixture
def laptop_strategy_with_config():
    """Pre-configured LaptopFilterStrategy with test YAML loaded.
    
    Returns:
        Tuple of (strategy, config_dict).
    """
    from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
    from neuro_stylometry.config import load_pipeline_config
    
    strategy = LaptopFilterStrategy()
    config = load_pipeline_config(mode="laptop")
    return strategy, config


@pytest.fixture
def hpc_strategy_with_config():
    """Pre-configured HPCFilterStrategy with test YAML loaded.
    
    Returns:
        Tuple of (strategy, config_dict).
    """
    from neuro_stylometry.pollution_guard.strategies.hpc import HPCFilterStrategy
    from neuro_stylometry.config import load_pipeline_config
    
    strategy = HPCFilterStrategy()
    config = load_pipeline_config(mode="hpc")
    return strategy, config


# ==============================================================================
# Test Fixtures Directory
# ==============================================================================

@pytest.fixture
def test_fixtures_dir():
    """Return path to test fixtures directory."""
    from pathlib import Path
    return Path(__file__).parent / "fixtures"


@pytest.fixture
def golden_fixtures_dir():
    """Return path to golden test fixtures directory."""
    from pathlib import Path
    return Path(__file__).parent / "golden" / "fixtures"


# ==============================================================================
# Aggressive Resource Cleanup
# ==============================================================================

@pytest.fixture(autouse=True)
def _aggressive_resource_cleanup():
    """Force Python + CUDA cleanup after every test function."""
    yield

    gc.collect()

    try:
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            # Helps release inter-process cached allocations when supported.
            try:
                torch.cuda.ipc_collect()
            except Exception:
                pass
    except Exception:
        # Tests may run without torch (or without CUDA); keep fixture safe.
        pass


# ==============================================================================
# Opt-in Gates for Real Model Tests
# ==============================================================================

@pytest.fixture
def require_real_models():
    """Opt-in gate for tests that must run real transformer/GLiNER models.

    Strict no-mocking tests are gated behind an explicit opt-in env var.
    These tests may download models from Hugging Face if not already present.
    """
    if os.getenv("NEURO_STYLOMETRY_RUN_REAL_MODELS") != "1":
        pytest.skip("Set NEURO_STYLOMETRY_RUN_REAL_MODELS=1 to run real-model tests")


# ==============================================================================
# Pytest Hooks for Auto-Skip Logic
# ==============================================================================

def pytest_runtest_setup(item):
    """Auto-skip tests based on markers and hardware profile."""
    # Check for HPC marker
    if item.get_closest_marker("hpc"):
        profile = HardwareDetector.detect()
        if profile.profile_type != ProfileType.HPC:
            pytest.skip("Test requires HPC hardware (32+ cores or 40GB+ VRAM)")
    
    # Check for CUDA marker
    if item.get_closest_marker("cuda"):
        if not torch.cuda.is_available():
            pytest.skip("Test requires CUDA device")
    
    # Check for real_models marker
    if item.get_closest_marker("real_models"):
        if os.getenv("NEURO_STYLOMETRY_RUN_REAL_MODELS") != "1":
            pytest.skip("Set NEURO_STYLOMETRY_RUN_REAL_MODELS=1 to run real-model tests")


def pytest_configure(config):
    """Register custom markers."""
    config.addinivalue_line("markers", "slow: marks tests as slow (e.g., model downloads)")
    config.addinivalue_line("markers", "real_models: requires NEURO_STYLOMETRY_RUN_REAL_MODELS=1")
    config.addinivalue_line("markers", "cuda: requires CUDA device")
    config.addinivalue_line("markers", "cpu: CPU-only tests")
    config.addinivalue_line("markers", "hpc: requires HPC-grade hardware")
    config.addinivalue_line("markers", "laptop: laptop-compatible tests")
    config.addinivalue_line("markers", "integration: integration tests")
    config.addinivalue_line("markers", "unit: unit tests")
    config.addinivalue_line("markers", "golden: golden reference tests")
    config.addinivalue_line("markers", "determinism: determinism verification tests")
    config.addinivalue_line("markers", "benchmark: performance benchmark tests")
