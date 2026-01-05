# Testing Guide

This document provides comprehensive guidance for running and extending the test suite for the neuro-stylometry project.

## Table of Contents

1. [Quick Start](#quick-start)
2. [Test Organization](#test-organization)
3. [Markers Reference](#markers-reference)
4. [Hardware Detection](#hardware-detection)
5. [Running Specific Test Categories](#running-specific-test-categories)
6. [CI/CD Integration](#cicd-integration)
7. [Writing New Tests](#writing-new-tests)
8. [Troubleshooting](#troubleshooting)
9. [Known Issues](#known-issues)

---

## Quick Start

```bash
# Install test dependencies
pip install -e ".[dev]"
pip install pytest pytest-xdist pytest-timeout

# Run all tests (auto-skips hardware-mismatched tests)
pytest

# Run fast tests only (excludes slow/real_models)
pytest -m "not slow and not real_models"

# Run with verbose output
pytest -v

# Run in parallel (4 workers)
pytest -n 4
```

## Test Organization

```
tests/
├── conftest.py           # Shared fixtures and pytest hooks
├── fixtures/             # YAML configs for testing
│   └── pipeline_small.yaml
├── unit/                 # Fast, isolated unit tests
│   ├── test_config.py    # Configuration validation
│   ├── test_leace.py     # LEACE mathematical properties
│   ├── test_masker.py    # SpanMasker functionality
│   └── test_dataset.py   # Dataset loading
├── integration/          # Component integration tests
│   ├── test_phase_a_handover.py  # Handover contract validation
│   ├── test_gliner_guard.py      # GLiNER pollution detection
│   ├── test_data_engine.py       # Data pipeline integration
│   ├── test_dual_mode.py         # Laptop/HPC mode switching
│   └── test_leace_strategy.py    # LEACE strategy execution
├── golden/               # Golden reference tests
│   ├── test_projection_consistency.py  # LEACE projection stability
│   └── fixtures/         # Golden reference data
└── benchmarks/           # Performance benchmarks (optional)
    └── test_performance.py  # Memory and throughput tests
```

## Markers Reference

The test suite uses custom pytest markers for filtering and auto-skipping:

| Marker | Description | Example Usage |
|--------|-------------|---------------|
| `@pytest.mark.unit` | Fast, isolated unit tests | `pytest -m unit` |
| `@pytest.mark.integration` | Component integration tests | `pytest -m integration` |
| `@pytest.mark.golden` | Golden reference comparisons | `pytest -m golden` |
| `@pytest.mark.slow` | Tests taking > 10s | `pytest -m "not slow"` |
| `@pytest.mark.real_models` | Loads actual ML models | `pytest -m "not real_models"` |
| `@pytest.mark.cuda` | Requires CUDA GPU | Auto-skipped on CPU |
| `@pytest.mark.cpu` | CPU-only tests | — |
| `@pytest.mark.hpc` | Requires HPC profile | Auto-skipped on laptop |
| `@pytest.mark.laptop` | Laptop-mode tests | Auto-skipped on HPC |
| `@pytest.mark.determinism` | Tests reproducibility | `pytest -m determinism` |
| `@pytest.mark.benchmark` | Performance benchmarks | `pytest -m benchmark` |

### Auto-Skip Behavior

Tests are automatically skipped based on detected hardware:

- `@pytest.mark.cuda` tests skip if `torch.cuda.is_available()` is False
- `@pytest.mark.hpc` tests skip if hardware profile is LAPTOP
- `@pytest.mark.laptop` tests skip if hardware profile is HPC

## Hardware Detection

Hardware is auto-detected using `HardwareDetector`:

```python
from neuro_stylometry.hardware_ops.detection import HardwareDetector, ProfileType

profile = HardwareDetector.detect()
print(profile)  # ProfileType.LAPTOP or ProfileType.HPC
```

**HPC criteria** (any of):
- 32+ CPU cores
- 40+ GB GPU VRAM

### Overriding Hardware Detection

For testing, you can mock hardware profile:

```python
@pytest.fixture
def mock_hardware_profile():
    """Mock hardware detection to return LAPTOP."""
    with patch.object(HardwareDetector, 'detect', return_value=ProfileType.LAPTOP):
        yield
```

Or use the environment variable:

```bash
NEURO_STYLOMETRY_PROFILE=laptop pytest -m laptop
NEURO_STYLOMETRY_PROFILE=hpc pytest -m hpc
```

## Running Specific Test Categories

### By Test Type

```bash
# Unit tests only (fast)
pytest tests/unit/

# Integration tests
pytest tests/integration/

# Golden reference tests
pytest tests/golden/

# Benchmarks (requires pytest-benchmark)
pytest tests/benchmarks/ -m benchmark
```

### By Hardware Requirement

```bash
# CPU-only tests
pytest -m cpu

# CUDA-required tests (skipped if no GPU)
pytest -m cuda

# Laptop-mode tests
pytest -m laptop

# HPC-mode tests
pytest -m hpc
```

### By Speed

```bash
# Fast tests only (good for development)
pytest -m "not slow and not real_models"

# All tests including slow ones
pytest

# Only slow tests (useful for overnight runs)
pytest -m slow
```

### By Feature

```bash
# LEACE-specific tests
pytest -k leace

# GLiNER-specific tests
pytest -k gliner

# Masking tests
pytest -k mask

# Configuration tests
pytest -k config
```

## CI/CD Integration

### GitHub Actions Example

```yaml
# .github/workflows/test.yml
name: Tests

on: [push, pull_request]

jobs:
  unit-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest -m "not slow and not real_models and not cuda" -v

  integration-tests:
    runs-on: ubuntu-latest
    steps:
      - uses: actions/checkout@v4
      - uses: actions/setup-python@v5
        with:
          python-version: '3.10'
      - run: pip install -e ".[dev]"
      - run: pytest tests/integration/ -m "not cuda" -v --timeout=300

  cuda-tests:
    runs-on: [self-hosted, gpu]
    steps:
      - uses: actions/checkout@v4
      - run: pip install -e ".[dev]"
      - run: pytest -m cuda -v
```

### Pre-commit Hook

```bash
# .git/hooks/pre-commit
#!/bin/bash
pytest -m "not slow and not real_models" -q --tb=short
```

### Make/Invoke Tasks

```python
# tasks.py (invoke)
from invoke import task

@task
def test_fast(c):
    """Run fast tests only."""
    c.run("pytest -m 'not slow and not real_models' -q")

@task
def test_all(c):
    """Run all tests."""
    c.run("pytest -v")

@task
def test_coverage(c):
    """Run tests with coverage report."""
    c.run("pytest --cov=src/neuro_stylometry --cov-report=html")
```

## Writing New Tests

### Unit Test Template

```python
# tests/unit/test_my_feature.py
import pytest
from neuro_stylometry.my_module import MyClass

@pytest.mark.unit
class TestMyClass:
    """Unit tests for MyClass."""
    
    def test_basic_functionality(self):
        """Test basic feature works."""
        obj = MyClass()
        result = obj.do_something()
        assert result == expected_value
    
    def test_edge_case(self):
        """Test edge case handling."""
        obj = MyClass()
        with pytest.raises(ValueError, match="expected error"):
            obj.do_something(invalid_input)
    
    @pytest.mark.parametrize("input_val,expected", [
        (1, "one"),
        (2, "two"),
        (3, "three"),
    ])
    def test_parametrized(self, input_val, expected):
        """Test multiple inputs."""
        obj = MyClass()
        assert obj.convert(input_val) == expected
```

### Integration Test Template

```python
# tests/integration/test_my_integration.py
import pytest
from neuro_stylometry.my_module import MyClass
from neuro_stylometry.other_module import OtherClass

@pytest.mark.integration
class TestMyIntegration:
    """Integration tests for MyClass with OtherClass."""
    
    @pytest.fixture
    def setup_components(self):
        """Set up components for integration testing."""
        return MyClass(), OtherClass()
    
    @pytest.mark.slow
    def test_full_pipeline(self, setup_components):
        """Test full data flow through components."""
        my_obj, other_obj = setup_components
        intermediate = my_obj.process(input_data)
        result = other_obj.finalize(intermediate)
        assert result.is_valid()
```

### Hardware-Aware Test Template

```python
# tests/integration/test_hardware_aware.py
import pytest
import torch
from neuro_stylometry.hardware_ops.detection import ProfileType

@pytest.mark.cuda
@pytest.mark.skipif(not torch.cuda.is_available(), reason="CUDA required")
def test_cuda_operation():
    """Test that requires CUDA GPU."""
    tensor = torch.randn(100, 100, device="cuda")
    # ... test GPU operation

@pytest.mark.laptop
def test_laptop_mode(mock_hardware_profile, laptop_strategy_with_config):
    """Test that runs in laptop mode."""
    strategy, config = laptop_strategy_with_config
    assert config["leace"]["force_cpu"] is True
```

### Determinism Test Template

```python
# tests/golden/test_determinism.py
import pytest
from tests.conftest import set_all_seeds

@pytest.mark.determinism
class TestDeterministicOutput:
    """Tests for reproducibility."""
    
    def test_same_seed_same_output(self):
        """Verify same seed produces identical output."""
        set_all_seeds(42)
        result_1 = compute_something()
        
        set_all_seeds(42)
        result_2 = compute_something()
        
        assert torch.allclose(result_1, result_2)
```

## Troubleshooting

### Common Issues

#### Tests hang or timeout

```bash
# Add timeout to prevent hanging
pytest --timeout=60

# Run with verbose to see which test hangs
pytest -v -s
```

#### Out of memory on laptop

```bash
# Run fewer tests in parallel
pytest -n 2

# Skip memory-intensive tests
pytest -m "not slow and not real_models"

# Use laptop subset for data
export NEURO_STYLOMETRY_PROFILE=laptop
```

#### CUDA out of memory

```bash
# Force single GPU
CUDA_VISIBLE_DEVICES=0 pytest -m cuda

# Clear CUDA cache between tests
# (Done automatically by fixtures)
```

#### Import errors

```bash
# Ensure editable install
pip install -e ".[dev]"

# Check for circular imports
python -c "from neuro_stylometry import *"
```

### Debug Mode

```bash
# Drop into debugger on failure
pytest --pdb

# Show print statements
pytest -s

# Show full tracebacks
pytest --tb=long

# Run specific failing test
pytest tests/unit/test_leace.py::TestLEACEMathematicalProperties::test_idempotence -v
```

## Known Issues

### 1. GLiNER Model Loading (~3s per test)

**Issue**: GLiNER model loads slowly, affecting test speed.

**Workaround**: Use `@pytest.mark.real_models` and skip in fast runs:
```bash
pytest -m "not real_models"
```

### 2. CUDA Memory Fragmentation

**Issue**: Repeated CUDA operations may fragment GPU memory.

**Workaround**: Fixtures clear CUDA cache:
```python
@pytest.fixture(autouse=True)
def clear_cuda_cache():
    yield
    if torch.cuda.is_available():
        torch.cuda.empty_cache()
```

### 3. Arrow Memory Mapping on Windows

**Issue**: Memory-mapped Arrow files may not release on Windows.

**Workaround**: Use `tmp_path` fixture for temporary files:
```python
def test_arrow_load(tmp_path):
    arrow_file = tmp_path / "test.arrow"
    # ... test
    # File automatically cleaned up
```

### 4. Non-Deterministic GLiNER Spans

**Issue**: GLiNER may produce slightly different spans across runs.

**Workaround**: Use approximate matching in tests:
```python
def test_gliner_detection():
    spans = detector.detect(text)
    # Check entity type, not exact offsets
    assert any(s["label"] == "AGE" for s in spans)
```

### 5. HPC vs Laptop Float Precision

**Issue**: Float64 on laptop vs float32 on HPC may differ slightly.

**Workaround**: Use correlation checks, not exact equality:
```python
def test_cross_mode():
    correlation = compute_correlation(laptop_P, hpc_P)
    assert correlation > 0.98  # Not exact match
```

---

## Contributing

When adding new tests:

1. Add appropriate markers (`@pytest.mark.unit`, etc.)
2. Use fixtures from `conftest.py` for hardware detection
3. Document any new fixtures or utilities
4. Ensure tests pass on both laptop and HPC modes
5. Keep unit tests fast (< 1s each)
6. Use `@pytest.mark.slow` for longer tests

Questions? Open an issue or contact the maintainers.
