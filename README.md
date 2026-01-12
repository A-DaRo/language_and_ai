# Neuro-Symbolic Stylometry Pipeline

A dual-mode (Laptop/HPC) implementation of a pollution-aware stylometry pipeline combining symbolic span detection with geometric projection and constrained Transformer training.

## Features

- **Phase A: Pollution Guard** - GLiNER-based span detection + LEACE geometric projection
- **Phase D: Constrained Transformer** - Affine Guard layer for stylometric constraint enforcement
- **Dual-Mode Execution** - Automatic adaptation between laptop debug and HPC production environments
- **Apache Arrow Backend** - Memory-efficient data processing with zero-copy reads

## Installation

```bash
pip install -e .
```

For development:
```bash
pip install -e ".[dev]"
```

For HPC features (cuml):
```bash
pip install ".[hpc]" --extra-index-url=https://pypi.nvidia.com
```

To install all (BEWARE cuml is Linux only)
```bash
pip install ".[all]" --extra-index-url=https://pypi.nvidia.com
```

## To begin

1. Make sure to have all SOBR csv files in `datasets/`

2. Create unified arrow dataset

```bash
# Convert raw data to Arrow format
python scripts/convert_pandas_to_arrow.py
```

3. Create laptop partition

```bash
# Create laptop arrow partition from unified table
python scripts/create_laptop_dataset.py
```

## Documentation

See `Docs_from_SM/Generated/phaseA-D_implementation_plan.md` for full specification.

## Testing

Comprehensive test suite with hardware-aware fixtures and auto-skip for environment-specific tests.

```bash
# Run all tests
pytest

# Run fast tests only (excludes slow/real_models)
pytest -m "not slow and not real_models"

# Run with coverage
pytest --cov=src/neuro_stylometry --cov-report=html
```

See [tests/README.md](tests/README.md) for the complete testing guide.

## License

MIT License
