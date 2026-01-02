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

For HPC features:
```bash
pip install -e ".[hpc]"
```

## Quick Start

```bash
# Convert raw data to Arrow format
neuro-stylometry convert-data --input-dir ./data/raw --output-path ./artifacts/data/sobr_unified.arrow

# Run Phase A (pollution detection and mitigation)
neuro-stylometry run-phase-a --dataset ./artifacts/data/sobr_unified.arrow --output-dir ./artifacts/phase_a

# Run Phase D (constrained training)
neuro-stylometry run-phase-d --dataset ./artifacts/data/sobr_clean.arrow --projection ./artifacts/phase_a/projection_matrix.pt --output-dir ./artifacts/phase_d

# Run verification
neuro-stylometry verify --model-a ./artifacts/phase_d/model_dirty --model-b ./artifacts/phase_d/model_clean --output ./artifacts/reports/comparison_report.md
```

## Documentation

See `Docs_from_SM/Generated/phaseA-D_implementation_plan.md` for full specification.

## License

MIT License
