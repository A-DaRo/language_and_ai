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
pip install -e ".[hpc]" --extra-index-url=https://pypi.nvidia.com
```

To install all (BEWARE cuml is Linux only)
```bash
pip install -e ".[all]" --extra-index-url=https://pypi.nvidia.com
```

**Remark**
FP8 precision requires `TransformerEngine`, to install:
```bash
pip install --no-build-isolation transformer_engine[pytorch]
```

## To Run

1. Make sure to have all SOBR csv files in `datasets/`

2. Create unified arrow dataset

```bash
# Convert raw data to Arrow format
python scripts/convert_pandas_to_arrow.py
```

3. (Optional)Create laptop partition

```bash
# Create laptop arrow partition from unified table
python scripts/create_laptop_dataset.py
```
4. Run full pipeline

```bash
# Run the full pipeline
python -m neuro_stylometry run-full-pipeline \
    --dataset artifacts/data/sobr.arrow \  # or sobr_laptop.arrow
    --output-dir artifacts/full_pipeline_output \
    --mode laptop \                          # or hpc
    --use-only nationality
```
## How to use

### The `convert_pandas_to_arrow` command

The `convert_pandas_to_arrow` command converts the raw SOBR demographic CSV files into a unified Arrow dataset. It validates the input files, processes the data, and outputs a single Arrow file for use in the pipeline.

#### Example Command

```bash
python scripts/convert_pandas_to_arrow.py \
    --raw-data-dir datasets/ \
    --output artifacts/data/sobr.arrow \
    --verbose
```

#### CLI Parameters

| Parameter         | Type                | Default Value                     | Description                                                                 |
|-------------------|---------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `--raw-data-dir`  | `Path` (optional)  | `datasets/`                       | Directory containing the 8 demographic CSV files.                          |
| `--output`        | `Path` (optional)  | `artifacts/data/sobr.arrow`       | Output path for unified Arrow file.                                        |
| `--verbose`       | `Flag`             | `False`                           | Enable verbose logging (DEBUG level).                                      |

#### Note

Ensure that the `--raw-data-dir` contains all 8 required demographic CSV files before running this command. Missing or invalid files will cause the conversion process to fail.

---

### The `run-full-pipeline` command

The `run-full-pipeline` command is the primary entry point for running the entire pipeline. It executes all phases (Phase A and Phase D) in sequence, including optional verification and report generation steps. This command is designed to simplify the pipeline execution process.

#### Example Command

```bash
python -m neuro_stylometry run-full-pipeline \
    --dataset datasets/nationality.arrow \
    --output-dir artifacts/full_pipeline_output \
    --mode laptop \
    --use-only nationality
```

#### CLI Parameters

| Parameter         | Type                | Default Value | Description                                                                 |
|-------------------|---------------------|---------------|-----------------------------------------------------------------------------|
| `--dataset`       | `Path` (required)  | None          | Path to the input dataset Arrow file.                                      |
| `--output-dir`    | `Path` (required)  | None          | Output directory for all pipeline artifacts.                               |
| `--mode`          | `Choice`           | `auto`        | Hardware mode: `auto`, `laptop`, or `hpc`.                                 |
| `--config`        | `Path` (optional)  | None          | Optional experiment YAML config to merge on top of base+mode configs.      |
| `--use-only`      | `str` (multiple)   | None          | Filter to specified demographic labels (e.g., `--use-only gender`).        |
| `--skip-phase-a`  | `Flag`             | `False`       | Skip Phase A (requires existing Phase A artifacts in `output-dir/phase_a`).|
| `--skip-phase-d`  | `Flag`             | `False`       | Skip Phase D (only run Phase A).                                           |
| `--skip-verify`   | `Flag`             | `False`       | Skip Phase D verification (CHG + SVS).                                     |
| `--skip-report`   | `Flag`             | `False`       | Skip Phase D report generation.                                            |
| `--dry-run`       | `Flag`             | `False`       | Print resolved config and execution plan without running.                  |

#### Note

Using a laptop-sized partition sample (e.g., created using the `create_laptop_dataset.py` script) may lead to statistically irrelevant findings. This can cause the pipeline to fail due to insufficient data for meaningful analysis. It is recommended to use the full dataset for robust results.

---

### The `create_laptop_dataset` command

The `create_laptop_dataset` command generates a lightweight, representative subset of the full SOBR Arrow dataset. This is particularly useful for development and debugging on systems with limited resources, as it allows for rapid iteration while preserving the structural fidelity of the dataset.

#### Example Command

```bash
python scripts/create_laptop_dataset.py \
    --input artifacts/data/sobr.arrow \
    --output artifacts/data/sobr_laptop.arrow \
    --target-size 5000 \
    --seed 123
```

#### CLI Parameters

| Parameter         | Type                | Default Value                     | Description                                                                 |
|-------------------|---------------------|-----------------------------------|-----------------------------------------------------------------------------|
| `--input`         | `Path` (optional)  | `artifacts/data/sobr.arrow`       | Path to the full SOBR Arrow file.                                          |
| `--output`        | `Path` (optional)  | `artifacts/data/sobr_laptop.arrow`| Output path for the laptop subset.                                         |
| `--target-size`   | `int` (optional)   | `10000`                           | Approximate number of posts in the subset.                                 |
| `--seed`          | `int` (optional)   | `42`                              | Random seed for reproducibility.                                           |
| `--verbose`       | `Flag`             | `False`                           | Enable verbose logging.                                                    |

#### Note

The laptop partition is an optional feature that allows you to create a smaller, representative subset of the full dataset for development and debugging purposes. This is particularly useful for rapid iteration on systems with limited resources. The size of the subset can be controlled using the `--target-size` CLI flag. For example:

```bash
python scripts/create_laptop_dataset.py \
    --input artifacts/data/sobr.arrow \
    --output artifacts/data/sobr_laptop.arrow \
    --target-size 5000
```

By default, the target size is set to 10,000 posts. However, this is not required for running the full pipeline. The full dataset can be used directly without creating a laptop partition.

**Important:** While the laptop partition is useful for development, it may lead to statistically irrelevant findings due to the reduced dataset size. For production-level training, we recommend using the full dataset. The pipeline has been tested and trained on a VM with an NVIDIA RTX 5090 GPU using the full dataset for optimal results.

---

## Advanced Commands

For advanced users, the pipeline can also be executed in separate phases using the following commands:

### The `run-phase-a` command

This command runs only Phase A of the pipeline, which performs pollution detection and mitigation.

#### Example Command

```bash
python -m neuro_stylometry run-phase-a \
    --dataset datasets/sobr.arrow \
    --output-dir artifacts/phase_a_output \
    --mode laptop
```

#### Purpose

- Detects pollution spans using GLiNER.
- Applies LEACE projection to remove pollution information.
- Outputs artifacts such as `clean_dataset.arrow`, `projection_matrix.pt`, and `pollution_logs.arrow`.

### The `run-phase-d` command

This command runs only Phase D of the pipeline, which trains a constrained Transformer model.

#### Example Command

```bash
python -m neuro_stylometry run-phase-d \
    --dataset artifacts/phase_d/tokenized_dataset.arrow \
    --output-dir artifacts/phase_d_test \
    --artifacts-dir artifacts/phase_a \
    --mode hpc \
    --config conf/experiments/quick_test.yaml
```

#### Purpose

- Trains a baseline Transformer model on unmasked data.
- Trains a constrained Transformer model using the LEACE projection matrix.
- Outputs comparative metrics and trained model checkpoints.

---

## License

MIT License
