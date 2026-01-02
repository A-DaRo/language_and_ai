# Code Usage

| Metadata | Value |
| :--- | :--- |
| **Document Type** | Usage |
| **Version** | 1.0 |
| **Last Updated** | 2025-02-14 |
| **Scope** | Interim Assignment Code Utilities |

---

## Index

- Index (this list)
- Executive Summary
- 1. Prerequisites
- 2. Installation
- 3. Ingestion Pipeline Usage
- 4. Outputs and Artifacts
- 5. Notes and Planned Updates

## Executive Summary

This document provides minimal usage instructions for the current code utilities in `Misc/Interim_Assignment/code/`. It focuses on the ingestion pipeline that consolidates the SOBR assignment CSVs into a unified Arrow dataset, and will be extended as additional functionality is introduced.

## 1. Prerequisites

- Python environment with `pip` available.
- Access to the assignment data under `assignment_data/`.

## 2. Installation

```bash
pip install -r Misc/Interim_Assignment/code/requirements.txt
```

## 3. Ingestion Pipeline Usage

**Default execution:**

```bash
python Misc/Interim_Assignment/code/ingestion_pipeline.py \
  --input-dir assignment_data \
  --output-dir artifacts/data \
  --seed 42 \
  --train-ratio 0.8 \
  --val-ratio 0.1
```

**Optional parameters:**

- `--input-dir`: Path to the folder containing the eight CSV files.
- `--output-dir`: Path to the folder where Arrow and metadata artifacts are written.
- `--seed`: Random seed for author-stratified splits.
- `--train-ratio`, `--val-ratio`: Split ratios for train and validation (test is the remainder).

## 4. Outputs and Artifacts

- `artifacts/data/sobr_unified.arrow`: Unified Arrow dataset.
- `artifacts/data/splits.json`: Author-stratified split metadata.
- `artifacts/data/ingestion_report.md`: Summary of ingestion metrics and conflicts.

## 5. Notes and Planned Updates

- This document will be updated as new utilities and processing stages are added.

