# Copilot instructions (neuro-stylometry)

## Big picture / architecture
- Core package is `src/neuro_stylometry/` (installed as `neuro-stylometry`; entry point in `src/neuro_stylometry/__main__.py`).
- Data flow is **Filter → Project → Verify**:
  - **Ingest** CSVs → unified Arrow: `src/neuro_stylometry/data_engine/converter.py` (schema in `src/neuro_stylometry/data_engine/schemas.py`).
  - **Phase A (pollution guard)**: detect spans with GLiNER → typed masking → compute LEACE projection `P` → write artifacts.
    - Orchestrator: `src/neuro_stylometry/phase_a_pipeline.py`.
    - Dual-mode strategies: `src/neuro_stylometry/pollution_guard/strategies/{laptop,hpc}.py` via `src/neuro_stylometry/factories/strategy_factory.py`.
  - **Handover contract** (Phase A → Phase D) is file-based and validated by `scripts/validate_phase_a_handover.py`.

## Data contracts & artifacts (don’t “rename outputs” casually)
- Unified dataset schema source-of-truth: `src/neuro_stylometry/data_engine/schemas.py` (`SOBR_SCHEMA`, `POLLUTION_LOG_SCHEMA`).
- Expected Phase A outputs (see `PhaseAPipeline.run()` in `src/neuro_stylometry/phase_a_pipeline.py`):
  - `clean_dataset.arrow` containing populated `post_masked`
  - `projection_matrix.pt` (projection matrix; validator checks idempotence)
  - `pollution_logs.arrow` (logged spans; matches `POLLUTION_LOG_SCHEMA`)

## Dual-mode execution conventions
- Hardware auto-detect: `src/neuro_stylometry/hardware_ops/detection.py` (`ProfileType.HPC` vs `ProfileType.LAPTOP`).
- Phase A execution is strategy-based:
  - Laptop: subsets via `max_samples` and batch accumulation (`LEACEComputer.accumulate_batch`).
  - HPC: full-batch GPU LEACE when CUDA is available.
- YAML configs live in `conf/` and layer as base + mode overrides: `conf/base/pipeline.yaml`, `conf/laptop/pipeline.yaml`, `conf/hpc/pipeline.yaml`.
  - Note: some runtime paths currently use `StrategyFactory.get_recommended_config()` (a dict) instead of the YAMLs; keep these consistent if you change one.

## How to run (current repo state)
- Install editable: `pip install -e .`
- Convert CSV → Arrow (expects the 8 CSV filenames and handles the `auhtor_ID` typo):
  - `python scripts/convert_pandas_to_arrow.py --raw-data-dir datasets --output artifacts/data/sobr.arrow`
- Validate Phase A handover artifacts:
  - `python scripts/validate_phase_a_handover.py artifacts/phase_a`
- CLI entry point exists (`neuro-stylometry`), but verify commands before expanding it: `src/neuro_stylometry/__main__.py` currently has partially implemented/possibly inconsistent command wiring.

## Testing & tooling
- Tests are organized as unit/integration/golden under `tests/`.
- Run tests: `pytest -q`
- Lint settings: Ruff configured in `pyproject.toml` (line length 100).

## Repo-specific gotchas
- Arrow loading should stay memory-mapped for scale: `SOBRDataset._load_with_mmap()` in `src/neuro_stylometry/data_engine/dataset.py`.
- Hugging Face `datasets` conversion can break on dictionary columns that become all-null in a split; use `SOBRDataset._sanitize_arrow_table()`.
- Mask tokens are typed and should remain tokenizer-safe; masking logic is in `src/neuro_stylometry/pollution_guard/masker.py` and taxonomy/prompts in `src/neuro_stylometry/pollution_guard/gliner_detector.py`.
