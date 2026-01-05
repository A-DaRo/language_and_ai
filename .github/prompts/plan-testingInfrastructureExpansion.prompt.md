# Implementation Plan: Comprehensive Testing Infrastructure & Configuration Hardening

**Objective:** Transform Phase A testing suite from ~36% adequacy to 95% production readiness while enforcing hardware duality, CPU-only LEACE for laptop mode, full YAML configuration authority, and improved CLI usability.

**Timeline:** 8-10 weeks (120-150 engineering hours)

**Scope:** Phase A only (Phase D follows as separate initiative)

---

## 1. Pytest Configuration & Hardware-Aware Test Markers

**File:** [pyproject.toml](pyproject.toml) + [tests/conftest.py](tests/conftest.py)

**Deliverables:**

1. Add `[tool.pytest.ini_options]` section to `pyproject.toml`:
   ```toml
   [tool.pytest.ini_options]
   testpaths = ["tests"]
   python_files = ["test_*.py"]
   python_classes = ["Test*"]
   python_functions = ["test_*"]
   
   markers = [
       "slow: marks tests as slow (e.g., model downloads)",
       "real_models: requires NEURO_STYLOMETRY_RUN_REAL_MODELS=1",
       "cuda: requires CUDA device",
       "cpu: CPU-only tests",
       "hpc: requires HPC-grade hardware (32+ cores or 40GB+ VRAM)",
       "laptop: laptop-compatible tests",
       "integration: integration tests",
       "unit: unit tests",
       "golden: golden reference tests",
       "determinism: determinism verification tests",
   ]
   
   addopts = [
       "-ra",
       "--strict-markers",
       "--strict-config",
       "--showlocals",
   ]
   ```

2. Extend `tests/conftest.py` with hardware-aware fixtures:
   - `detected_hardware_profile()`: Caches result of `HardwareDetector.detect()` once per session
   - `mock_hardware_profile(monkeypatch)`: Factory fixture allowing tests to override detected hardware
   - `device_for_hardware(detected_hardware_profile)`: Returns appropriate device ("cpu" for laptop, "cuda" for HPC)
   - `@pytest.fixture(params=["cpu", "cuda"]) def device(request)`: Parametrized device fixture, skips CUDA tests if unavailable
   - `laptop_strategy_with_config()`: Pre-configured `LaptopFilterStrategy` with test YAML loaded
   - `hpc_strategy_with_config()`: Pre-configured `HPCFilterStrategy` with test YAML loaded
   - `set_all_seeds(seed: int)`: Fixture setting NumPy, PyTorch CPU/CUDA, and Python random state

3. Auto-skip logic for markers:
   - Implement pytest hook `pytest_runtest_setup()` that auto-skips tests marked `@pytest.mark.hpc` if `HardwareDetector.detect().profile_type != ProfileType.HPC`
   - Implement pytest hook for `@pytest.mark.cuda` to skip if CUDA unavailable
   - Implement pytest hook for `@pytest.mark.real_models` to skip unless `NEURO_STYLOMETRY_RUN_REAL_MODELS=1`

| **Dependencies:** None | **Risk:** Low

---

## 2. Enforce CPU-Only LEACE Computation in Laptop Mode

**Files:** [conf/laptop/pipeline.yaml](conf/laptop/pipeline.yaml), [src/neuro_stylometry/pollution_guard/strategies/laptop.py](src/neuro_stylometry/pollution_guard/strategies/laptop.py), [src/neuro_stylometry/pollution_guard/embedder.py](src/neuro_stylometry/pollution_guard/embedder.py), [src/neuro_stylometry/pollution_guard/leace.py](src/neuro_stylometry/pollution_guard/leace.py)

**Deliverables:**

1. Update `conf/laptop/pipeline.yaml`:
   ```yaml
   leace:
     device: "cpu"          # Strict CPU for stability
     force_cpu: true        # Explicit override
     regularization: 1e-5
     batch_size: 50
     compute_dtype: "float64"  # CPU uses higher precision
   ```

2. Modify `LaptopFilterStrategy.execute()`:
   - After `FrozenEmbedder.embed_texts()` returns embeddings, explicitly move to CPU:
     ```python
     embeddings = embeddings.detach().cpu()
     ```
   - Pass `output_device="cpu"` to embedder constructor
   - Log device placement: "Laptop mode: embeddings on GPU → moving to CPU for LEACE accumulation"

3. Update `FrozenEmbedder.__init__()` and `embed_texts()`:
   - Add `output_device: Optional[str] = None` parameter
   - If `output_device` provided, move final embeddings before returning:
     ```python
     cls_embeddings = outputs.last_hidden_state[:, 0, :]
     if output_device:
         cls_embeddings = cls_embeddings.to(output_device)
     all_embeddings.append(cls_embeddings)
     ```
   - Document in docstring: "output_device='cpu' recommended for laptop mode to reduce VRAM fragmentation"

4. Validate `LEACEComputer.accumulate_batch()`:
   - Add assertion: if `self.force_cpu and embeddings.device.type == 'cuda'`, log warning and auto-move to CPU
   - Update docstring: "Batch accumulation always transfers to CPU to prevent VRAM fragmentation"

5. Document device flow in module docstrings:
   ```
   Device Pipeline (Laptop Mode):
   1. Text → GLiNER on GPU (config: gliner.device="cuda")
   2. Embeddings extracted on GPU
   3. Embeddings → CPU (explicit output_device="cpu")
   4. LEACE accumulation on CPU (force_cpu=true)
   5. P matrix computed on CPU (FP64 for stability)
   6. P saved to disk (FP32 for downstream compatibility)
   ```

| **Dependencies:** Step 1 | **Risk:** Medium (empirical validation needed)

**Testing:** Add `test_laptop_mode_embeddings_on_cpu()` to verify embeddings not on GPU after extraction

---

## 3. Golden Reference Test Suite

**Files:** [tests/golden/test_projection_consistency.py](tests/golden/test_projection_consistency.py), [tests/golden/fixtures/](tests/golden/fixtures/), [scripts/generate_golden_fixtures.py](scripts/generate_golden_fixtures.py)

**Deliverables:**

1. Create golden test fixtures:
   - Generate `tests/golden/fixtures/sobr_small_seed42.arrow`:
     - 100 authors, ~5000 posts
     - Fixed seed=42 for reproducibility
     - Same schema as production `sobr.arrow`
     - Script: `scripts/generate_golden_fixtures.py`

2. Generate reference artifacts:
   - Run Phase A on `sobr_small_seed42.arrow` twice (laptop mode, HPC mode) with seed=42
   - Save outputs:
     - `expected_P_laptop_seed42.pt` (projection matrix)
     - `expected_P_hpc_seed42.pt` (projection matrix)
     - `expected_clean_dataset_seed42.arrow`
     - `expected_pollution_logs_seed42.arrow`

3. Implement `tests/golden/test_projection_consistency.py`:
   ```python
   @pytest.mark.golden
   @pytest.mark.determinism
   def test_leace_idempotence_golden(self):
       """P^2 ≈ P for golden reference projection matrix."""
       P = torch.load("tests/golden/fixtures/expected_P_laptop_seed42.pt")
       P2 = P @ P
       rel_error = torch.norm(P2 - P) / torch.norm(P)
       assert rel_error < 1e-5, f"Idempotence violated: {rel_error}"
   
   @pytest.mark.golden
   @pytest.mark.determinism
   def test_cross_mode_projection_equivalence(self):
       """Laptop and HPC projections correlate > 0.98."""
       P_laptop = torch.load("tests/golden/fixtures/expected_P_laptop_seed42.pt")
       P_hpc = torch.load("tests/golden/fixtures/expected_P_hpc_seed42.pt")
       
       # Flatten to vectors
       p_laptop_vec = P_laptop.flatten()
       p_hpc_vec = P_hpc.flatten()
       
       # Pearson correlation
       correlation = torch.corrcoef(torch.stack([p_laptop_vec, p_hpc_vec]))[0, 1]
       assert correlation > 0.98, f"Cross-mode correlation: {correlation}"
       
       # Frobenius relative error
       rel_error = torch.norm(P_laptop - P_hpc) / torch.norm(P_laptop)
       assert rel_error < 0.05, f"Frobenius error: {rel_error}"
   
   @pytest.mark.golden
   @pytest.mark.determinism
   def test_deterministic_rerun_artifact_identity(self):
       """Running Phase A twice with seed=42 produces identical artifacts."""
       dataset_path = Path("tests/golden/fixtures/sobr_small_seed42.arrow")
       
       # Run 1
       artifacts_1 = PhaseAPipeline.from_yaml(
           strategy=LaptopFilterStrategy(),
           mode="laptop"
       ).run(dataset_path, tmp_path_1)
       
       # Run 2
       artifacts_2 = PhaseAPipeline.from_yaml(
           strategy=LaptopFilterStrategy(),
           mode="laptop"
       ).run(dataset_path, tmp_path_2)
       
       # Assert byte-for-byte identity
       assert artifacts_1.clean_dataset_path.read_bytes() == artifacts_2.clean_dataset_path.read_bytes()
       assert artifacts_1.pollution_logs_path.read_bytes() == artifacts_2.pollution_logs_path.read_bytes()
       
       # Assert projection matrix numerical equivalence
       P1 = torch.load(artifacts_1.projection_matrix_path)
       P2 = torch.load(artifacts_2.projection_matrix_path)
       assert torch.allclose(P1, P2, atol=1e-7)
   ```

4. Create `scripts/generate_golden_fixtures.py`:
   - Accepts `--output-dir` and `--seed` arguments
   - Generates small SOBR dataset with fixed seed
   - Runs Phase A in both laptop and HPC modes
   - Saves reference artifacts with deterministic filenames

| **Dependencies:** Step 2 | **Risk:** Medium (golden data size management)

---

## 4. Numerical Correctness Test Suite

**File:** [tests/unit/test_leace.py](tests/unit/test_leace.py)

**Deliverables:**

1. Populate previously-stub file with unit tests:

   ```python
   @pytest.mark.unit
   class TestLEACEMathematicalProperties:
       
       @pytest.mark.determinism
       def test_leace_idempotence_property(self, device):
           """Verify P^2 ≈ P for computed projection matrix."""
           # Generate synthetic embeddings and labels
           embeddings = torch.randn(100, 768, device=device)
           labels = torch.randint(0, 7, (100,), device=device)
           
           # Compute projection
           computer = LEACEComputer(embedding_dim=768, device=str(device))
           P = computer.compute(embeddings, labels)
           
           # Check idempotence
           P2 = P @ P
           rel_error = torch.norm(P2 - P) / torch.norm(P)
           assert rel_error < 1e-5, f"Idempotence failed: {rel_error}"
       
       def test_leace_condition_number_sparse_demographics(self):
           """LEACE remains numerically stable with imbalanced classes."""
           # 95% female, 5% male
           embeddings = torch.randn(200, 768)
           labels = torch.tensor([0]*190 + [1]*10)  # Highly imbalanced
           
           computer = LEACEComputer(embedding_dim=768, device="cpu")
           P = computer.compute(embeddings, labels)
           
           # Compute condition number of covariance matrix
           # (via saved stats in LEACEComputer)
           U, S, _ = torch.svd(computer._last_stats.within_class_cov)
           cond_number = S[0] / S[S > 1e-12][-1]
           
           assert cond_number < 1e8, f"Ill-conditioned matrix: cond={cond_number}"
           assert torch.allclose(P @ P, P, atol=1e-5)
       
       def test_leace_batch_accumulation_equals_full_batch(self):
           """Mini-batch accumulation produces same P as full-batch."""
           embeddings = torch.randn(1000, 768)
           labels = torch.randint(0, 7, (1000,))
           
           # Full-batch computation
           computer_full = LEACEComputer(embedding_dim=768)
           P_full = computer_full.compute(embeddings, labels)
           
           # Mini-batch accumulation (batch_size=100)
           computer_mini = LEACEComputer(embedding_dim=768)
           for i in range(0, 1000, 100):
               computer_mini.accumulate_batch(
                   embeddings[i:i+100],
                   labels[i:i+100]
               )
           P_mini = computer_mini.finalize()
           
           # Assert < 0.1% relative error
           rel_error = torch.norm(P_full - P_mini) / torch.norm(P_full)
           assert rel_error < 1e-3, f"Accumulation error: {rel_error}"
       
       def test_leace_numerical_precision_across_dtypes(self):
           """Projection matrix consistent across FP32/FP64."""
           embeddings = torch.randn(500, 768)
           labels = torch.randint(0, 7, (500,))
           
           # FP32 computation
           computer32 = LEACEComputer(embedding_dim=768, device="cpu")
           P32 = computer32.compute(embeddings.float(), labels)
           
           # FP64 computation
           computer64 = LEACEComputer(embedding_dim=768, device="cpu")
           P64 = computer64.compute(embeddings.double(), labels.double())
           
           # Relative difference should be small
           rel_diff = torch.norm(P32 - P64.float()) / torch.norm(P32)
           assert rel_diff < 1e-4, f"Precision mismatch: {rel_diff}"
       
       def test_leace_cpu_gpu_equivalence(self, device):
           """CPU and GPU projections match within tolerance."""
           if device.type == "cpu":
               pytest.skip("Test requires GPU comparison")
           
           embeddings = torch.randn(500, 768)
           labels = torch.randint(0, 7, (500,))
           
           # CPU computation
           computer_cpu = LEACEComputer(embedding_dim=768, device="cpu", force_cpu=True)
           P_cpu = computer_cpu.compute(embeddings, labels)
           
           # GPU computation
           computer_gpu = LEACEComputer(embedding_dim=768, device="cuda", force_cpu=False)
           P_gpu = computer_gpu.compute(embeddings.cuda(), labels.cuda()).cpu()
           
           # Allow higher tolerance for GPU float32 vs CPU float64
           assert torch.allclose(P_cpu, P_gpu, atol=1e-3, rtol=1e-2), \
               f"CPU-GPU mismatch exceeds tolerance"
   ```

2. Cover eigendecomposition stability, SVD truncation logic, regularization effects

3. Add parametrized tests for different embedding dimensions (128, 768, 1024) and label cardinalities (2-20)

| **Dependencies:** None | **Risk:** Low

---

## 5. Functional Correctness & Edge Case Test Suite

**Files:** [tests/integration/test_gliner_guard.py](tests/integration/test_gliner_guard.py), [tests/unit/test_masker.py](tests/unit/test_masker.py)

**Deliverables:**

1. Enhance `test_gliner_guard.py` with edge cases:
   ```python
   @pytest.mark.integration
   @pytest.mark.parametrize("text,description", [
       ("", "empty_string"),
       ("a", "single_character"),
       ("a" * 10000, "extremely_long_text"),
       ("... ... ...", "punctuation_only"),
       ("🎉👍🌟", "emoji_only"),
   ])
   def test_gliner_detector_boundary_texts(self, gliner_detector, text, description):
       """GLiNER handles boundary condition inputs robustly."""
       result = gliner_detector.detect_spans([text], batch_size=1, show_progress=False)
       
       if description == "empty_string":
           assert result[0] == [], "Empty text should yield no entities"
       elif description == "extremely_long_text":
           # Text longer than max_length should be chunked
           assert isinstance(result[0], list), "Result should be list"
       else:
           # Other cases should process without error
           assert isinstance(result[0], list), f"Processing failed for {description}"
   
   @pytest.mark.integration
   def test_gliner_unicode_handling(self, gliner_detector):
       """Pollution detection works with diverse Unicode scripts."""
       texts = [
           "我叫李明，我今年25岁。",  # Simplified Chinese + age
           "مرحبا، أنا من مصر وعمري 30 سنة.",  # Arabic + age
           "Je m'appelle Pierre et je suis français.",  # French + nationality
           "Мне 28 лет и я из Москвы.",  # Russian + age
       ]
       results = gliner_detector.detect_spans(texts, batch_size=2)
       
       # Should process all texts without encoding errors
       assert len(results) == len(texts)
       for entities in results:
           assert isinstance(entities, list)
   
   @pytest.mark.integration
   def test_gliner_adversarial_repetition(self, gliner_detector):
       """Detector doesn't hallucinate on repetitive patterns."""
       text = "I am I am I am I am I am."
       result = gliner_detector.detect_spans([text], batch_size=1)
       
       # Should not report more than 3 detections for "I am"
       entities = result[0]
       age_entities = [e for e in entities if e["label"] == "age_statement"]
       assert len(age_entities) <= 3, "Hallucinated age detection on repetition"
   
   @pytest.mark.integration
   def test_span_width_constraints(self, gliner_detector):
       """EntityWidthConstraints correctly filter out-of-bounds spans."""
       # Construct text where some entities exceed width limits
       text = " ".join(["word"] * 50)  # 50 words
       
       constraints = gliner_detector.width_constraints
       entities = gliner_detector.detect_spans([text], batch_size=1)[0]
       
       for entity in entities:
           span_text = entity["text"]
           span_tokens = gliner_detector.model.data_processor.transformer_tokenizer.tokenize(span_text)
           token_count = len(span_tokens)
           
           label = entity["label"]
           if label in constraints:
               min_w, max_w = constraints[label]
               assert min_w <= token_count <= max_w, \
                   f"Entity '{span_text}' violates constraints {label}: {min_w}-{max_w}, got {token_count}"
   ```

2. Create `tests/unit/test_masker.py`:
   ```python
   @pytest.mark.unit
   class TestSpanMasker:
       
       def test_masking_offsets_round_trip_consistency(self, span_masker):
           """Masked text offsets allow reconstruction of original."""
           original_text = "I am 25 years old and live in Berlin, Germany."
           spans = [
               {"start": 5, "end": 7, "text": "25", "label": "age_statement"},
               {"start": 31, "end": 37, "text": "Berlin", "label": "nationality_statement"},
           ]
           
           masked_text, logs = span_masker.mask(original_text, spans)
           
           # Reconstruct original
           reconstructed = masked_text
           for log in sorted(logs, key=lambda x: x["char_start"], reverse=True):
               mask_token = log["mask_token"]
               char_start = log["char_start"]
               char_end = log["char_end"]
               original_span = original_text[char_start:char_end]
               reconstructed = reconstructed.replace(mask_token, original_span, 1)
           
           assert reconstructed == original_text, \
               f"Round-trip failed.\nOriginal: {original_text}\nReconstructed: {reconstructed}"
       
       def test_masking_overlapping_spans_resolution(self, span_masker):
           """Overlapping spans are resolved consistently."""
           text = "I am a 25-year-old engineer from the USA."
           
           # Overlapping spans: "25" and "25-year-old"
           spans = [
               {"start": 8, "end": 10, "text": "25", "label": "age_statement"},
               {"start": 8, "end": 19, "text": "25-year-old", "label": "age_statement"},
           ]
           
           masked_text, logs = span_masker.mask(text, spans)
           
           # Should resolve to longest span (or earliest - check implementation)
           # Verify no double-masking
           mask_count = masked_text.count("[MASK:AGE]")
           assert mask_count == 1, f"Expected 1 mask token, got {mask_count}"
   ```

3. Add integration tests for real SOBR data: `test_gliner_real_sobr_texts()` with actual posts

| **Dependencies:** None | **Risk:** Medium (fixture complexity)

---

## 6. Phase A Handover Contract Validation Suite

**File:** [tests/integration/test_phase_a_handover.py](tests/integration/test_phase_a_handover.py) (new)

**Deliverables:**

1. Create comprehensive handover validation tests:
   ```python
   @pytest.mark.integration
   @pytest.mark.determinism
   class TestPhaseAHandoverContract:
       
       def test_handover_contract_artifact_existence(self, phase_a_artifacts):
           """All 3 required artifacts exist and are non-empty."""
           assert phase_a_artifacts.clean_dataset_path.exists(), \
               f"clean_dataset.arrow not found at {phase_a_artifacts.clean_dataset_path}"
           assert phase_a_artifacts.projection_matrix_path.exists(), \
               f"projection_matrix.pt not found"
           assert phase_a_artifacts.pollution_logs_path.exists(), \
               f"pollution_logs.arrow not found"
           
           # Check file sizes
           assert phase_a_artifacts.clean_dataset_path.stat().st_size > 0
           assert phase_a_artifacts.projection_matrix_path.stat().st_size > 0
           assert phase_a_artifacts.pollution_logs_path.stat().st_size > 0
       
       def test_handover_contract_schema_compliance(self, phase_a_artifacts):
           """clean_dataset.arrow matches SOBR_SCHEMA."""
           import pyarrow.parquet as pq
           
           table = pq.read_table(phase_a_artifacts.clean_dataset_path)
           
           # Validate schema
           expected_schema = SOBR_SCHEMA
           assert table.schema == expected_schema, \
               f"Schema mismatch.\nExpected: {expected_schema}\nGot: {table.schema}"
           
           # Validate post_masked column
           post_masked = table["post_masked"].to_pylist()
           assert all(isinstance(pm, str) for pm in post_masked), \
               "post_masked column contains non-string values"
           
           # Spot-check: some rows should contain [MASK:*] tokens
           masked_rows = [pm for pm in post_masked if "[MASK:" in pm]
           assert len(masked_rows) > 0, \
               "No rows contain mask tokens (no pollution detected?)"
       
       def test_handover_contract_projection_matrix_validity(self, phase_a_artifacts):
           """projection_matrix.pt is valid and idempotent."""
           P = torch.load(phase_a_artifacts.projection_matrix_path, map_location="cpu")
           
           # Check shape
           assert P.shape == (768, 768), f"Expected shape (768, 768), got {P.shape}"
           
           # Check dtype
           assert P.dtype == torch.float32, f"Expected FP32, got {P.dtype}"
           
           # Check idempotence
           P2 = P @ P
           rel_error = torch.norm(P2 - P) / torch.norm(P)
           assert rel_error < 1e-5, \
               f"Idempotence violated: ||P^2 - P||_F / ||P||_F = {rel_error}"
       
       def test_handover_contract_pollution_logs_schema(self, phase_a_artifacts):
           """pollution_logs.arrow matches POLLUTION_LOG_SCHEMA."""
           import pyarrow.parquet as pq
           
           logs_table = pq.read_table(phase_a_artifacts.pollution_logs_path)
           
           # Validate schema
           assert logs_table.schema == POLLUTION_LOG_SCHEMA, \
               f"Schema mismatch in pollution logs"
           
           # Validate offsets
           for row in logs_table.to_pylist():
               char_start = row["char_start"]
               char_end = row["char_end"]
               assert 0 <= char_start < char_end, \
                   f"Invalid offset range: [{char_start}, {char_end})"
       
       def test_handover_contract_cross_reference_validity(self, phase_a_artifacts):
           """All spans in pollution logs reference valid posts in clean dataset."""
           import pyarrow.parquet as pq
           
           dataset = pq.read_table(phase_a_artifacts.clean_dataset_path)
           logs = pq.read_table(phase_a_artifacts.pollution_logs_path)
           
           valid_post_ids = set(dataset["post_id"].to_pylist())
           
           for row in logs.to_pylist():
               post_id = row["post_id"]
               assert post_id in valid_post_ids, \
                   f"Span references unknown post_id: {post_id}"
               
               # Cross-check: offsets fit within post
               post_row = dataset.filter(dataset["post_id"] == post_id).to_pylist()[0]
               post_masked = post_row["post_masked"]
               char_end = row["char_end"]
               
               assert char_end <= len(post_masked), \
                   f"Span offset {char_end} exceeds post length {len(post_masked)}"
   
   @pytest.fixture
   def phase_a_artifacts(tmp_path, set_all_seeds):
       """Run Phase A once and return artifacts for validation."""
       dataset_path = Path("artifacts/data/sobr_laptop.arrow")
       if not dataset_path.exists():
           pytest.skip("Laptop dataset not found")
       
       pipeline = PhaseAPipeline.from_yaml(
           strategy=LaptopFilterStrategy(),
           mode="laptop"
       )
       artifacts = pipeline.run(dataset_path, tmp_path)
       return artifacts
   ```

2. Integrate script validation into test suite (move/wrap `scripts/validate_phase_a_handover.py` logic)

3. Mark all with `@pytest.mark.integration` and `@pytest.mark.determinism`

| **Dependencies:** None | **Risk:** Low

---

## 7. Refactor CLI to Use PhaseAPipeline Facade

**File:** [src/neuro_stylometry/__main__.py](src/neuro_stylometry/__main__.py)

**Deliverables:**

1. Simplify `run_phase_a` command:
   ```python
   @cli.command("run-phase-a")
   @click.option("--dataset", type=click.Path(exists=True), required=True,
                 help="Path to dataset Arrow file")
   @click.option("--output-dir", type=click.Path(), required=True,
                 help="Output directory for artifacts")
   @click.option("--mode", type=click.Choice(["auto", "laptop", "hpc"]), 
                 default="auto", help="Hardware mode")
   @click.option("--config", "config_path", type=click.Path(exists=True),
                 help="Optional experiment config override")
   @click.option("--dry-run", is_flag=True, 
                 help="Print resolved config without execution")
   def run_phase_a(dataset, output_dir, mode, config_path, dry_run):
       """Execute Phase A pollution guard pipeline."""
       try:
           # Load strategy based on hardware detection
           if mode == "auto":
               profile = HardwareDetector.detect()
               mode = "laptop" if profile.profile_type == ProfileType.LAPTOP else "hpc"
               logger.info(f"Auto-detected mode: {mode}")
           
           strategy = StrategyFactory.create_filter_strategy(mode)
           
           # Create pipeline from YAML
           pipeline = PhaseAPipeline.from_yaml(
               strategy=strategy,
               mode=mode,
               experiment_config_path=Path(config_path) if config_path else None
           )
           
           if dry_run:
               config_dict = pipeline.config
               print("Resolved Configuration:")
               import yaml
               print(yaml.dump(config_dict, default_flow_style=False))
               return
           
           # Execute pipeline
           artifacts = pipeline.run(
               input_dataset_path=Path(dataset),
               output_dir=Path(output_dir)
           )
           
           logger.info(f"Phase A complete. Artifacts:")
           logger.info(f"  - Clean dataset: {artifacts.clean_dataset_path}")
           logger.info(f"  - Projection matrix: {artifacts.projection_matrix_path}")
           logger.info(f"  - Pollution logs: {artifacts.pollution_logs_path}")
           
       except Exception as e:
           logger.error(f"Phase A execution failed: {e}", exc_info=True)
           raise click.ClickException(str(e))
   ```

2. Add `config-show` command:
   ```python
   @cli.command("config-show")
   @click.option("--mode", type=click.Choice(["auto", "laptop", "hpc"]), 
                 default="auto")
   @click.option("--config", "config_path", type=click.Path(exists=True))
   def config_show(mode, config_path):
       """Display merged configuration for a given mode."""
       config = load_pipeline_config(
           mode=mode,
           experiment_config_path=Path(config_path) if config_path else None
       )
       import yaml
       print(yaml.dump(config, default_flow_style=False))
   ```

3. Fix `validate_handover` command:
   ```python
   @cli.command("validate-handover")
   @click.option("--artifacts-dir", type=click.Path(exists=True), required=True,
                 help="Phase A artifacts directory")
   def validate_handover(artifacts_dir):
       """Validate Phase A handover contract compliance."""
       from pathlib import Path
       from neuro_stylometry.pollution_guard.leace import validate_handover_contract
       
       try:
           result = validate_handover_contract(Path(artifacts_dir))
           if result["valid"]:
               click.echo("✓ Handover contract satisfied")
               for key, value in result.items():
                   if key != "valid":
                       click.echo(f"  {key}: {value}")
           else:
               click.echo("✗ Handover contract violated")
               for error in result.get("errors", []):
                   click.echo(f"  - {error}")
               raise click.ClickException("Validation failed")
       except Exception as e:
           raise click.ClickException(f"Validation error: {e}")
   ```

4. Remove/stub Phase D commands (placeholder only):
   ```python
   @cli.command("run-phase-d")
   def run_phase_d():
       """Phase D (Neural Stylometry) - Not implemented."""
       raise click.ClickException(
           "Phase D is not yet implemented. "
           "Phase A artifacts are ready at artifacts/phase_a/"
       )
   ```

| **Dependencies:** None | **Risk:** Low

---

## 8. Configuration Validation & Type Safety

**Files:** [src/neuro_stylometry/config.py](src/neuro_stylometry/config.py), [tests/unit/test_config.py](tests/unit/test_config.py)

**Deliverables:**

1. Create `PipelineConfigSchema` dataclass:
   ```python
   from dataclasses import dataclass
   from typing import Annotated, Optional
   from pydantic import Field
   import pydantic.dataclasses
   
   @pydantic.dataclasses.dataclass
   class GLiNERConfig:
       device: str = Field(default="auto", pattern="^(auto|cpu|cuda)$")
       batch_size: int = Field(default=32, gt=0)
       confidence_threshold: float = Field(default=0.85, ge=0.0, le=1.0)
       max_length: int = Field(default=512, gt=0)
       center_window_keep: int = Field(default=100, ge=0)
   
   @pydantic.dataclasses.dataclass
   class LEACEConfig:
       device: str = Field(default="cpu", pattern="^(cpu|cuda)$")
       force_cpu: bool = Field(default=True)
       regularization: float = Field(default=1e-5, ge=1e-10)
       batch_size: int = Field(default=50, gt=0)
       compute_dtype: str = Field(default="float64", pattern="^(float32|float64)$")
   
   @pydantic.dataclasses.dataclass
   class PipelineConfigSchema:
       seed: int = Field(default=42, ge=0)
       gliner: GLiNERConfig = Field(default_factory=GLiNERConfig)
       leace: LEACEConfig = Field(default_factory=LEACEConfig)
       subset: SubsetConfig = Field(default_factory=SubsetConfig)
       quality: QualityConfig = Field(default_factory=QualityConfig)
   ```

2. Implement `validate_pipeline_config()`:
   ```python
   def validate_pipeline_config(config: Dict[str, Any]) -> PipelineConfigSchema:
       """Validate and convert OmegaConf dict to typed dataclass."""
       try:
           schema = PipelineConfigSchema(**config)
           return schema
       except pydantic.ValidationError as e:
           errors = [
               f"{'.'.join(str(x) for x in err['loc'])}: {err['msg']}"
               for err in e.errors()
           ]
           raise ConfigValidationError(
               f"Invalid pipeline configuration:\n" + "\n".join(errors)
           )
   ```

3. Update `load_pipeline_config()` to call validation:
   ```python
   def load_pipeline_config(...) -> PipelineConfigSchema:
       # ... merge YAML ...
       merged = OmegaConf.merge(...)
       config_dict = OmegaConf.to_container(merged, resolve=True)
       
       # Validate before returning
       validated = validate_pipeline_config(config_dict)
       return validated
   ```

4. Create `tests/unit/test_config.py`:
   ```python
   @pytest.mark.unit
   class TestConfigValidation:
       
       def test_config_rejects_invalid_confidence_threshold(self):
           """confidence_threshold must be in [0, 1]."""
           invalid_configs = [
               {"gliner": {"confidence_threshold": -0.1}},
               {"gliner": {"confidence_threshold": 1.5}},
           ]
           for config in invalid_configs:
               with pytest.raises(ConfigValidationError):
                   validate_pipeline_config(config)
       
       def test_config_rejects_negative_batch_size(self):
           """batch_size must be positive."""
           with pytest.raises(ConfigValidationError):
               validate_pipeline_config({"leace": {"batch_size": -1}})
       
       def test_config_requires_cuda_on_hpc_mode(self):
           """HPC mode must enforce gliner.device='cuda'."""
           config = {"gliner": {"device": "cpu"}}
           # For HPC mode: add custom validation or document requirement
           # Recommendation: Add optional `mode` parameter to schema
       
       def test_config_mode_specific_overrides_applied(self):
           """Mode-specific configs properly override base configs."""
           # Load base config
           base = load_pipeline_config(mode="base")
           laptop = load_pipeline_config(mode="laptop")
           
           # Verify laptop overrides
           assert laptop.leace.force_cpu == True, "Laptop should set force_cpu=true"
           assert laptop.leace.batch_size <= base.leace.batch_size or \
                  laptop.gliner.batch_size <= base.gliner.batch_size
   ```

**Effort:** 10-12 hours | **Dependencies:** None | **Risk:** Medium (schema design)

---

## 9. Memory & Performance Benchmarking Suite (Optional)

**File:** [tests/benchmarks/test_performance.py](tests/benchmarks/test_performance.py)

**Deliverables:**

1. Create memory profiling tests:
   ```python
   @pytest.mark.benchmark
   @pytest.mark.slow
   @pytest.mark.skip(reason="Manual execution only")
   def test_leace_memory_footprint_full_dataset():
       """Profile LEACE memory usage on full 390k dataset."""
       # Requires tracemalloc or memory_profiler
   ```

2. Create latency benchmarks:
   ```python
   @pytest.mark.benchmark
   @pytest.mark.slow
   def test_embedder_throughput_benchmark(benchmark):
       """Measure tokens/second for embedding extraction."""
       # Use pytest-benchmark plugin
   ```

3. Create regression detection fixtures

| **Dependencies:** Optional | **Risk:** Low

---

## 10. Test Execution Documentation

**Files:** [tests/README.md](tests/README.md), root [README.md](README.md)

**Deliverables:**

1. Create `tests/README.md`:
   ```markdown
   # Testing Guide
   
   ## Quick Start
   
   ```bash
   # Run all unit tests (CPU only)
   pytest tests/unit -m "not real_models"
   
   # Run integration tests with real models
   export NEURO_STYLOMETRY_RUN_REAL_MODELS=1
   pytest tests/integration -m "not hpc"
   
   # Run HPC-specific tests (on HPC cluster)
   pytest -m hpc
   
   # Run golden reference tests
   pytest tests/golden -m determinism
   ```
   
   ## Test Markers
   
   - `@pytest.mark.unit`: Fast unit tests, no models
   - `@pytest.mark.integration`: Slower integration tests
   - `@pytest.mark.golden`: Golden reference tests (requires golden fixtures)
   - `@pytest.mark.determinism`: Determinism/reproducibility tests
   - `@pytest.mark.real_models`: Tests requiring GLiNER/RoBERTa downloads
   - `@pytest.mark.cuda`: GPU-specific tests
   - `@pytest.mark.cpu`: CPU-only tests
   - `@pytest.mark.hpc`: HPC-profile tests (32+ cores or 40GB+ VRAM)
   - `@pytest.mark.laptop`: Laptop-profile tests
   - `@pytest.mark.slow`: Long-running tests
   - `@pytest.mark.benchmark`: Performance benchmarks
   
   ## Hardware Detection
   
   Tests automatically adapt to detected hardware:
   ```python
   # Detected as HPC (64 cores, 512GB RAM, A100 GPU)
   pytest tests/integration  # Runs HPC tests, full dataset
   
   # Detected as Laptop (8 cores, 16GB RAM, GTX1080)
   pytest tests/integration  # Skips HPC tests, uses subset config
   ```
   
   To override:
   ```python
   # Force laptop mode
   HARDWARE_PROFILE=laptop pytest tests/integration
   
   # Or use fixture in test:
   def test_something(mock_hardware_profile):
       mock_hardware_profile(ProfileType.LAPTOP, has_cuda=False)
   ```
   
   ## CI/CD Matrix
   
   **GitHub Actions Example:**
   ```yaml
   strategy:
     matrix:
       os: [ubuntu-latest, windows-latest]
       python-version: ["3.10", "3.11"]
       hardware: [laptop, hpc-simulation]
   
   steps:
     - run: pytest tests/unit tests/integration -m "not real_models"
     - run: pytest tests/integration -m real_models --cov
     - run: pytest tests/golden -m determinism
   ```
   
   ## Known Issues & Workarounds
   
   - **CUDA OOM on weak GPUs**: Set `CUDA_VISIBLE_DEVICES=0` to limit GPU visibility
   - **BLAS non-determinism**: Golden tests use relaxed tolerance (1e-4) on CPU to account for BLAS variation
   - **Windows fork() limitation**: Use `pytest-forked` for memory leak detection tests
   ```

2. Update root `README.md` with link to testing guide

| **Dependencies:** All prior steps | **Risk:** Low

---

## Effort & Timeline Summary

| Step | Hours | Weeks | Dependencies | Risk |
|------|-------|-------|-------------|------|
| 1. Pytest Config | 8-10 | 1 | None | Low |
| 2. Laptop CPU-LEACE | 6-8 | 1 | Step 1 | Medium |
| 3. Golden Tests | 12-15 | 2 | Step 2 | Medium |
| 4. Numerical Correctness | 10-12 | 1.5 | None | Low |
| 5. Functional/Edge Cases | 14-16 | 2 | None | Medium |
| 6. Handover Validation | 8-10 | 1 | None | Low |
| 7. CLI Refactor | 6-8 | 1 | None | Low |
| 8. Config Validation | 10-12 | 1.5 | None | Medium |
| 9. Benchmarking (Opt.) | 8-10 | 1 | None | Low |
| 10. Documentation | 4-6 | 0.5 | All | Low |
| **TOTAL** | **120-150** | **8-10** | — | — |

---

## Critical Path

1. **Week 1:** Steps 1, 4 (in parallel)
2. **Week 2:** Step 2, Step 7, Step 8 (in parallel)
3. **Weeks 3-4:** Steps 3, 5 (in parallel)
4. **Week 5:** Step 6, Step 9 (in parallel)
5. **Weeks 6-10:** Integration testing, CI/CD pipeline setup, documentation (Step 10)

**Critical blocking dependencies:**
- Step 3 (Golden Tests) blocked by Step 2 (Laptop CPU-LEACE)
- All testing requires Step 1 (Pytest Config)

---

## Key Metrics for Success

1. **Code Coverage:** Phase A achieves >75% line coverage, >50% branch coverage
2. **Test Execution Time:** Unit tests <30sec, integration <10min, golden <5min
3. **Determinism:** Golden tests pass with byte-identical artifacts on rerun
4. **Cross-Mode Equivalence:** Laptop and HPC projection matrices correlate > 0.98
5. **Numerical Stability:** LEACE idempotence error < 1e-5 across all condition numbers
6. **CI/CD:** All tests pass on Ubuntu + Windows, CPU + CUDA matrices

---

## Further Considerations

### Configuration-Driven Testing vs Hardcoded Fixtures

**Recommendation:** Tests should load minimal YAML fixtures from `tests/fixtures/` rather than hardcoding device strings:
- `tests/fixtures/pipeline_test_cpu.yaml` for CPU-only tests
- `tests/fixtures/pipeline_test_cuda.yaml` for CUDA tests
- Benefits: Tests validate config composition logic itself

### Mock vs Real Models in CI/CD

**Recommendation:**
- Unit tests: Mocked GLiNER/RoBERTa using dummy models
- Integration tests: Real models gated by `@pytest.mark.real_models`
- CI/CD: Run integration with real models nightly or on release branches

### Cross-Platform Determinism

**Issue:** PyTorch CUDA operations have known non-determinism due to cuBLAS atomics.

**Recommendation:**
- Golden tests allow 1e-4 relative tolerance on CPU (BLAS variation)
- Allow 1e-3 on CUDA (cuBLAS non-determinism)
- Document in test docstrings

### Test Data Version Control

Golden fixtures (~50-100MB) too large for Git.

**Options:**
1. **Git LFS:** `tests/golden/fixtures/*.arrow` tracked with LFS
2. **HuggingFace Hub:** `datasets.load_dataset("org/neuro-stylometry-golden", revision="v0.1")`
3. **Generate on CI:** Run `scripts/generate_golden_fixtures.py` in test setup

**Recommendation:** Use Git LFS for simplicity, add `*(binary)` to `.gitattributes`

### Laptop Mode GPU Memory Optimization

**Current:** Embeddings extracted on GPU, then moved to CPU for LEACE.

**Alternative:** Keep everything on GPU if >8GB VRAM, reduce batch sizes if <4GB.

**Recommendation:** Add adaptive batch sizing based on available VRAM:
```python
available_vram = torch.cuda.get_device_properties(0).total_memory
adaptive_batch_size = 256 if available_vram > 8e9 else 32
```

---

## Acceptance Criteria

- [ ] All pytest markers registered and functional
- [ ] Laptop mode enforces CPU-only LEACE with explicit logging
- [ ] Golden test suite runs in <5 minutes, produces identical artifacts on rerun
- [ ] Numerical correctness tests validate LEACE math across edge cases
- [ ] Functional tests cover boundary conditions and Unicode handling
- [ ] Handover contract validation integrated into CI/CD
- [ ] CLI uses PhaseAPipeline facade exclusively (no parameter leakage)
- [ ] Config validation rejects invalid values before execution
- [ ] Documentation updated with marker usage and CI/CD examples
- [ ] Phase A achieves >75% line coverage, >50% branch coverage
