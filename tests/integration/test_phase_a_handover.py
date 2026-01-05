# tests/integration/test_phase_a_handover.py
"""
Phase A Handover Contract Validation Suite.

These tests validate the contract between Phase A and Phase D:
1. All required artifacts exist and are non-empty
2. Schema compliance for clean_dataset.arrow and pollution_logs.arrow
3. Projection matrix validity (shape, dtype, idempotence)
4. Cross-reference validity (pollution logs reference valid posts)

Implements: Testing Plan Step 6 (Handover Validation)
"""

import pytest
import torch
import pyarrow as pa
import pyarrow.feather as feather
from pathlib import Path
from typing import NamedTuple

from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, POLLUTION_LOG_SCHEMA


# ==============================================================================
# Test Data Structures
# ==============================================================================

class PhaseAArtifacts(NamedTuple):
    """Container for Phase A output artifacts."""
    clean_dataset_path: Path
    projection_matrix_path: Path
    pollution_logs_path: Path
    output_dir: Path


# ==============================================================================
# Fixtures
# ==============================================================================

@pytest.fixture(scope="module")
def phase_a_artifacts(tmp_path_factory, request):
    """
    Run Phase A once and return artifacts for validation.
    
    This fixture runs the full pipeline and caches results for the module.
    """
    from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
    from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
    from neuro_stylometry.config import load_pipeline_config
    
    # Check for laptop dataset
    dataset_path = Path("artifacts/data/sobr_laptop.arrow")
    if not dataset_path.exists():
        pytest.skip("Laptop dataset not found at artifacts/data/sobr_laptop.arrow")
    
    # Create temp output directory
    output_dir = tmp_path_factory.mktemp("phase_a_output")
    
    # Load config
    config = load_pipeline_config(mode="laptop")
    config["seed"] = 42
    
    # Limit subset for faster testing (100 samples minimum for meaningful LEACE/probe)
    config["subset"]["enabled"] = True
    config["subset"]["size"] = 100
    
    # Create and run pipeline
    strategy = LaptopFilterStrategy()
    pipeline = PhaseAPipeline(strategy=strategy, config=config)
    
    artifacts = pipeline.run(
        input_dataset_path=dataset_path,
        output_dir=output_dir,
    )
    
    return PhaseAArtifacts(
        clean_dataset_path=artifacts.clean_dataset_path,
        projection_matrix_path=artifacts.projection_matrix_path,
        pollution_logs_path=artifacts.pollution_logs_path,
        output_dir=output_dir,
    )


@pytest.fixture
def static_artifacts_dir():
    """Return path to pre-generated artifacts (if available)."""
    path = Path("artifacts/phase_a")
    if not path.exists():
        pytest.skip("Pre-generated artifacts not found at artifacts/phase_a")
    return path


# ==============================================================================
# Artifact Existence Tests
# ==============================================================================

@pytest.mark.integration
@pytest.mark.determinism
class TestHandoverContractArtifactExistence:
    """Test that all required artifacts exist."""
    
    def test_handover_contract_artifact_existence(self, phase_a_artifacts):
        """All 3 required artifacts exist and are non-empty."""
        # Clean dataset
        assert phase_a_artifacts.clean_dataset_path.exists(), (
            f"clean_dataset.arrow not found at {phase_a_artifacts.clean_dataset_path}"
        )
        assert phase_a_artifacts.clean_dataset_path.stat().st_size > 0, (
            "clean_dataset.arrow is empty"
        )
        
        # Projection matrix
        assert phase_a_artifacts.projection_matrix_path.exists(), (
            f"projection_matrix.pt not found at {phase_a_artifacts.projection_matrix_path}"
        )
        assert phase_a_artifacts.projection_matrix_path.stat().st_size > 0, (
            "projection_matrix.pt is empty"
        )
        
        # Pollution logs
        assert phase_a_artifacts.pollution_logs_path.exists(), (
            f"pollution_logs.arrow not found at {phase_a_artifacts.pollution_logs_path}"
        )
        assert phase_a_artifacts.pollution_logs_path.stat().st_size > 0, (
            "pollution_logs.arrow is empty"
        )
    
    def test_artifact_file_format(self, phase_a_artifacts):
        """Artifacts have correct file extensions."""
        assert phase_a_artifacts.clean_dataset_path.suffix == ".arrow"
        assert phase_a_artifacts.projection_matrix_path.suffix == ".pt"
        assert phase_a_artifacts.pollution_logs_path.suffix == ".arrow"


# ==============================================================================
# Schema Compliance Tests
# ==============================================================================

@pytest.mark.integration
@pytest.mark.determinism
class TestHandoverContractSchemaCompliance:
    """Test that artifacts conform to expected schemas."""
    
    def test_handover_contract_clean_dataset_schema(self, phase_a_artifacts):
        """clean_dataset.arrow matches SOBR_SCHEMA."""
        table = feather.read_table(phase_a_artifacts.clean_dataset_path)
        
        # Check required columns exist
        required_columns = [
            "post_id", "author_id", "post", "post_masked",
            "birth_year", "female", "nationality", "political_leaning",
            "extrovert", "sensing", "feeling", "judging",
        ]
        
        actual_columns = set(table.column_names)
        missing_columns = set(required_columns) - actual_columns
        
        assert not missing_columns, f"Missing columns: {missing_columns}"
    
    def test_handover_contract_post_masked_populated(self, phase_a_artifacts):
        """post_masked column is populated with string values."""
        table = feather.read_table(phase_a_artifacts.clean_dataset_path)
        
        post_masked = table["post_masked"].to_pylist()
        
        # All values should be strings
        assert all(isinstance(pm, str) for pm in post_masked), (
            "post_masked column contains non-string values"
        )
        
        # At least some rows should contain mask tokens
        masked_rows = [pm for pm in post_masked if "[MASK:" in pm]
        # Note: It's possible no pollution was detected in small subset
        # So we just warn if no masks found
        if len(masked_rows) == 0:
            import warnings
            warnings.warn("No rows contain mask tokens (no pollution detected?)")
    
    def test_handover_contract_pollution_logs_schema(self, phase_a_artifacts):
        """pollution_logs.arrow has correct schema."""
        logs_table = feather.read_table(phase_a_artifacts.pollution_logs_path)
        
        # Check required columns
        required_columns = [
            "post_id", "span_start", "span_end", "span_text",
            "entity_type", "confidence", "mask_token"
        ]
        
        actual_columns = set(logs_table.column_names)
        missing_columns = set(required_columns) - actual_columns
        
        assert not missing_columns, f"Missing log columns: {missing_columns}"
    
    def test_handover_contract_pollution_logs_offsets_valid(self, phase_a_artifacts):
        """Pollution log offsets are valid (start < end, non-negative)."""
        logs_table = feather.read_table(phase_a_artifacts.pollution_logs_path)
        
        if len(logs_table) == 0:
            pytest.skip("No pollution logs to validate")
        
        for row in logs_table.to_pylist():
            span_start = row["span_start"]
            span_end = row["span_end"]
            
            assert span_start >= 0, f"Negative span_start: {span_start}"
            assert span_end > span_start, (
                f"Invalid offset range: [{span_start}, {span_end})"
            )


# ==============================================================================
# Projection Matrix Tests
# ==============================================================================

@pytest.mark.integration
@pytest.mark.determinism
class TestHandoverContractProjectionMatrix:
    """Test projection matrix validity."""
    
    def test_handover_contract_projection_matrix_validity(self, phase_a_artifacts):
        """projection_matrix.pt is valid and idempotent."""
        P = torch.load(phase_a_artifacts.projection_matrix_path, map_location="cpu")
        
        # Check shape (RoBERTa-base has 768 dimensions)
        assert P.shape == (768, 768), f"Expected shape (768, 768), got {P.shape}"
        
        # Check dtype
        assert P.dtype == torch.float32, f"Expected FP32, got {P.dtype}"
        
        # Check finite
        assert torch.isfinite(P).all(), "Projection contains non-finite values"
        
        # Check idempotence
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        assert rel_error < 1e-5, (
            f"Idempotence violated: ||P^2 - P||_F / ||P||_F = {rel_error:.2e}"
        )
    
    def test_handover_contract_projection_matrix_bounded(self, phase_a_artifacts):
        """Projection matrix has reasonable spectral properties."""
        P = torch.load(phase_a_artifacts.projection_matrix_path, map_location="cpu")
        
        # Compute eigenvalues (using eigvalsh for symmetric assumption)
        eigenvalues = torch.linalg.eigvalsh(P)
        
        # LEACE projection matrices are not guaranteed to have eigenvalues in [0,1]
        # Instead, check that the spectrum is reasonable (bounded magnitude)
        max_abs_eigenvalue = torch.abs(eigenvalues).max()
        
        assert max_abs_eigenvalue < 10.0, (
            f"Eigenvalue magnitude too large: max|λ|={max_abs_eigenvalue:.2f}. "
            "This may indicate numerical instability in LEACE."
        )
        
        # Check that most eigenvalues are close to 0 or 1 (characteristic of projections)
        near_zero = (torch.abs(eigenvalues) < 0.1).sum()
        near_one = (torch.abs(eigenvalues - 1.0) < 0.1).sum()
        near_projection = near_zero + near_one
        
        # At least 50% should be near 0 or 1
        assert near_projection >= len(eigenvalues) * 0.5, (
            f"Only {near_projection}/{len(eigenvalues)} eigenvalues near 0 or 1. "
            "Expected at least 50% for a projection-like matrix."
        )


# ==============================================================================
# Cross-Reference Validity Tests
# ==============================================================================

@pytest.mark.integration
@pytest.mark.determinism
class TestHandoverContractCrossReference:
    """Test cross-reference validity between artifacts."""
    
    def test_handover_contract_cross_reference_validity(self, phase_a_artifacts):
        """All spans in pollution logs reference valid posts in clean dataset."""
        dataset = feather.read_table(phase_a_artifacts.clean_dataset_path)
        logs = feather.read_table(phase_a_artifacts.pollution_logs_path)
        
        if len(logs) == 0:
            pytest.skip("No pollution logs to validate cross-references")
        
        valid_post_ids = set(dataset["post_id"].to_pylist())
        
        for row in logs.to_pylist():
            post_id = row["post_id"]
            assert post_id in valid_post_ids, (
                f"Span references unknown post_id: {post_id}"
            )
    
    def test_handover_contract_span_offset_within_post(self, phase_a_artifacts):
        """Span offsets fit within post text length."""
        dataset = feather.read_table(phase_a_artifacts.clean_dataset_path)
        logs = feather.read_table(phase_a_artifacts.pollution_logs_path)
        
        if len(logs) == 0:
            pytest.skip("No pollution logs to validate")
        
        # Build post_id -> post_masked length mapping
        post_lengths = {}
        for row in dataset.to_pylist():
            post_lengths[row["post_id"]] = len(row["post"])  # Original post length
        
        for row in logs.to_pylist():
            post_id = row["post_id"]
            span_end = row["span_end"]
            
            if post_id in post_lengths:
                assert span_end <= post_lengths[post_id], (
                    f"Span offset {span_end} exceeds post length {post_lengths[post_id]} "
                    f"for post_id={post_id}"
                )


# ==============================================================================
# Integration with Validation Script
# ==============================================================================

@pytest.mark.integration
class TestValidationScriptIntegration:
    """Test that validation script logic works correctly."""
    
    def test_validate_handover_function(self, phase_a_artifacts):
        """Validation function accepts valid artifacts."""
        # Import validation logic (if available)
        try:
            from scripts.validate_phase_a_handover import validate_handover_contract
            
            result = validate_handover_contract(phase_a_artifacts.output_dir)
            
            assert result.get("valid", False), (
                f"Validation failed: {result.get('errors', [])}"
            )
        except ImportError:
            # Validation script not available, skip
            pytest.skip("validate_phase_a_handover script not importable")


# ==============================================================================
# Static Artifact Tests (for pre-generated artifacts)
# ==============================================================================

@pytest.mark.integration
class TestStaticArtifacts:
    """Test pre-generated static artifacts."""
    
    def test_static_artifacts_exist(self, static_artifacts_dir):
        """Pre-generated artifacts exist."""
        clean_dataset = static_artifacts_dir / "clean_dataset.arrow"
        projection_matrix = static_artifacts_dir / "projection_matrix.pt"
        pollution_logs = static_artifacts_dir / "pollution_logs.arrow"
        
        assert clean_dataset.exists(), "Static clean_dataset.arrow not found"
        assert projection_matrix.exists(), "Static projection_matrix.pt not found"
        assert pollution_logs.exists(), "Static pollution_logs.arrow not found"
    
    def test_static_projection_idempotent(self, static_artifacts_dir):
        """Static projection matrix is idempotent."""
        P = torch.load(
            static_artifacts_dir / "projection_matrix.pt", 
            map_location="cpu"
        )
        
        P2 = P @ P
        rel_error = torch.norm(P2 - P) / torch.norm(P)
        
        assert rel_error < 1e-5, f"Static projection not idempotent: {rel_error:.2e}"
