# tests/unit/test_config.py
"""
Unit tests for configuration loading and validation.

These tests validate:
1. Configuration validation rejects invalid values
2. Mode-specific overrides are applied correctly
3. Experiment configs merge properly
4. ConfigValidationError is raised for invalid configs

Implements: Testing Plan Step 8 (Configuration Validation)
"""

import pytest
from pathlib import Path

from neuro_stylometry.config import (
    load_pipeline_config,
    validate_pipeline_config,
    ConfigValidationError,
    GLiNERConfig,
    LEACEConfig,
    EncoderConfig,
    SubsetConfig,
    ProbeConfig,
)


# ==============================================================================
# GLiNER Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestGLiNERConfigValidation:
    """Test GLiNER configuration validation."""
    
    def test_valid_gliner_config(self):
        """Valid GLiNER config passes validation."""
        config = GLiNERConfig(
            device="cuda",
            batch_size=32,
            confidence_threshold=0.85,
        )
        assert config.device == "cuda"
    
    def test_config_rejects_invalid_confidence_threshold_high(self):
        """confidence_threshold > 1 is rejected."""
        with pytest.raises(ConfigValidationError, match="confidence_threshold"):
            GLiNERConfig(confidence_threshold=1.5)
    
    def test_config_rejects_invalid_confidence_threshold_low(self):
        """confidence_threshold < 0 is rejected."""
        with pytest.raises(ConfigValidationError, match="confidence_threshold"):
            GLiNERConfig(confidence_threshold=-0.1)
    
    def test_config_rejects_invalid_device(self):
        """Invalid device string is rejected."""
        with pytest.raises(ConfigValidationError, match="device"):
            GLiNERConfig(device="gpu")  # Should be "cuda"
    
    def test_config_rejects_negative_batch_size(self):
        """Negative batch_size is rejected."""
        with pytest.raises(ConfigValidationError, match="batch_size"):
            GLiNERConfig(batch_size=-1)
    
    def test_config_rejects_zero_batch_size(self):
        """Zero batch_size is rejected."""
        with pytest.raises(ConfigValidationError, match="batch_size"):
            GLiNERConfig(batch_size=0)


# ==============================================================================
# LEACE Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestLEACEConfigValidation:
    """Test LEACE configuration validation."""
    
    def test_valid_leace_config(self):
        """Valid LEACE config passes validation."""
        config = LEACEConfig(
            device="cpu",
            force_cpu=True,
            regularization=1e-5,
            batch_size=50,
        )
        assert config.force_cpu is True
    
    def test_config_rejects_invalid_leace_device(self):
        """LEACE device must be 'cpu' or 'cuda'."""
        with pytest.raises(ConfigValidationError, match="device"):
            LEACEConfig(device="auto")  # LEACE doesn't support "auto"
    
    def test_config_rejects_tiny_regularization(self):
        """Regularization < 1e-10 is rejected."""
        with pytest.raises(ConfigValidationError, match="regularization"):
            LEACEConfig(regularization=1e-12)
    
    def test_config_rejects_invalid_compute_dtype(self):
        """Invalid compute_dtype is rejected."""
        with pytest.raises(ConfigValidationError, match="compute_dtype"):
            LEACEConfig(compute_dtype="float16")


# ==============================================================================
# Encoder Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestEncoderConfigValidation:
    """Test encoder configuration validation."""
    
    def test_valid_encoder_config(self):
        """Valid encoder config passes validation."""
        config = EncoderConfig(
            model="roberta-base",
            max_length=512,
            batch_size=32,
        )
        assert config.model == "roberta-base"
    
    def test_config_rejects_negative_max_length(self):
        """Negative max_length is rejected."""
        with pytest.raises(ConfigValidationError, match="max_length"):
            EncoderConfig(max_length=-1)


# ==============================================================================
# Subset Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestSubsetConfigValidation:
    """Test subset configuration validation."""
    
    def test_valid_subset_config_disabled(self):
        """Disabled subset config is valid."""
        config = SubsetConfig(enabled=False, size=None)
        assert config.enabled is False
    
    def test_valid_subset_config_enabled(self):
        """Enabled subset config with valid size is valid."""
        config = SubsetConfig(enabled=True, size=10000)
        assert config.size == 10000
    
    def test_config_rejects_negative_subset_size(self):
        """Negative subset size is rejected when enabled."""
        with pytest.raises(ConfigValidationError, match="size"):
            SubsetConfig(enabled=True, size=-100)


# ==============================================================================
# Probe Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestProbeConfigValidation:
    """Test probe configuration validation."""
    
    def test_valid_probe_config(self):
        """Valid probe config passes validation."""
        config = ProbeConfig(
            train_split=0.8,
            amnesic_drop_threshold=0.3,
        )
        assert config.train_split == 0.8
    
    def test_config_rejects_invalid_train_split_high(self):
        """train_split >= 1 is rejected."""
        with pytest.raises(ConfigValidationError, match="train_split"):
            ProbeConfig(train_split=1.0)
    
    def test_config_rejects_invalid_train_split_low(self):
        """train_split <= 0 is rejected."""
        with pytest.raises(ConfigValidationError, match="train_split"):
            ProbeConfig(train_split=0.0)


# ==============================================================================
# Full Pipeline Config Validation Tests
# ==============================================================================

@pytest.mark.unit
class TestPipelineConfigValidation:
    """Test full pipeline configuration validation."""
    
    def test_validate_valid_config(self):
        """Valid configuration passes validation."""
        config = {
            "seed": 42,
            "gliner": {
                "device": "auto",
                "batch_size": 32,
                "confidence_threshold": 0.85,
            },
            "encoder": {
                "model": "roberta-base",
                "batch_size": 32,
            },
            "leace": {
                "device": "cpu",
                "force_cpu": True,
                "regularization": 1e-5,
            },
            "subset": {
                "enabled": False,
            },
            "probe": {
                "train_split": 0.8,
            },
        }
        
        validated = validate_pipeline_config(config)
        assert validated["seed"] == 42
    
    def test_validate_rejects_invalid_gliner(self):
        """Invalid GLiNER config is rejected."""
        config = {
            "seed": 42,
            "gliner": {
                "confidence_threshold": 2.0,  # Invalid
            },
        }
        
        with pytest.raises(ConfigValidationError, match="gliner"):
            validate_pipeline_config(config)
    
    def test_validate_rejects_negative_seed(self):
        """Negative seed is rejected."""
        config = {
            "seed": -1,
        }
        
        with pytest.raises(ConfigValidationError, match="seed"):
            validate_pipeline_config(config)


# ==============================================================================
# Mode-Specific Override Tests
# ==============================================================================

@pytest.mark.unit
class TestModeSpecificOverrides:
    """Test mode-specific configuration overrides."""
    
    def test_laptop_mode_enables_force_cpu(self):
        """Laptop mode sets force_cpu=true."""
        config = load_pipeline_config(mode="laptop")
        
        assert config["leace"]["force_cpu"] is True, (
            "Laptop mode should set leace.force_cpu=true"
        )
    
    def test_laptop_mode_enables_subset(self):
        """Laptop mode enables dataset subset."""
        config = load_pipeline_config(mode="laptop")
        
        assert config["subset"]["enabled"] is True, (
            "Laptop mode should enable subset"
        )
    
    def test_hpc_mode_loads_successfully(self):
        """HPC mode config loads without error."""
        config = load_pipeline_config(mode="hpc")
        
        assert "gliner" in config
        assert "leace" in config
    
    def test_base_config_values_preserved(self):
        """Base config values are preserved when not overridden."""
        config = load_pipeline_config(mode="laptop")
        
        # Base config should define model name
        assert config["gliner"]["model"] == "urchade/gliner_large-v2.1"
        assert config["encoder"]["model"] == "roberta-base"


# ==============================================================================
# Experiment Config Merge Tests
# ==============================================================================

@pytest.mark.unit
class TestExperimentConfigMerge:
    """Test experiment configuration merging."""
    
    def test_experiment_config_overrides(self, tmp_path):
        """Experiment config overrides base+mode configs."""
        # Create temporary experiment config
        experiment_yaml = tmp_path / "experiment.yaml"
        experiment_yaml.write_text("""
seed: 123
gliner:
  batch_size: 64
""")
        
        config = load_pipeline_config(
            mode="laptop",
            experiment_config_path=experiment_yaml,
        )
        
        assert config["seed"] == 123
        assert config["gliner"]["batch_size"] == 64
    
    def test_nonexistent_experiment_config_ignored(self):
        """Non-existent experiment config path is handled gracefully."""
        # Should not raise
        config = load_pipeline_config(
            mode="laptop",
            experiment_config_path=Path("/nonexistent/path.yaml"),
        )
        
        # Should return valid config
        assert "gliner" in config


# ==============================================================================
# Validation Skip Tests
# ==============================================================================

@pytest.mark.unit
class TestValidationSkip:
    """Test validation can be skipped."""
    
    def test_validation_can_be_disabled(self, tmp_path):
        """Validation can be disabled to load invalid configs."""
        # Create invalid config
        invalid_yaml = tmp_path / "invalid.yaml"
        invalid_yaml.write_text("""
gliner:
  confidence_threshold: 999.0
""")
        
        # Should not raise when validation disabled
        config = load_pipeline_config(
            mode="laptop",
            experiment_config_path=invalid_yaml,
            validate=False,
        )
        
        # Invalid value is loaded
        assert config["gliner"]["confidence_threshold"] == 999.0
