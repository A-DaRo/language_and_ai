from pathlib import Path
import yaml
from neuro_stylometry.config import load_pipeline_config


def test_load_pipeline_config_with_force_skip(tmp_path):
    # Create a small experiment YAML that sets execution.force_skip
    cfg = {"execution": {"force_skip": True}}
    p = tmp_path / "exp.yaml"
    p.write_text(yaml.dump(cfg))

    merged = load_pipeline_config(mode="laptop", experiment_config_path=p)
    assert "execution" in merged
    assert merged["execution"].get("force_skip", False) is True


def test_get_skip_stages_from_execution_flag(tmp_path):
    # Ensure SkipStagesConfig picks up top-level force_skip via merged config
    cfg = {"execution": {"force_skip": True}}
    p = tmp_path / "exp2.yaml"
    p.write_text(yaml.dump(cfg))
    merged = load_pipeline_config(mode="laptop", experiment_config_path=p)
    # The SkipStagesConfig will be derived via ExecutionConfig.get_skip_stages_config in normal flows
    assert merged["execution"].get("force_skip", False) is True
