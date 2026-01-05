import pytest

from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
from neuro_stylometry.pollution_guard.strategies.laptop import LaptopFilterStrategy
from neuro_stylometry.config import load_pipeline_config


def test_load_pipeline_config_has_phase_a_keys():
    cfg = load_pipeline_config(mode="laptop")
    assert isinstance(cfg, dict)
    assert "gliner" in cfg and "encoder" in cfg and "leace" in cfg
    assert "model" in cfg["gliner"]
    assert "taxonomy_path" in cfg["gliner"]
    assert "regularization" in cfg["leace"]
    assert "batch_size" in cfg["leace"]


def test_facade_requires_config_keys_when_user_config_provided():
    # Missing required nested keys should fail fast.
    strategy = LaptopFilterStrategy()
    pipeline = PhaseAPipeline(strategy=strategy, config={"gliner": {}})

    with pytest.raises(KeyError):
        _ = pipeline.get_detector()


def test_facade_accepts_default_yaml_when_config_none():
    strategy = LaptopFilterStrategy()
    pipeline = PhaseAPipeline(strategy=strategy, config=None)
    assert isinstance(pipeline.config, dict)
    assert "gliner" in pipeline.config