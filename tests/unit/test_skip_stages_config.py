from neuro_stylometry.pollution_guard.strategies.base import SkipStagesConfig


def test_skip_stages_from_config_reads_force_skip():
    cfg = {"execution": {"force_skip": True}}
    skip_cfg = SkipStagesConfig.from_config(cfg)
    assert skip_cfg.force_skip is True


def test_skip_stages_from_config_reads_skip_flags():
    cfg = {"execution": {"skip_stages": {"skip_chunking": True}}}
    skip_cfg = SkipStagesConfig.from_config(cfg)
    assert skip_cfg.skip_chunking is True
    assert skip_cfg.skip_inference is False
