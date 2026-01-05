from neuro_stylometry.phase_a_pipeline import PhaseAPipeline
from neuro_stylometry.factories.strategy_factory import StrategyFactory


def test_facade_detector_and_masker_real_models(require_real_models):
    # Strict no-mocking: loads real models and runs a tiny inference.
    strategy = StrategyFactory.create_filter_strategy()
    pipeline = PhaseAPipeline(strategy=strategy)

    detector = pipeline.get_detector()
    masker = pipeline.get_masker()

    texts = [
        "I'm 25 and I was born in 1999.",
        "As a woman from Canada, I love hiking.",
    ]
    post_ids = ["t1", "t2"]

    entities = detector.detect_spans_long(texts, batch_size=int(pipeline.config["gliner"]["batch_size"]))
    masked, logs = masker.mask_batch(texts, entities, post_ids)

    assert len(masked) == len(texts)
    assert all(isinstance(t, str) for t in masked)
    assert all("post_id" in row for row in logs) if logs else True