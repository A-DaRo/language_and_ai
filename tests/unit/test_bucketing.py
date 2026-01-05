import numpy as np
from datasets import Dataset

from neuro_stylometry.data_engine.bucketing import BucketSampler, compute_bucket_boundaries


def test_compute_bucket_boundaries_monotonic_and_unique():
    lengths = np.array([10, 11, 12, 20, 25, 30, 64, 128, 256, 512], dtype=np.int32)
    boundaries = compute_bucket_boundaries(lengths, num_buckets=5)

    assert boundaries == sorted(boundaries)
    assert len(boundaries) == len(set(boundaries))
    assert all(b > 0 for b in boundaries)


def test_bucket_sampler_yields_all_indices_exactly_once():
    lengths = [10, 12, 50, 60, 200, 220, 512, 520]
    ds = Dataset.from_dict({"text_length": lengths})

    sampler = BucketSampler(ds, batch_size=2, num_buckets=4, shuffle=False, seed=123)

    seen = []
    for batch in sampler:
        seen.extend(batch)

    assert sorted(seen) == list(range(len(lengths)))


def test_bucket_sampler_epoch_changes_order_when_shuffling():
    lengths = [10, 12, 50, 60, 200, 220, 512, 520]
    ds = Dataset.from_dict({"text_length": lengths})

    sampler = BucketSampler(ds, batch_size=2, num_buckets=4, shuffle=True, seed=123)

    sampler.set_epoch(0)
    first = list(sampler)

    sampler.set_epoch(1)
    second = list(sampler)

    assert first != second
