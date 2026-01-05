import numpy as np
import pyarrow as pa

from neuro_stylometry.data_engine.schemas import SOBR_SCHEMA, get_demographic_columns
from neuro_stylometry.pollution_guard.concept_encoding import DemographicEncoder, extract_probe_labels


def _make_table():
    rows = [
        {
            "post_id": "1",
            "author_id": "a",
            "post": "x",
            "post_masked": "x",
            "birth_year": 1980,
            "female": 1,
            "nationality": "UK",
            "political_leaning": "left",
            "extrovert": 1,
            "sensing": 0,
            "feeling": 1,
            "judging": 0,
            "split": "train",
            "text_length": 10,
        },
        {
            "post_id": "2",
            "author_id": "b",
            "post": "y",
            "post_masked": "y",
            "birth_year": None,
            "female": None,
            "nationality": "US",
            "political_leaning": None,
            "extrovert": 0,
            "sensing": 1,
            "feeling": 0,
            "judging": 1,
            "split": "train",
            "text_length": 20,
        },
        {
            "post_id": "3",
            "author_id": "c",
            "post": "z",
            "post_masked": "z",
            "birth_year": 1990,
            "female": 0,
            "nationality": None,
            "political_leaning": "right",
            "extrovert": None,
            "sensing": None,
            "feeling": None,
            "judging": None,
            "split": "train",
            "text_length": 30,
        },
    ]
    # Build using schema to match project contracts.
    return pa.Table.from_pylist(rows, schema=SOBR_SCHEMA)


def test_demographic_encoder_consistent_feature_dim_across_slices():
    table = _make_table()
    encoder = DemographicEncoder(get_demographic_columns()).fit(table)

    z_full = encoder.transform(table)
    z_a = encoder.transform(table.slice(0, 1))
    z_b = encoder.transform(table.slice(1, 2))

    assert z_full.shape[1] == encoder.feature_dim
    assert z_a.shape[1] == encoder.feature_dim
    assert z_b.shape[1] == encoder.feature_dim


def test_demographic_encoder_missing_indicators_present():
    table = _make_table()
    encoder = DemographicEncoder(get_demographic_columns()).fit(table)
    z = encoder.transform(table)

    # At least one row has missingness, so some missing indicator should be 1.
    assert float(z.max()) >= 1.0


def test_extract_probe_labels_stable_and_missing_as_minus_one():
    table = _make_table()

    y_nat = extract_probe_labels(table, "nationality")
    assert y_nat.shape == (len(table),)
    assert -1 in y_nat

    y_birth = extract_probe_labels(table, "birth_year")
    assert y_birth.shape == (len(table),)
    assert -1 in y_birth
