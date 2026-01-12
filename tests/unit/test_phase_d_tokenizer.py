# tests/unit/test_phase_d_tokenizer.py
import pytest

from neuro_stylometry.stylometry_net.tokenizer import PhaseDTokenizer


@pytest.mark.unit
@pytest.mark.real_models
def test_phase_d_tokenizer_aligns_mask_tokens(require_real_models):
    tokenizer = PhaseDTokenizer(
        model_name="roberta-base",
        taxonomy_path="conf/base/gliner_taxonomy.yaml",
    )

    assert tokenizer.mask_tokens, "Expected mask tokens from taxonomy"

    for token in tokenizer.mask_tokens:
        token_ids = tokenizer.tokenizer.encode(token, add_special_tokens=False)
        assert len(token_ids) == 1, f"{token} encoded as {len(token_ids)} tokens"
