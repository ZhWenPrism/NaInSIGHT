import pytest
import torch

from nainsight.model.prior_adapter import PatientConditionedGraphPriorAdapter


def make_adapter(n_heads, top_k):
    return PatientConditionedGraphPriorAdapter(
        8, torch.zeros(3, 4), torch.tensor([0, 1, 1]), torch.ones(3),
        2, n_heads, top_k, -1.0, False, 0.0, None,
    )


@pytest.mark.parametrize("n_heads", [0, 3, True])
def test_invalid_prior_attention_head_count_fails_at_construction(n_heads):
    with pytest.raises(ValueError, match="n_heads"):
        make_adapter(n_heads, 2)


@pytest.mark.parametrize("top_k", [0, -1, 1.5, True])
def test_invalid_prior_attention_top_k_fails_at_construction(top_k):
    with pytest.raises(ValueError, match="top_k"):
        make_adapter(2, top_k)


@pytest.mark.parametrize("top_k", [1, 4, None])
def test_valid_prior_attention_settings_run_forward(top_k):
    adapter = make_adapter(2, top_k)
    out, aux = adapter(torch.zeros(2, 8), torch.zeros(2, 3, 2))

    assert out.shape == (2, 8)
    torch.testing.assert_close(aux["prior_attention"].sum(-1), torch.ones(2))
