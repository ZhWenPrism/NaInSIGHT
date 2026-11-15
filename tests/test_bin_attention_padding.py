import torch

from nainsight.model.interaction import CausalBinCrossAttention


def test_padded_nonfinite_values_do_not_reach_real_tokens():
    torch.manual_seed(5)
    layer = CausalBinCrossAttention(4, 2, resolution_h=1, dropout=0).eval()
    tokens = torch.randn(1, 2, 4)
    time = torch.tensor([[0., 1.]])
    expected = layer(tokens, tokens, time, torch.ones(1, 2, dtype=torch.bool))
    padded = torch.cat([tokens, torch.full((1, 1, 4), float("nan"))], dim=1)
    actual = layer(padded, padded, torch.tensor([[0., 1., float("nan")]]), torch.tensor([[True, True, False]]))
    torch.testing.assert_close(actual[:, :2], expected)
    torch.testing.assert_close(actual[:, 2], torch.zeros((1, 4)))
