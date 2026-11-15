from unittest.mock import patch

import torch

from nainsight.model.ssm import segsum, sequential_scan, ssd_scan


def test_short_scan_matches_recurrence_without_padding_the_chunk():
    g = torch.Generator().manual_seed(14)
    x, b, c = [torch.randn(1, 3, 2, 2, generator=g, dtype=torch.float64) for _ in range(3)]
    a = -torch.rand(1, 3, 2, generator=g, dtype=torch.float64)
    values = [t.requires_grad_() for t in (x, a, b, c)]
    with patch("nainsight.model.ssm.segsum", wraps=segsum) as summed:
        actual = ssd_scan(*values, chunk=64)
    expected = sequential_scan(*values)
    torch.testing.assert_close(actual, expected)
    for gradient, reference in zip(torch.autograd.grad(actual.sum(), values), torch.autograd.grad(expected.sum(), values)):
        torch.testing.assert_close(gradient, reference)
    assert summed.call_args.args[0].shape[-1] == 3
