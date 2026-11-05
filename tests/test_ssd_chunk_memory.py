from unittest.mock import patch

import pytest
import torch

from nainsight.model.ssm import segsum, sequential_scan, ssd_scan


@pytest.mark.parametrize("device", ["cpu", "cuda"])
@pytest.mark.parametrize("chunk", [1, 4, 16])
def test_chunk_carry_matches_reference_values_and_gradients(device, chunk):
    if device == "cuda" and not torch.cuda.is_available():
        pytest.skip("CUDA unavailable")
    generator = torch.Generator().manual_seed(7)
    tensors = [torch.randn(2, 19, 2, 3, generator=generator, dtype=torch.float64),
               -torch.rand(2, 19, 2, generator=generator, dtype=torch.float64),
               torch.randn(2, 19, 2, 4, generator=generator, dtype=torch.float64),
               torch.randn(2, 19, 2, 4, generator=generator, dtype=torch.float64)]
    inputs = [t.to(device).requires_grad_() for t in tensors]
    actual = ssd_scan(*inputs, chunk)
    expected = sequential_scan(*inputs)
    torch.testing.assert_close(actual, expected, atol=1e-10, rtol=1e-10)
    weights = torch.linspace(0.1, 1., actual.numel(), device=device).reshape_as(actual)
    grad_actual = torch.autograd.grad((actual * weights).sum(), inputs, retain_graph=True)
    grad_expected = torch.autograd.grad((expected * weights).sum(), inputs)
    for left, right in zip(grad_actual, grad_expected):
        torch.testing.assert_close(left, right, atol=1e-9, rtol=1e-9)


def test_sequence_length_does_not_expand_the_quadratic_chunk_work():
    x = torch.ones(1, 1025, 1, 1)
    a = torch.full((1, 1025, 1), -0.1)
    sizes = []

    def bounded_segsum(value):
        sizes.append(value.shape[-1])
        assert value.shape[-1] <= 16
        return segsum(value)

    with patch("nainsight.model.ssm.segsum", side_effect=bounded_segsum):
        actual = ssd_scan(x, a, x, x, 16)
    torch.testing.assert_close(actual, sequential_scan(x, a, x, x), rtol=1e-5, atol=1e-5)
    assert sizes == [16]


def test_empty_sequence_and_invalid_chunk():
    x = torch.empty(1, 0, 1, 1)
    assert ssd_scan(x, x[..., 0], x, x, 4).shape == x.shape
    for chunk in (0, -1, 1.5, True):
        with pytest.raises(ValueError, match="chunk"):
            ssd_scan(x, x[..., 0], x, x, chunk)
