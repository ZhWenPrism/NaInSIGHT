import pytest
import torch

from nainsight.model.transformer import TimeBucketBias


@pytest.mark.parametrize("bounds", [[2, 1], [1, 1], [float("nan")], [-1, 2], [[1, 2]]])
def test_time_bucket_rejects_ambiguous_boundaries(bounds):
    with pytest.raises(ValueError, match="bucket boundaries"):
        TimeBucketBias(2, bounds)


def test_valid_boundaries_preserve_bucket_edges():
    layer = TimeBucketBias(1, [1, 2])
    with torch.no_grad():
        layer.bias.weight[:, 0] = torch.tensor([0.0, 1.0, 2.0])
    actual = layer(torch.tensor([[0., 1., 2., 3.]]))[0, 0, 0]
    torch.testing.assert_close(actual, torch.tensor([0., 0., 1., 2.]))
