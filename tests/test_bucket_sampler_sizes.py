import numpy as np
import pytest

from nainsight.data.dataset import LengthBucketSampler


@pytest.mark.parametrize("size, multiplier", [(0, 2), (-1, 2), (True, 2), (2, 0), (2, -1), (2, 1.5)])
def test_invalid_bucket_dimensions_fail_before_iteration(size, multiplier):
    with pytest.raises(ValueError, match="positive integer"):
        LengthBucketSampler(np.array([1, 2, 3]), size, mult=multiplier)


def test_numpy_integer_sizes_produce_complete_batches():
    sampler = LengthBucketSampler(np.array([3, 1, 2, 5, 4]), np.int64(2), shuffle=False, mult=np.int64(2))
    batches = list(sampler)
    assert len(batches) == len(sampler) == 3
    assert sorted(index for batch in batches for index in batch) == list(range(5))
