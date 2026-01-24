import pytest
import torch

from tgp.data import PoolCollater


def test_non_base_data():
    collater = PoolCollater(dataset=[], follow_batch=None, exclude_keys=None)

    result = collater([[42], [99], [123]])  # batch[0] = 42 → not a BaseData
    assert torch.equal(result[0], torch.tensor([42, 99, 123]))


if __name__ == "__main__":
    pytest.main([__file__])
