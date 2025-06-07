import pytest
import torch

from tgp.reduce import dense_global_reduce, global_reduce


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min", "any"])
def test_dense_global_reduce_all_ops(reduce_op):
    # Create a dense batch of shape [B=2, N=3, F=2]
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # graph 0
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],  # graph 1
        ],
        dtype=torch.float,
    )
    # For each reduce_op, call dense_global_reduce and compare against manual
    out = dense_global_reduce(x, reduce_op=reduce_op, node_dim=-2)
    assert out.shape == (2, 2)  # [B, F]

    if reduce_op == "sum":
        expected0 = torch.tensor([1.0 + 3.0 + 5.0, 2.0 + 4.0 + 6.0])
        expected1 = torch.tensor([-1.0 + 0.0 + 2.0, 0.0 + 1.0 + (-2.0)])
    elif reduce_op == "mean":
        expected0 = torch.tensor([(1.0 + 3.0 + 5.0) / 3.0, (2.0 + 4.0 + 6.0) / 3.0])
        expected1 = torch.tensor([(-1.0 + 0.0 + 2.0) / 3.0, (0.0 + 1.0 + (-2.0)) / 3.0])
    elif reduce_op == "max":
        expected0 = torch.tensor([5.0, 6.0])
        expected1 = torch.tensor([2.0, 1.0])
    elif reduce_op == "min":
        expected0 = torch.tensor([1.0, 2.0])
        expected1 = torch.tensor([-1.0, -2.0])
    else:  # reduce_op == "any"
        expected0 = torch.tensor([1.0, 1.0])
        expected1 = torch.tensor([1.0, 1.0])

    expected = torch.stack([expected0, expected1], dim=0)
    assert torch.equal(out, expected)
    
    # test also invalid reduce_op
    with pytest.raises(ValueError):
        _ = dense_global_reduce(x, reduce_op="invalid", node_dim=-2)
        
def test_global_reduce():
    # Create a dense batch of shape [N=4, F=2]
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)  # Two graphs in the batch

    out = global_reduce(x, reduce_op="sum", batch=batch, node_dim=-2)
    assert out.shape == (2, 2)  # [B, F]

if __name__ == "__main__":
    pytest.main([__file__])
