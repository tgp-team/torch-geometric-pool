import pytest
import torch

from tgp.reduce import BaseReduce, Reduce
from tgp.select import SelectOutput


def test_reduce_batch_dense_multi_graph():
    """Test that dense multi-graph batches are now supported with the [N, K] representation."""
    s = torch.randn(4, 2)
    so = SelectOutput(s=s)
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)

    # Dense multi-graph batch is now supported
    out = BaseReduce.reduce_batch(so, batch)

    # Should produce B*K supernodes: 2 graphs * 2 clusters = 4 supernodes
    K = so.num_supernodes
    batch_size = 2
    assert out.shape == (batch_size * K,)
    # Check batch assignment: first K supernodes belong to graph 0, next K to graph 1
    assert torch.all(out[:K] == 0)
    assert torch.all(out[K:] == 1)


def test_reduce_batch_dense_single_graph_returns():
    s = torch.randn(3, 2)
    so = SelectOutput(s=s)
    batch = torch.zeros(3, dtype=torch.long)

    out = BaseReduce.reduce_batch(so, batch)

    # Single graph: K supernodes all belonging to graph 0
    K = so.num_supernodes
    assert out.shape == (K,)
    assert torch.all(out == 0)


def test_reduce_batch_dense_empty_batch():
    """Test that empty batches are handled gracefully."""
    s = torch.empty((0, 2))
    so = SelectOutput(s=s)
    batch = torch.empty((0,), dtype=torch.long)

    # Empty batch should return empty tensor
    out = BaseReduce.reduce_batch(so, batch)
    assert out.shape == (0,)


def test_base_reduce_forward_uses_select_output_batch_when_batch_is_none():
    x = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [2.0, 0.0],
            [0.0, 2.0],
        ]
    )
    s = torch.tensor(
        [
            [1.0, 0.0],
            [0.0, 1.0],
            [1.0, 0.0],
            [0.0, 1.0],
        ]
    )
    so = SelectOutput(s=s, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))

    x_pool, batch_pool = BaseReduce()(x, so, batch=None)

    assert x_pool.shape == (4, 2)  # [B*K, F] with B=2 and K=2
    assert torch.equal(batch_pool, torch.tensor([0, 0, 1, 1], dtype=torch.long))


def test_base_reduce_forward_sparse_rejects_return_batched():
    x = torch.randn(3, 2)
    indices = torch.tensor([[0, 1, 2], [0, 0, 1]], dtype=torch.long)
    s = torch.sparse_coo_tensor(indices, torch.ones(3), size=(3, 2)).coalesce()
    so = SelectOutput(s=s)

    with pytest.raises(ValueError, match="return_batched=True is only supported"):
        BaseReduce()(x, so, return_batched=True)


def test_base_reduce_forward_rejects_invalid_dense_assignment_rank():
    x = torch.randn(2, 3)
    so = SelectOutput(s=torch.randn(1, 2, 3, 4))

    with pytest.raises(ValueError, match="Dense SelectOutput.s must be 2D \\[N, K\\]"):
        BaseReduce()(x, so)


def test_base_reduce_forward_dense_batched_assignment():
    x = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[2.0, 0.0], [0.0, 2.0]],
        ]
    )  # [B=2, N=2, F=2]
    s = torch.tensor(
        [
            [[1.0, 0.0], [0.0, 1.0]],
            [[1.0, 0.0], [0.0, 1.0]],
        ]
    )  # [B=2, N=2, K=2]
    so = SelectOutput(s=s)

    x_pool, batch_pool = BaseReduce()(x, so, batch=None)

    assert x_pool.shape == (2, 2, 2)
    assert batch_pool is None


def test_base_reduce_forward_dense_single_graph_return_batched():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    so = SelectOutput(s=s)

    x_pool, batch_pool = BaseReduce()(x, so, return_batched=True)

    assert x_pool.shape == (1, 2, 2)
    assert batch_pool is None


def test_base_reduce_forward_dense_single_graph_flat_output():
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]])
    s = torch.tensor([[1.0, 0.0], [0.0, 1.0]])
    so = SelectOutput(s=s)

    x_pool, batch_pool = BaseReduce()(x, so, return_batched=False)

    assert x_pool.shape == (2, 2)
    assert batch_pool is None


def test_reduce_and_base_reduce_repr():
    assert repr(Reduce()) == "Reduce()"
    assert repr(BaseReduce()) == "BaseReduce()"
