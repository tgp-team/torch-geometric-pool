import torch

from tgp.reduce import BaseReduce
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
