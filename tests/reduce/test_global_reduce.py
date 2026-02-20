import pytest
import torch

from tgp.reduce import readout


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min", "any"])
def test_readout_dense_all_ops(reduce_op):
    # Dense: x [B=2, N=3, F=2] -> readout infers dense
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],  # graph 0
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],  # graph 1
        ],
        dtype=torch.float,
    )
    out = readout(x, reduce_op=reduce_op)
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

    with pytest.raises(ValueError, match="Unknown aggregator alias|invalid"):
        readout(x, reduce_op="invalid")


def test_readout_sparse():
    # Sparse: x [N=4, F=2], batch -> readout infers sparse
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = readout(x, reduce_op="sum", batch=batch)
    assert out.shape == (2, 2)  # [B, F]


def test_readout_sparse_single_graph():
    # Sparse with batch=None -> single graph, output [1, F]
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    out = readout(x, reduce_op="sum", batch=None)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[9.0, 12.0]]))


def test_readout_sparse_with_size():
    # Sparse with explicit size
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = readout(x, reduce_op="sum", batch=batch, size=2)
    assert out.shape == (2, 2)


@pytest.mark.parametrize("reduce_op", ["mean", "max", "min", "sum"])
def test_readout_sparse_with_mask(reduce_op):
    # Sparse with mask: only aggregate valid nodes
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    mask = torch.tensor([True, True, False, True], dtype=torch.bool)  # node 2 masked
    out = readout(x, reduce_op=reduce_op, batch=batch, mask=mask)
    assert out.shape == (2, 2)
    if reduce_op == "sum":
        # Graph 0: nodes 0,1 -> 1+3, 2+4 = [4, 6]; Graph 1: node 3 only -> [7, 8]
        expected = torch.tensor([[4.0, 6.0], [7.0, 8.0]])
        assert torch.allclose(out, expected)
    elif reduce_op == "mean":
        # Graph 0: mean of 2 nodes; Graph 1: 1 node
        expected = torch.tensor([[2.0, 3.0], [7.0, 8.0]])
        assert torch.allclose(out, expected)


def test_readout_invalid_ndim():
    x = torch.randn(2, 3, 4, 5)  # 4D
    with pytest.raises(ValueError, match="2D.*3D"):
        readout(x, reduce_op="sum")


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max", "min", "any"])
def test_readout_dense_with_mask(reduce_op):
    # Dense with mask: [B=2, N=3, F=2], mask zeros out one node per graph
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor(
        [[True, True, False], [True, False, True]], dtype=torch.bool
    )  # (2, 3)
    out = readout(x, reduce_op=reduce_op, mask=mask)
    assert out.shape == (2, 2)
    if reduce_op == "sum":
        # Graph 0: nodes 0,1 -> [4, 6]; Graph 1: nodes 0,2 -> [1, -2]
        assert torch.allclose(out, torch.tensor([[4.0, 6.0], [1.0, -2.0]]))
    elif reduce_op == "mean":
        # Graph 0: mean of 2 nodes [2, 3]; Graph 1: mean of 2 nodes [0.5, -1]
        assert torch.allclose(out, torch.tensor([[2.0, 3.0], [0.5, -1.0]]))
    elif reduce_op == "max":
        # Graph 0: max over nodes 0,1 -> [3, 4]; Graph 1: max over nodes 0,2 -> [2, 0]
        assert torch.allclose(out, torch.tensor([[3.0, 4.0], [2.0, 0.0]]))
    elif reduce_op == "min":
        # Graph 0: min over nodes 0,1 -> [1, 2]; Graph 1: min over nodes 0,2 -> [-1, -2]
        assert torch.allclose(out, torch.tensor([[1.0, 2.0], [-1.0, -2.0]]))
    else:  # any: both graphs have at least one True
        assert out.dtype == torch.bool
        assert out.shape == (2, 2)


@pytest.mark.parametrize("reduce_op", ["sum", "mean"])
def test_readout_pyg_aggr(reduce_op):
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    if reduce_op == "sum":
        aggr_module = aggr.SumAggregation()
    else:
        aggr_module = aggr.MeanAggregation()
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = readout(x, reduce_op=aggr_module, batch=batch)
    assert out.shape == (2, 2)
    # Compare to string path
    out_str = readout(x, reduce_op=reduce_op, batch=batch)
    assert torch.allclose(out, out_str)


def test_readout_pyg_aggr_sparse_single_graph():
    # PyG aggr with batch=None (single graph)
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0]], dtype=torch.float)
    out = readout(x, reduce_op=aggr.SumAggregation(), batch=None)
    assert out.shape == (1, 2)
    assert torch.allclose(out, torch.tensor([[4.0, 6.0]]))


def test_readout_pyg_aggr_sparse_with_mask():
    # Sparse + PyG aggr + mask
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor([[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]], dtype=torch.float)
    batch = torch.tensor([0, 0, 1], dtype=torch.long)
    mask = torch.tensor([True, False, True], dtype=torch.bool)
    out = readout(x, reduce_op=aggr.SumAggregation(), batch=batch, mask=mask)
    assert out.shape == (2, 2)
    # Graph 0: only node 0 -> [1, 2]; Graph 1: node 2 -> [5, 6]
    assert torch.allclose(out, torch.tensor([[1.0, 2.0], [5.0, 6.0]]))


def test_readout_pyg_aggr_dense_with_mask():
    # Dense + PyG aggr + mask
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor([[True, True, False], [False, True, True]], dtype=torch.bool)
    out = readout(x, reduce_op=aggr.MeanAggregation(), mask=mask)
    assert out.shape == (2, 2)
    # Graph 0: mean of nodes 0,1 -> [2, 3]; Graph 1: mean of nodes 1,2 -> [1, -0.5]
    expected = torch.tensor([[2.0, 3.0], [1.0, -0.5]])
    assert torch.allclose(out, expected)


def test_readout_node_dim():
    # node_dim is -2 by default; exercise it explicitly
    x = torch.tensor(
        [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0], [7.0, 8.0]], dtype=torch.float
    )
    batch = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    out = readout(x, reduce_op="sum", batch=batch, node_dim=-2)
    assert out.shape == (2, 2)


if __name__ == "__main__":
    pytest.main([__file__])
