import pytest
import torch

from tgp.reduce import AggrReduce, BaseReduce
from tgp.select import SelectOutput


def _make_sparse_select_output(num_nodes=6, num_supernodes=3):
    """Minimal sparse SelectOutput for testing."""
    # node_index (row): which nodes contribute; cluster_index (col): to which supernode
    node_index = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    cluster_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    indices = torch.stack([node_index, cluster_index], dim=0)
    values = torch.ones(6, dtype=torch.float)
    s = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_supernodes)
    ).coalesce()
    return SelectOutput(s=s)


@pytest.mark.parametrize("reduce_op", ["sum", "mean"])
def test_aggr_reduce_vs_base_reduce_sparse(reduce_op):
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    if reduce_op == "sum":
        aggr_module = aggr.SumAggregation()
    else:
        aggr_module = aggr.MeanAggregation()

    so = _make_sparse_select_output()
    x = torch.randn(6, 4)  # 6 nodes, 4 features
    batch = None

    aggr_reduce = AggrReduce(aggr_module)
    x_aggr, batch_aggr = aggr_reduce(x, so, batch=batch)

    assert x_aggr.shape == (3, 4)
    if reduce_op == "sum":
        base = BaseReduce()
        x_base, batch_base = base(x, so, batch=batch)
        assert torch.allclose(x_base, x_aggr)
        assert batch_base == batch_aggr


def test_aggr_reduce_sparse_with_weights():
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    node_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    values = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    indices = torch.stack([node_index, cluster_index], dim=0)
    s = torch.sparse_coo_tensor(indices, values, size=(4, 2)).coalesce()
    so = SelectOutput(s=s)
    x = torch.ones(4, 2)
    base = BaseReduce()
    aggr_reduce = AggrReduce(aggr.SumAggregation())
    x_base, _ = base(x, so, batch=None)
    x_aggr, _ = aggr_reduce(x, so, batch=None)
    assert torch.allclose(x_base, x_aggr)


def test_aggr_reduce_ignores_mask_for_2d_x_with_warning():
    """AggrReduce warns and ignores mask when x is 2D (unbatched); all nodes valid."""
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    so = _make_sparse_select_output()
    x = torch.randn(6, 4)  # 2D
    mask = torch.ones(6, dtype=torch.bool)
    reducer = AggrReduce(aggr.SumAggregation())
    with pytest.warns(UserWarning, match="mask is only supported for batched"):
        out, _ = reducer(x, so=so, batch=None, mask=mask)
    expected, _ = reducer(x, so=so, batch=None, mask=None)
    assert torch.allclose(out, expected)


def test_aggr_reduce_readout_dense_with_mask():
    """AggrReduce with so=None and dense x applies mask (readout path)."""
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    # x [B=2, N=3, F=2]; mask zeros out one node per graph
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    mask = torch.tensor([[True, True, False], [True, False, True]], dtype=torch.bool)
    reducer = AggrReduce(aggr.SumAggregation())
    out, _ = reducer(x, so=None, size=2, mask=mask)
    assert out.shape == (2, 2)
    # Graph 0: nodes 0,1 -> [4, 6]; Graph 1: nodes 0,2 -> [1, -2]
    expected = torch.tensor([[4.0, 6.0], [1.0, -2.0]])
    assert torch.allclose(out, expected)


def test_aggr_reduce_dense_with_so_in_mask():
    """AggrReduce uses so.in_mask (batched dense only, shape [B, N]) when mask argument is None."""
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    # Batched dense: B=2, N=3, K=2; in_mask zeros out one node per graph
    B, N, K = 2, 3, 2
    s = torch.zeros(B, N, K)
    s[0, 0, 0] = s[0, 1, 0] = 1.0  # graph 0: nodes 0,1 -> cluster 0
    s[0, 2, 1] = 1.0  # graph 0: node 2 -> cluster 1
    s[1, 0, 0] = 1.0  # graph 1: node 0 -> cluster 0
    s[1, 1, 1] = s[1, 2, 1] = 1.0  # graph 1: nodes 1,2 -> cluster 1
    so = SelectOutput(s=s)
    so.in_mask = torch.tensor(
        [[True, True, False], [True, False, True]], dtype=torch.bool
    )
    x = torch.tensor(
        [
            [[1.0, 2.0], [3.0, 4.0], [5.0, 6.0]],
            [[-1.0, 0.0], [0.0, 1.0], [2.0, -2.0]],
        ],
        dtype=torch.float,
    )
    reducer = AggrReduce(aggr.SumAggregation())
    out, _ = reducer(x, so=so, batch=None, mask=None)
    assert out.shape == (B, K, 2)
    # Graph 0: nodes 0,1 (masked) -> cluster 0 sum [4, 6]; node 2 masked out
    # Graph 1: node 0 -> cluster 0 [-1, 0]; nodes 1,2 (node 1 masked) -> cluster 1 [2, -2]
    assert torch.allclose(out[0, 0], torch.tensor([4.0, 6.0]))
    assert torch.allclose(out[0, 1], torch.tensor([0.0, 0.0]))
    assert torch.allclose(out[1, 0], torch.tensor([-1.0, 0.0]))
    assert torch.allclose(out[1, 1], torch.tensor([2.0, -2.0]))


if __name__ == "__main__":
    pytest.main([__file__])
