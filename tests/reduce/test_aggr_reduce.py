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

    base = BaseReduce(reduce_op=reduce_op)
    aggr_reduce = AggrReduce(aggr_module)

    x_base, batch_base = base(x, so, batch=batch)
    x_aggr, batch_aggr = aggr_reduce(x, so, batch=batch)

    assert x_base.shape == x_aggr.shape == (3, 4)
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
    base = BaseReduce(reduce_op="sum")
    aggr_reduce = AggrReduce(aggr.SumAggregation())
    x_base, _ = base(x, so, batch=None)
    x_aggr, _ = aggr_reduce(x, so, batch=None)
    assert torch.allclose(x_base, x_aggr)


if __name__ == "__main__":
    pytest.main([__file__])
