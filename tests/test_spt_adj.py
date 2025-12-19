import pytest
import torch
from torch_geometric.utils import add_self_loops

from tgp.poolers import get_pooler, pooler_map


@pytest.fixture(scope="module")
def simple_graph():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    N = 10
    F = 2
    row = torch.arange(9, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    E = edge_index.size(1)
    x = torch.randn((N, F), dtype=torch.float)
    edge_weight = torch.ones((E), dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    batch = torch.zeros(N, dtype=torch.long)
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
    return x, adj, edge_weight, batch


poolers = list(pooler_map.keys())


@pytest.mark.parametrize("pooler_name", poolers)
@pytest.mark.torch_sparse
def test_output_with_spt_adj(simple_graph, pooler_name):
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    x, adj, edge_weight, batch = simple_graph
    N, F = x.size()

    PARAMS = {
        "in_channels": F,
        "ratio": 0.5,
        "k": max(1, N // 2),
        "cached": True,
        "lift": "precomputed",
        "s_inv_op": "transpose",
        "lift_red_op": "mean",
        "loss_coeff": 1.0,
        "remove_self_loops": True,
        "scorer": "degree",
        "reduce": "sum",
    }
    pooler = get_pooler(pooler_name, **PARAMS)
    pooler.eval()

    # Preprocessing data
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=adj, edge_weight=edge_weight, x=x, batch=batch, use_cache=False
    )
    if pooler.is_dense:
        assert isinstance(adj_pre, torch.Tensor) and adj_pre.ndim == 3
    else:
        assert isinstance(adj_pre, SparseTensor)

    # Pooling operation
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    if pooler.is_dense:
        assert isinstance(out.edge_index, torch.Tensor)
    else:
        # edge_index should be either SparseTensor or torch COO tensor
        assert isinstance(out.edge_index, (SparseTensor, torch.Tensor))
        if isinstance(out.edge_index, torch.Tensor) and not out.edge_index.is_sparse:
            assert out.edge_index.shape[0] == 2

    # lift
    x_pool = out.x.clone()
    x_lifted = pooler(x=x_pool, so=out.so, lifting=True)
    assert isinstance(x_lifted, torch.Tensor)
    assert x_lifted.size(-2) == N
    assert x_lifted.size(-1) == x_pool.size(-1)

    # reset params check
    if hasattr(pooler, "reset_parameters"):
        pooler.reset_parameters()

    # repr check
    rep = repr(pooler)
    assert pooler_map[pooler_name].__name__.lower() in rep.lower()


if __name__ == "__main__":
    pytest.main([__file__])
