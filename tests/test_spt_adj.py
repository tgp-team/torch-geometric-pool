import pytest
import torch

from tgp.poolers import get_pooler, pooler_map

poolers = list(pooler_map.keys())


@pytest.mark.parametrize("pooler_name", poolers)
@pytest.mark.torch_sparse
def test_output_with_spt_adj(pooler_test_graph_sparse_spt, pooler_name):
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    x, adj, edge_weight, batch = pooler_test_graph_sparse_spt
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
    if pooler.is_dense_batched:
        assert isinstance(adj_pre, torch.Tensor) and adj_pre.ndim == 3
    else:
        assert isinstance(adj_pre, SparseTensor)

    # Pooling operation
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    if pooler.is_dense_batched:
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
