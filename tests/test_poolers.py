import pytest
import torch
from torch_geometric.nn import DenseGCNConv, GCNConv
from torch_geometric.utils import add_self_loops

from tgp.poolers import get_pooler, pooler_map
from tgp.select.base_select import SelectOutput


@pytest.fixture(scope="module")
def simple_graph():
    N = 10
    F = 3
    # Chain graph: edges (0-1, 1-2, ..., 8-9), made undirected
    row = torch.arange(9, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    E = edge_index.size(1)

    x = torch.randn((N, F), dtype=torch.float)
    edge_weight = torch.ones((E, 1), dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


poolers = list(pooler_map.keys())
excluded_poolers = ["pan"]
poolers = [p for p in poolers if p not in excluded_poolers]


@pytest.mark.parametrize("pooler_name", poolers)
def test_poolers_forward_and_lifting(simple_graph, pooler_name):
    x, edge_index, edge_weight, batch = simple_graph
    N, F = x.size()

    # Common parameters for all poolers (some may ignore irrelevant keys):
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

    # Instantiate the pooler
    pooler = get_pooler(pooler_name, **PARAMS)
    pooler.eval()
    print(f"Testing pooler: {pooler_name}")

    # Preprocessing data
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_index,
        edge_weight=edge_weight,
        x=x,
        batch=batch,
        use_cache=False,
    )
    if pooler.is_dense_batched:
        assert isinstance(adj_pre, torch.Tensor) and adj_pre.ndim == 3
    if mask is not None:
        assert isinstance(mask, torch.Tensor) and mask.dtype == torch.bool

    # Forward pass pooling
    out = pooler(x=x_pre, adj=adj_pre, edge_weight=edge_weight, batch=batch, mask=mask)
    assert hasattr(out, "x")
    assert hasattr(out, "so") and isinstance(out.so, SelectOutput)
    assert isinstance(out.x, torch.Tensor)
    num_supernodes = out.so.num_supernodes
    assert 1 <= num_supernodes <= N
    assert 1 <= out.x.size(0) <= N

    ei = out.edge_index
    assert isinstance(ei, torch.Tensor)

    # If edge_weight present, check shape consistency
    if hasattr(out, "edge_weight") and out.edge_weight is not None:
        ew = out.edge_weight
        assert isinstance(ew, torch.Tensor)
        assert ew.numel() == ei.size(1)

    # Apply message passing to ensure output is correct type
    conv = GCNConv(F, F) if not pooler.is_dense_batched else DenseGCNConv(F, F)
    out.x = conv(out.x, out.edge_index)
    assert isinstance(out.x, torch.Tensor)

    # Lifting check
    x_pool = out.x.clone()
    x_lifted = pooler(x=x_pool, so=out.so, lifting=True)
    assert isinstance(x_lifted, torch.Tensor)
    assert x_lifted.size(-2) == N
    assert x_lifted.size(-1) == x_pool.size(-1)

    # Aux loss check
    if hasattr(out, "loss") and out.loss is not None:
        for loss_val in out.loss.keys():
            # assert isinstance(loss_val, torch.Tensor) or isinstance(loss_val, float)
            assert isinstance(out.loss[loss_val], (torch.Tensor, float)), (
                f"Loss value {loss_val} should be a tensor or float, got {type(out.loss[loss_val])}"
            )

    # repr check
    rep = repr(pooler)
    assert pooler_map[pooler_name].__name__.lower() in rep.lower()

    # Cache check
    if hasattr(pooler, "clear_cache"):
        pooler.clear_cache()

    # reset params check
    if hasattr(pooler, "reset_parameters"):
        pooler.reset_parameters()

    x_pre2, adj_pre2, mask2 = pooler.preprocessing(
        edge_index=edge_index, edge_weight=edge_weight, x=x, batch=batch, use_cache=True
    )
    out2 = pooler(x=x_pre2, adj=adj_pre2, batch=batch, mask=mask2)
    assert isinstance(out2, type(out))


def test_wrong_pooler_name():
    with pytest.raises(ValueError):
        get_pooler("non_existent_pooler")


if __name__ == "__main__":
    pytest.main([__file__])
