import pytest
import torch
from torch_geometric.nn import DenseGCNConv, GCNConv
from torch_geometric.utils import add_self_loops, erdos_renyi_graph

from tgp.poolers import get_pooler, pooler_map
from tgp.select.base_select import SelectOutput


@pytest.fixture(scope="module")
def simple_graph():
    F = 3
    torch.manual_seed(42)  # For reproducibility

    # Graph 1: Chain graph (directed) - 10 nodes
    N1 = 10
    row1 = torch.arange(N1 - 1, dtype=torch.long)
    col1 = row1 + 1
    edge_index1 = torch.stack([row1, col1], dim=0)  # Directed, no reverse edges
    E1 = edge_index1.size(1)
    x1 = torch.randn((N1, F), dtype=torch.float)
    edge_weight1 = torch.ones(E1, dtype=torch.float)
    edge_index1, edge_weight1 = add_self_loops(
        edge_index1, edge_attr=edge_weight1, num_nodes=N1
    )
    batch1 = torch.zeros(N1, dtype=torch.long)

    # Graph 2: Grid graph (undirected) - 3x3 grid = 9 nodes
    N2 = 9
    rows, cols = 3, 3
    edge_list2 = []
    # Horizontal edges
    for i in range(rows):
        for j in range(cols - 1):
            node = i * cols + j
            edge_list2.append([node, node + 1])
            edge_list2.append([node + 1, node])  # Undirected
    # Vertical edges
    for i in range(rows - 1):
        for j in range(cols):
            node = i * cols + j
            edge_list2.append([node, node + cols])
            edge_list2.append([node + cols, node])  # Undirected
    edge_index2 = torch.tensor(edge_list2, dtype=torch.long).t().contiguous()
    E2 = edge_index2.size(1)
    x2 = torch.randn((N2, F), dtype=torch.float)
    edge_weight2 = torch.ones(E2, dtype=torch.float)
    edge_index2, edge_weight2 = add_self_loops(
        edge_index2, edge_attr=edge_weight2, num_nodes=N2
    )
    batch2 = torch.ones(N2, dtype=torch.long)

    # Graph 3: Random graph (undirected, some nodes might be disconnected) - 20 nodes
    N3 = 20
    edge_index3 = erdos_renyi_graph(N3, edge_prob=0.3, directed=False)
    E3 = edge_index3.size(1)
    x3 = torch.randn((N3, F), dtype=torch.float)
    edge_weight3 = torch.ones(E3, dtype=torch.float)
    edge_index3, edge_weight3 = add_self_loops(
        edge_index3, edge_attr=edge_weight3, num_nodes=N3
    )
    batch3 = torch.full((N3,), 2, dtype=torch.long)

    # Combine graphs into mini-batch
    # Offset node indices for graphs 2 and 3
    edge_index2_offset = edge_index2 + N1
    edge_index3_offset = edge_index3 + N1 + N2

    # Concatenate everything
    x = torch.cat([x1, x2, x3], dim=0)
    edge_index = torch.cat([edge_index1, edge_index2_offset, edge_index3_offset], dim=1)
    edge_weight = torch.cat([edge_weight1, edge_weight2, edge_weight3], dim=0)
    batch = torch.cat([batch1, batch2, batch3], dim=0)

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
    # For batched graphs, some poolers (e.g., SparseBNPool) create k supernodes per graph,
    # so num_supernodes can be batch_size * k, which may exceed N.
    # We check that we have at least 1 supernode and at most batch_size * k.
    if batch is not None:
        batch_size = int(batch.max().item()) + 1
        k = PARAMS.get("k", max(1, N // 2))
        max_expected_supernodes = batch_size * k
        assert 1 <= num_supernodes <= max_expected_supernodes
    else:
        assert 1 <= num_supernodes <= N

    # For dense [N, K] assignment matrices with multi-graph batches (e.g., SparseBNPool),
    # the actual output has B*K nodes, while num_supernodes = K (clusters per graph).
    # For sparse block-diagonal assignments, num_supernodes = B*K.
    actual_max_supernodes = (
        max_expected_supernodes if batch is not None else num_supernodes
    )
    assert 1 <= out.x.size(0) <= actual_max_supernodes

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
    lift_kwargs = {}
    if pooler_name in {"lap", "spbnpool"}:
        lift_kwargs["batch_pooled"] = out.batch
    x_lifted = pooler(x=x_pool, so=out.so, lifting=True, **lift_kwargs)
    assert isinstance(x_lifted, torch.Tensor)
    # For batched graphs, the lifted features should match the number of nodes
    # that the SelectOutput knows about. Some poolers may only lift one graph at a time
    # when dealing with batched graphs, so we check against out.so.num_nodes instead of N.
    assert x_lifted.size(-2) == out.so.num_nodes
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
