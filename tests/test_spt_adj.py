import pytest
import torch
from torch_geometric.utils import add_self_loops, erdos_renyi_graph
from torch_sparse import SparseTensor

from tgp.poolers import get_pooler, pooler_map


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


@pytest.mark.parametrize("pooler_name", poolers)
def test_output_with_spt_adj(simple_graph, pooler_name):
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

    # PANPooling requires SparseTensor input, so convert edge_index to SparseTensor
    # and set edge_weight to None since it's stored in the SparseTensor
    if pooler_name == "pan":
        adj = SparseTensor.from_edge_index(adj, edge_attr=edge_weight)
        edge_weight = None

    # Preprocessing data
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=adj, edge_weight=edge_weight, x=x, batch=batch, use_cache=False
    )
    if pooler.is_dense_batched:
        assert isinstance(adj_pre, torch.Tensor) and adj_pre.ndim == 3
    else:
        # For sparse poolers, adj_pre can be either SparseTensor or Tensor (edge_index format)
        assert isinstance(adj_pre, (SparseTensor, torch.Tensor))

    # Pooling operation
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    if pooler.is_dense_batched:
        assert isinstance(out.edge_index, torch.Tensor)
    else:
        # For sparse poolers, edge_index can be either SparseTensor or Tensor
        assert isinstance(out.edge_index, (SparseTensor, torch.Tensor))

    # lift
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

    # reset params check
    if hasattr(pooler, "reset_parameters"):
        pooler.reset_parameters()

    # repr check
    rep = repr(pooler)
    assert pooler_map[pooler_name].__name__.lower() in rep.lower()


if __name__ == "__main__":
    pytest.main([__file__])
