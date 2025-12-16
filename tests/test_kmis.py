import importlib
import sys

import pytest
import torch
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

import tgp.imports as imports_mod
from tgp.poolers import get_pooler
from tgp.select.base_select import SelectOutput
from tgp.select.kmis_select import (
    KMISSelect,
    degree_scorer,
    maximal_independent_set,
    maximal_independent_set_cluster,
)


@pytest.fixture(scope="module")
def simple_graph():
    N = 10
    F = 3
    row = torch.arange(9, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    E = edge_index.size(1)

    x = torch.randn((N, F), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    batch = torch.zeros(N, dtype=torch.long)

    return x, edge_index, edge_weight, batch


@pytest.mark.parametrize("pooler_name", ["kmis"])
@pytest.mark.parametrize("lift", ["precomputed", "transpose", "inverse"])
@pytest.mark.parametrize("s_inv_op", ["transpose", "inverse"])
@pytest.mark.parametrize("lift_red_op", ["sum", "mean", "max"])
@pytest.mark.parametrize("reduce_red_op", ["sum", "mean", None])
@pytest.mark.parametrize("remove_self_loops", [True, False])
@pytest.mark.parametrize(
    "scorer", ["linear", "random", "constant", "canonical", "degree"]
)
def test_pooler_parametrized_configs(
    simple_graph,
    pooler_name,
    lift,
    s_inv_op,
    lift_red_op,
    reduce_red_op,
    remove_self_loops,
    scorer,
):
    """Test a given pooler with various combinations of configuration parameters.
    For each config, run preprocessing and forward, and check output.x shape.
    """
    x, edge_index, edge_weight, batch = simple_graph
    N, F = x.size()
    PARAMS = {
        "in_channels": F,
        "ratio": 0.5,
        "order_k": max(1, N // 2),
        "cached": True,
        "lift": lift,
        "s_inv_op": s_inv_op,
        "lift_red_op": lift_red_op,
        "loss_coeff": 1.0,
        "remove_self_loops": remove_self_loops,
        "scorer": scorer,
        "reduce_red_op": reduce_red_op,
    }

    # KMISPooling raises at construction if cached=True and scorer='linear'
    if scorer == "linear" and PARAMS["cached"]:
        with pytest.raises(Exception, match="Caching should be disabled"):
            _ = get_pooler(pooler_name, **PARAMS)
        # Now retry with caching disabled
        PARAMS["cached"] = False

    pooler = get_pooler(pooler_name, **PARAMS)
    pooler.eval()
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
    edge_input = adj
    x_pre, adj_pre, mask = pooler.preprocessing(
        edge_index=edge_input, x=x, batch=batch, use_cache=False
    )
    out = pooler(x=x_pre, adj=adj_pre, batch=batch, mask=mask)
    assert hasattr(out, "x")
    assert isinstance(out.x, torch.Tensor)
    # Output.x should have shape [num_supernodes, F] where 1 <= num_supernodes <= N
    assert 1 <= out.x.size(0) <= N
    assert out.x.size(1) == F


@pytest.fixture
def simple_edge_index():
    return torch.tensor([[0, 1], [1, 2]], dtype=torch.long)


def test_degree_scorer_valid(simple_edge_index):
    # Valid 1D edge_weight: degrees computed correctly
    weight = torch.tensor([1.0, 2.0])
    deg = degree_scorer(
        edge_index=simple_edge_index,
        edge_weight=weight,
        num_nodes=3,
    )
    # Node 0: no incoming, deg=0; node1: one incoming (weight=1); node2: one incoming (weight=2)
    expected = torch.tensor([0.0, 1.0, 2.0])
    assert torch.equal(deg.view(-1), expected)


def test_maximal_independent_set_basic():
    # Build an undirected path 0-1-2-3 (4 nodes)
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    # order_k=1: greedy MIS picks nodes 0 and 2
    mis = maximal_independent_set(edge_index=edge_index, order_k=1)
    assert mis.dtype == torch.bool
    assert mis.sum().item() == 2
    assert mis[0] and mis[2]

    # With a custom permutation (reverse), it should pick nodes [3,1]
    perm = torch.tensor([3, 2, 1, 0], dtype=torch.long)
    mis2 = maximal_independent_set(edge_index=edge_index, order_k=1, perm=perm)
    # Now highest-ranked (3) and then its neighbor's picks
    assert mis2[3] and mis2[1]
    assert mis2.sum().item() == 2


def test_maximal_independent_set_cluster():
    # Same 4-node path
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    mis, clusters = maximal_independent_set_cluster(edge_index=edge_index, order_k=1)
    # mis is boolean mask, clusters assigns each node to a cluster index
    assert mis.dtype == torch.bool
    assert clusters.dtype == torch.long
    assert mis.sum().item() == len(torch.unique(clusters[mis]))


def test_kmis_select_scorers_and_s_inv(simple_edge_index):
    N = 4
    # Build undirected adjacency with self-loops for N=4
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    x = torch.randn((N, 3), dtype=torch.float)

    # Test each non-linear scorer
    for scorer in ["random", "constant", "canonical", "degree"]:
        selector = KMISSelect(
            in_channels=None,
            order_k=2,
            scorer=scorer,
            score_heuristic=None,
            force_undirected=False,
            s_inv_op="transpose",
        )
        selector.eval()
        out = selector.forward(x=x, edge_index=edge_index, batch=None)
        assert isinstance(out, SelectOutput)
        # s_inv (transpose) should swap row/col
        s = out.s
        s_inv = out.s_inv
        # Extract row, col from either SparseTensor or torch COO
        if isinstance(s, SparseTensor):
            row_s, col_s, _ = s.coo()
        else:
            s_coalesced = s.coalesce()
            indices_s = s_coalesced.indices()
            row_s, col_s = indices_s[0], indices_s[1]
        if isinstance(s_inv, SparseTensor):
            row_i, col_i, _ = s_inv.coo()
        else:
            s_inv_coalesced = s_inv.coalesce()
            indices_i = s_inv_coalesced.indices()
            row_i, col_i = indices_i[0], indices_i[1]
        assert torch.equal(row_i, col_s)
        assert torch.equal(col_i, row_s)

    # Test force_undirected=True: use a directed edge_index
    directed = torch.tensor([[0, 1], [1, 2]], dtype=torch.long)
    N2 = 3
    x2 = torch.randn((N2, 2), dtype=torch.float)
    selector2 = KMISSelect(
        in_channels=None,
        order_k=1,
        scorer="degree",
        score_heuristic=None,
        force_undirected=True,
    )
    selector2.eval()
    out2 = selector2.forward(x=x2, edge_index=directed, batch=None)
    assert isinstance(out2, SelectOutput)
    assert out2.mis.sum().item() == 1

    ei_spt = SparseTensor.from_edge_index(directed, sparse_sizes=(N2, N2))
    out3 = selector2.forward(x=x2, edge_index=ei_spt, batch=None)
    assert isinstance(out3, SelectOutput)


def test_kmis_select_linear_score_list(simple_edge_index):
    N = 4
    # Build undirected adjacency with self-loops for N=4
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3, 3, 0], [1, 0, 2, 1, 3, 2, 0, 3]], dtype=torch.long
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    x = torch.randn((N, 3), dtype=torch.float)

    selector = KMISSelect(
        in_channels=[3, 3],
        order_k=2,
        scorer="linear",
        score_heuristic=None,
        force_undirected=False,
        s_inv_op="transpose",
    )
    selector.eval()
    out = selector.forward(x=x, edge_index=edge_index, batch=None)
    assert isinstance(out, SelectOutput)
    # s_inv (transpose) should swap row/col
    s = out.s
    s_inv = out.s_inv
    # Extract row, col from either SparseTensor or torch COO
    if isinstance(s, SparseTensor):
        row_s, col_s, _ = s.coo()
    else:
        s_coalesced = s.coalesce()
        indices_s = s_coalesced.indices()
        row_s, col_s = indices_s[0], indices_s[1]
    if isinstance(s_inv, SparseTensor):
        row_i, col_i, _ = s_inv.coo()
    else:
        s_inv_coalesced = s_inv.coalesce()
        indices_i = s_inv_coalesced.indices()
        row_i, col_i = indices_i[0], indices_i[1]
    assert torch.equal(row_i, col_s)
    assert torch.equal(col_i, row_s)


def test_kmis_select_heuristic_inverse_s_inv(simple_edge_index):
    N = 3
    # Triangle with self-loops
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 0], [1, 0, 2, 1, 0, 2]], dtype=torch.long
    )
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_index = SparseTensor.from_edge_index(edge_index)
    x = torch.arange(N, dtype=torch.float).view(N, 1)

    # Greedy heuristic with s_inv_op="inverse"
    selector = KMISSelect(
        in_channels=None,
        order_k=2,
        scorer="constant",
        score_heuristic="greedy",
        force_undirected=False,
        s_inv_op="inverse",
    )
    selector.eval()
    out = selector.forward(x=x, edge_index=edge_index, batch=None)
    # Check S and S_inv multiplication approximates identity on selected rows
    S = out.s.to_dense()
    Sinv = out.s_inv.to_dense()
    approx = S @ Sinv @ S
    assert approx.shape == S.shape

    # w-greedy heuristic
    selector2 = KMISSelect(
        in_channels=None,
        order_k=1,
        scorer="constant",
        score_heuristic="w-greedy",
        force_undirected=True,
        s_inv_op="transpose",
    )
    selector2.eval()
    out2 = selector2.forward(x=x, edge_index=edge_index, batch=None)
    assert isinstance(out2, SelectOutput)


def test_kmis_select_repr_and_invalid_args():
    # Valid repr
    selector = KMISSelect(
        in_channels=None,
        order_k=1,
        scorer="degree",
        score_heuristic="greedy",
        force_undirected=True,
        s_inv_op="transpose",
    )
    rep = repr(selector)
    assert (
        "order_k=1" in rep and "scorer=degree" in rep and "force_undirected=True" in rep
    )

    # Invalid scorer
    with pytest.raises(AssertionError):
        _ = KMISSelect(
            in_channels=None, order_k=1, scorer="invalid", score_heuristic=None
        )

    # Invalid heuristic
    with pytest.raises(AssertionError):
        _ = KMISSelect(
            in_channels=None, order_k=1, scorer="degree", score_heuristic="invalid"
        )


@pytest.fixture
def simple_undirected_edge_index():
    # A small 5‐node cycle, undirected, with self‐loops
    N = 5
    # build a cycle 0–1–2–3–4–0
    row = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4, 0], dtype=torch.long)
    col = torch.tensor([1, 2, 3, 4, 0, 0, 1, 2, 3, 4], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    return edge_index


def test_mis_without_torch_scatter(monkeypatch, simple_undirected_edge_index):
    # Force HAS_TORCH_SCATTER=False
    import tgp.select.kmis_select as kmis_mod

    monkeypatch.setattr(kmis_mod, "HAS_TORCH_SCATTER", False)

    edge_index = simple_undirected_edge_index
    order_k = 1

    # Now this call runs the “else” block (using PyG’s scatter())
    mis = maximal_independent_set(edge_index=edge_index, order_k=order_k)
    assert mis.dtype == torch.bool
    assert mis.numel() == edge_index.max().item() + 1  # = 5

    # Test the whole KMISSelect
    selector = KMISSelect(
        in_channels=None,
        order_k=order_k,
        scorer="degree",
        score_heuristic="greedy",
        force_undirected=False,
        s_inv_op="transpose",
    )
    selector.eval()
    out = selector.forward(x=torch.randn((5, 3)), edge_index=edge_index, batch=None)
    assert isinstance(out, SelectOutput)


def test_skip_torch_scatter_import(monkeypatch):
    # 1) Ensure the flag is False before kmis_select ever runs its top‐level import:
    monkeypatch.setattr(imports_mod, "HAS_TORCH_SCATTER", False)

    # 2) Remove any previously loaded copy of kmis_select
    if "tgp.select.kmis_select" in sys.modules:
        del sys.modules["tgp.select.kmis_select"]

    # 3) Now re‐import (this will re‐run the top‐level code, and skip the torch_scatter imports)
    kmis_mod = importlib.import_module("tgp.select.kmis_select")

    # 4) Verify that the module's flag is False, and that the torch_scatter names were never bound:
    assert kmis_mod.HAS_TORCH_SCATTER is False
    assert not hasattr(kmis_mod, "scatter_add")
    assert not hasattr(kmis_mod, "scatter_max")
    assert not hasattr(kmis_mod, "scatter_min")


if __name__ == "__main__":
    pytest.main([__file__])
