import pytest
import torch
from torch import Tensor
from torch_geometric.utils import add_self_loops

from tests.test_utils import (
    make_chain_edge_index,
    make_chain_graph_sparse,
    make_simple_undirected_graph,
)
from tgp.poolers import EdgeContractionPooling
from tgp.select.edge_contraction_select import (
    EdgeContractionSelect,
    maximal_matching,
    maximal_matching_cluster,
)
from tgp.src import PoolingOutput


def test_forward():
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float).unsqueeze(1)  # make it 2D [N, 1]
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    pooler = EdgeContractionPooling(in_channels=1)
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_supernodes
    assert out.x.shape == (k, 1)


@pytest.mark.torch_sparse
def test_forward_with_sparse_adj():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    x, edge_index, edge_weight, batch = make_chain_graph_sparse(N=4, F_dim=2)
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)
    pooler = EdgeContractionPooling(in_channels=2)
    pooler.eval()

    out = pooler(x=x, adj=adj, batch=batch, lifting=False)
    assert isinstance(out, PoolingOutput)
    assert out.x.dim() == 2


def test_maximal_matching_without_perm():
    edge_index = make_simple_undirected_graph()
    # m = number of edges
    m = edge_index.size(1)
    # Calling without perm should return a boolean mask of length m
    match = maximal_matching(edge_index=edge_index, num_nodes=None)
    assert isinstance(match, torch.Tensor)
    assert match.dtype == torch.bool
    assert match.numel() == m
    # Ensure at least one edge is matched
    assert match.any().item() is True


def test_maximal_matching_with_perm():
    edge_index = make_simple_undirected_graph()
    m = edge_index.size(1)
    # Create a permutation reversing edge order
    perm = torch.arange(m - 1, -1, -1, dtype=torch.long)
    match = maximal_matching(edge_index=edge_index, num_nodes=4, perm=perm)
    assert isinstance(match, torch.Tensor)
    assert match.dtype == torch.bool
    assert match.numel() == m
    # Still should match at least one edge
    assert match.any().item() is True


def test_maximal_matching_cluster_basic():
    # 4-node path: 0–1–2–3
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    # No permutation: cluster nodes by matching
    match, cluster = maximal_matching_cluster(edge_index=edge_index, num_nodes=None)
    assert isinstance(match, torch.Tensor) and match.dtype == torch.bool
    assert isinstance(cluster, torch.Tensor) and cluster.dtype == torch.long
    assert cluster.numel() == 4
    # Each matched pair should share a cluster label
    for idx in match.nonzero().view(-1):
        u, v = edge_index[0, idx].item(), edge_index[1, idx].item()
        assert cluster[u].item() == cluster[v].item()


def test_maximal_matching_cluster_with_perm():
    # 4-node path, reversed permutation
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]],
        dtype=torch.long,
    )
    perm = torch.tensor([5, 4, 3, 2, 1, 0], dtype=torch.long)
    match, cluster = maximal_matching_cluster(
        edge_index=edge_index, num_nodes=4, perm=perm
    )
    assert isinstance(match, torch.Tensor) and match.dtype == torch.bool
    assert isinstance(cluster, torch.Tensor) and cluster.dtype == torch.long
    assert cluster.numel() == 4


@pytest.mark.parametrize(
    "method",
    [
        "compute_edge_score_softmax",
        "compute_edge_score_tanh",
        "compute_edge_score_sigmoid",
    ],
)
def test_compute_edge_score_methods(method):
    # Build a simple graph: 3 nodes, 2 edges 0–1 and 1–2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    num_nodes = 3
    raw_scores = torch.tensor([0.5, 1.0, -0.5, 2.0], dtype=torch.float)
    func = getattr(EdgeContractionSelect, method)
    if method == "compute_edge_score_softmax":
        out = func(raw_scores, edge_index, num_nodes)
        # Softmax over incoming edges; sum of out for each destination should be 1
        dst = edge_index[1]
        for node in range(num_nodes):
            mask = dst == node
            if mask.any().item():
                assert torch.allclose(out[mask].sum(), torch.tensor(1.0), atol=1e-6)
    elif method == "compute_edge_score_tanh":
        out = func(raw_scores, None, None)
        assert torch.allclose(out, torch.tanh(raw_scores))
    else:  # sigmoid
        out = func(raw_scores, None, None)
        assert torch.allclose(out, torch.sigmoid(raw_scores))


def test_edge_contraction_select_forward_and_repr():
    # 5-node cycle with self-loops
    N = 5
    row = torch.tensor([0, 1, 2, 3, 4, 1, 2, 3, 4, 0], dtype=torch.long)
    col = torch.tensor([1, 2, 3, 4, 0, 0, 1, 2, 3, 4], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    x = torch.randn((N, 3), dtype=torch.float)

    # Instantiate with tanh scoring and dropout
    selector = EdgeContractionSelect(
        in_channels=3,
        edge_score_method=EdgeContractionSelect.compute_edge_score_tanh,
        dropout=0.5,
        add_to_edge_score=0.1,
        s_inv_op="inverse",
    )
    selector.train()  # enable dropout
    so = selector(x=x, edge_index=edge_index)
    assert hasattr(so, "node_index")
    assert hasattr(so, "cluster_index")
    assert hasattr(so, "weight")
    assert isinstance(so.node_index, torch.Tensor)
    assert isinstance(so.cluster_index, torch.Tensor)
    assert isinstance(so.weight, torch.Tensor)
    # repr should contain parameter settings
    rep = repr(selector)
    assert "in_channels=3" in rep
    assert "add_to_edge_score=0.1" in rep
    assert "s_inv_op=inverse" in rep


def test_edge_contraction_select_forward_sigmoid_and_no_dropout():
    # Graph of 4 isolated nodes: no edges
    N = 4
    edge_index = torch.zeros((2, 0), dtype=torch.long)
    x = torch.randn((N, 2), dtype=torch.float)
    # Use sigmoid scoring, no dropout
    selector = EdgeContractionSelect(
        in_channels=2,
        edge_score_method=EdgeContractionSelect.compute_edge_score_sigmoid,
        dropout=0.0,
        add_to_edge_score=0.0,
        s_inv_op="transpose",
    )
    so = selector(x=x, edge_index=edge_index)
    # All nodes are unmatched; each forms its own cluster
    assert so.num_supernodes == N
    assert torch.equal(so.cluster_index, torch.arange(N))
    # Weight should be ones of length N
    assert torch.equal(so.weight, torch.ones(N))


@pytest.mark.torch_sparse
def test_maximal_matching_with_sparsetensor_and_perm():
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    N = 3
    edge_index = make_chain_edge_index(N, add_self_loops=True)
    sp = SparseTensor.from_edge_index(edge_index)

    m = sp.nnz()  # number of nonzeros
    perm = torch.arange(m - 1, -1, -1, dtype=torch.long)

    match = maximal_matching(edge_index=sp, num_nodes=None, perm=perm)
    assert isinstance(match, torch.Tensor)
    assert match.dtype == torch.bool
    assert match.numel() == m
    # At least one edge should be matched
    assert match.any().item()


@pytest.mark.torch_sparse
def test_maximal_matching_cluster_with_sparsetensor_and_perm():
    """Cover lines 100-102 (SparseTensor branch) and 109 (cluster labeling) in maximal_matching_cluster."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # Build a 4-node cycle 0–1–2–3–0 plus self-loops
    N = 4
    row = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    col = torch.tensor([1, 2, 3, 0], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    sp = SparseTensor.from_edge_index(edge_index)

    m = sp.nnz()
    perm = torch.arange(m - 1, -1, -1, dtype=torch.long)

    match, cluster = maximal_matching_cluster(edge_index=sp, num_nodes=None, perm=perm)
    assert isinstance(match, Tensor) and match.dtype == torch.bool
    assert match.numel() == m
    assert isinstance(cluster, Tensor) and cluster.dtype == torch.long
    assert cluster.numel() == N


@pytest.mark.torch_sparse
def test_maximal_matching_cluster_with_sparsetensor_no_perm():
    """Cover the default-perm branch in maximal_matching_cluster when passing a SparseTensor."""
    pytest.importorskip("torch_sparse")
    from torch_sparse import SparseTensor

    # 3-node chain 0–1–2 plus self-loops
    N = 3
    edge_index = make_chain_edge_index(N, add_self_loops=True)
    sp = SparseTensor.from_edge_index(edge_index)

    match, cluster = maximal_matching_cluster(edge_index=sp, num_nodes=None, perm=None)
    assert isinstance(match, Tensor) and match.dtype == torch.bool
    assert isinstance(cluster, Tensor) and cluster.dtype == torch.long
    assert cluster.numel() == N


def test_edgecontractionselect_weight_assignment_and_clusters():
    """Cover line 195 (constructor init), 204 (forward), and weight/cluster logic."""
    # Build a 3-node path 0–1–2 without self-loops (to see default behavior).
    N = 3
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    x = torch.randn((N, 2), dtype=torch.float)

    # Use tanh scoring (compute_edge_score_tanh) and nonzero add_to_edge_score
    selector = EdgeContractionSelect(
        in_channels=2,
        edge_score_method=EdgeContractionSelect.compute_edge_score_tanh,
        dropout=0.0,
        add_to_edge_score=0.5,
        s_inv_op="transpose",
    )
    # Ensure repr contains chosen parameters (covers __repr__)
    rep = repr(selector)
    assert "in_channels=2" in rep
    assert "compute_edge_score_tanh" in rep
    assert "add_to_edge_score=0.5" in rep
    assert "s_inv_op=transpose" in rep

    # Call forward (covers line 204 onward)
    so = selector(x=x, edge_index=edge_index)
    # Check that each node appears in node_index
    assert torch.equal(so.node_index, torch.arange(N))
    # cluster_index should have length N
    assert so.cluster_index.numel() == N
    # weight length equals number of clusters = so.num_supernodes
    # assert so.weight.numel() == so.num_supernodes


if __name__ == "__main__":
    pytest.main([__file__])
