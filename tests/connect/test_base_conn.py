"""Tests for tgp.connect.base_conn module.

Tests the Connect base class, sparse_connect function, and SparseConnect class.
"""

import pytest
import torch

from tgp.connect import Connect, SparseConnect, sparse_connect
from tgp.select import SelectOutput
from tgp.utils import connectivity_to_edge_index

CONNECTIVITY_FORMATS = [
    "edge_index",
    "torch_coo",
    pytest.param("torch_sparse", marks=pytest.mark.torch_sparse),
]


def _as_connectivity(edge_index, edge_weight, num_nodes, fmt):
    if fmt == "edge_index":
        return edge_index, edge_weight
    if fmt == "torch_coo":
        values = (
            edge_weight if edge_weight is not None else torch.ones(edge_index.size(1))
        )
        return torch.sparse_coo_tensor(
            edge_index, values, (num_nodes, num_nodes)
        ).coalesce(), None
    sparse = pytest.importorskip("torch_sparse")
    return sparse.SparseTensor.from_edge_index(
        edge_index, edge_weight, sparse_sizes=(num_nodes, num_nodes)
    ), None


def _dense_connectivity(adj, weight, num_nodes):
    edge_index, weight = connectivity_to_edge_index(adj, weight)
    if weight is None:
        weight = torch.ones(edge_index.size(1))
    return torch.sparse_coo_tensor(
        edge_index, weight, (num_nodes, num_nodes)
    ).to_dense()


@pytest.mark.parametrize(
    "nodes,clusters",
    [
        ([3, 4, 1], None),
        (None, [0, 1, 0, 2, 1]),
        ([4, 1, 3], [1, 2, 0]),
    ],
    ids=["node_only", "cluster_only", "paired"],
)
def test_sparse_connect_direct_matches_select_output(nodes, clusters):
    edge_index = torch.tensor([[4, 1, 4, 0, 3, 3], [1, 4, 1, 4, 3, 1]])
    edge_weight = torch.tensor([2.0, 3.0, 5.0, 11.0, 7.0, 13.0])
    kwargs = dict(
        node_index=None if nodes is None else torch.tensor(nodes),
        cluster_index=None if clusters is None else torch.tensor(clusters),
        num_nodes=5,
    )

    direct_edges, direct_weights = sparse_connect(edge_index, edge_weight, **kwargs)
    so = SelectOutput(**kwargs)
    wrapped_edges, wrapped_weights = SparseConnect()(
        edge_index, so, edge_weight=edge_weight
    )

    # SelectOutput sorts sparse rows; that must not change pooled node IDs or edges.
    assert torch.equal(direct_edges, wrapped_edges)
    torch.testing.assert_close(direct_weights, wrapped_weights)


@pytest.mark.parametrize("fmt", CONNECTIVITY_FORMATS)
@pytest.mark.parametrize("remove_self_loops", [False, True])
@pytest.mark.parametrize(
    "nodes,clusters,num_supernodes",
    [
        ([3, 4, 1], None, 3),
        ([1, 3, 4], [2, 0, 1], 3),
        ([4, 2, 0, 3, 1], None, 5),
        (None, [1, 0, 1, 2, 0], 3),
        ([4, 1, 3], [0, 0, 1], 2),
        ([3], None, 1),
        ([], None, 0),
    ],
)
def test_sparse_connect_matches_assignment_matrix(
    fmt, remove_self_loops, nodes, clusters, num_supernodes
):
    adjacency = torch.tensor(
        [
            [2.0, 3.0, 0.0, 0.0, 5.0],
            [3.0, 0.0, 7.0, 0.0, 11.0],
            [0.0, 7.0, 0.0, 13.0, 0.0],
            [0.0, 0.0, 13.0, 17.0, 19.0],
            [5.0, 11.0, 0.0, 19.0, 0.0],
        ]
    )
    edge_index = adjacency.nonzero().t()
    edge_weight = adjacency[edge_index[0], edge_index[1]]
    adj, weight = _as_connectivity(edge_index, edge_weight, 5, fmt)
    node_index = None if nodes is None else torch.tensor(nodes, dtype=torch.long)
    cluster_index = (
        None if clusters is None else torch.tensor(clusters, dtype=torch.long)
    )

    pooled_adj, pooled_weight = sparse_connect(
        adj,
        weight,
        node_index=node_index,
        cluster_index=cluster_index,
        num_nodes=5,
        num_supernodes=num_supernodes,
        remove_self_loops=remove_self_loops,
    )

    assignment = torch.zeros(5, num_supernodes)
    original_ids = list(range(5)) if nodes is None else nodes
    pooled_ids = list(range(len(original_ids))) if clusters is None else clusters
    assignment[original_ids, pooled_ids] = 1
    expected = assignment.t() @ adjacency @ assignment
    if remove_self_loops:
        expected.fill_diagonal_(0)
    actual = _dense_connectivity(pooled_adj, pooled_weight, num_supernodes)
    torch.testing.assert_close(actual, expected)
    if fmt == "edge_index":
        assert not pooled_adj.is_sparse
    else:
        assert pooled_weight is None
        if fmt == "torch_coo":
            assert pooled_adj.is_sparse
            assert pooled_adj.shape == (num_supernodes, num_supernodes)
        else:
            assert pooled_adj.sparse_sizes() == (num_supernodes, num_supernodes)


@pytest.mark.parametrize("reduce_op", ["sum", "mean", "max"])
@pytest.mark.parametrize("weighted", [False, True])
@pytest.mark.parametrize("full_assignment", [False, True])
def test_sparse_connect_preserves_duplicate_edge_policy(
    reduce_op, weighted, full_assignment
):
    edge_index = torch.tensor([[4, 1, 4, 0, 3, 3], [1, 4, 1, 4, 3, 1]])
    edge_weight = torch.tensor([2.0, 3.0, 5.0, 11.0, 7.0, 13.0]) if weighted else None
    nodes = [4, 1, 3, 0, 2] if full_assignment else [4, 1, 3]

    actual_edges, actual_weights = sparse_connect(
        edge_index,
        edge_weight,
        node_index=torch.tensor(nodes),
        num_nodes=5,
        reduce_op=reduce_op,
    )

    if full_assignment:
        expected_edges = torch.tensor([[0, 1, 2, 3], [1, 0, 1, 0]])
        merged_weight = {"sum": 7.0, "mean": 3.5, "max": 5.0}[reduce_op]
        expected_weights = torch.tensor([merged_weight, 3.0, 13.0, 11.0])
    else:
        expected_edges = torch.tensor([[0, 1, 0, 2], [1, 0, 1, 1]])
        expected_weights = torch.tensor([2.0, 3.0, 5.0, 13.0])
    assert torch.equal(actual_edges, expected_edges)
    if weighted:
        torch.testing.assert_close(actual_weights, expected_weights)
    else:
        assert actual_weights is None


@pytest.mark.parametrize(
    "reduce_op,expected", [("sum", 10.0), ("mean", 5.0), ("max", 8.0)]
)
def test_sparse_connect_aggregates_edges_after_merging_nodes(reduce_op, expected):
    edges, weights = sparse_connect(
        torch.tensor([[0, 1, 0], [2, 2, 1]]),
        torch.tensor([2.0, 8.0, 13.0]),
        cluster_index=torch.tensor([0, 0, 1]),
        num_nodes=3,
        reduce_op=reduce_op,
    )
    assert torch.equal(edges, torch.tensor([[0], [1]]))
    torch.testing.assert_close(weights, torch.tensor([expected]))


@pytest.mark.parametrize("fmt", CONNECTIVITY_FORMATS)
@pytest.mark.parametrize("num_nodes", [0, 5])
def test_sparse_connect_empty_edges_and_isolated_nodes(fmt, num_nodes):
    adj, weight = _as_connectivity(
        torch.empty((2, 0), dtype=torch.long), None, num_nodes, fmt
    )
    nodes = (
        torch.tensor([4, 1], dtype=torch.long)
        if num_nodes
        else torch.empty(0, dtype=torch.long)
    )
    pooled_adj, pooled_weight = sparse_connect(
        adj,
        weight,
        node_index=nodes,
        # Sparse inputs carry their dimensions even when no edges exist.
        num_nodes=num_nodes if fmt == "edge_index" else None,
    )
    expected_size = nodes.numel()
    torch.testing.assert_close(
        _dense_connectivity(pooled_adj, pooled_weight, expected_size),
        torch.zeros(expected_size, expected_size),
    )
    if fmt == "torch_coo":
        assert pooled_adj.shape == (expected_size, expected_size)
    elif fmt == "torch_sparse":
        assert pooled_adj.sparse_sizes() == (expected_size, expected_size)


def test_sparse_connect_removes_all_incident_edges():
    edges, weights = sparse_connect(
        torch.tensor([[0, 1, 2], [1, 2, 0]]),
        torch.tensor([2.0, 3.0, 4.0]),
        node_index=torch.tensor([3, 4]),
        num_nodes=5,
    )
    assert edges.shape == (2, 0)
    assert weights.shape == (0,)


@pytest.mark.parametrize("fmt", CONNECTIVITY_FORMATS)
@pytest.mark.parametrize(
    "degree_norm,edge_weight_norm",
    [(False, False), (True, False), (False, True), (True, True)],
)
@pytest.mark.parametrize("remove_self_loops", [False, True])
def test_sparse_connect_batched_normalization_matches_dense_reference(
    fmt, degree_norm, edge_weight_norm, remove_self_loops
):
    block = torch.tensor(
        [
            [5.0, 2.0, 0.0, 4.0],
            [2.0, 6.0, 3.0, 0.0],
            [0.0, 3.0, 7.0, 8.0],
            [4.0, 0.0, 8.0, 9.0],
        ]
    )
    adjacency = torch.block_diag(block, 2 * block)
    edge_index = adjacency.nonzero().t()
    adj, weight = _as_connectivity(
        edge_index, adjacency[edge_index[0], edge_index[1]], 8, fmt
    )
    selected = torch.tensor([3, 1, 0, 6, 4, 7])
    so = SelectOutput(node_index=selected, num_nodes=8)
    pooled_batch = torch.tensor([0, 0, 0, 1, 1, 1])
    connector = SparseConnect(
        degree_norm=degree_norm,
        edge_weight_norm=edge_weight_norm,
        remove_self_loops=remove_self_loops,
    )
    result = connector(adj, so, edge_weight=weight, batch_pooled=pooled_batch)

    expected = adjacency[selected][:, selected]
    if remove_self_loops:
        expected.fill_diagonal_(0)
    if degree_norm:
        inv_degree = expected.sum(dim=1).rsqrt()
        expected = inv_degree[:, None] * expected * inv_degree[None, :]
    if edge_weight_norm:
        for start in [0, 3]:
            graph = expected[start : start + 3, start : start + 3]
            graph /= graph.abs().max()
    torch.testing.assert_close(_dense_connectivity(*result, 6), expected)
    pooled_edges, _ = connectivity_to_edge_index(*result)
    assert torch.equal(pooled_batch[pooled_edges[0]], pooled_batch[pooled_edges[1]])


@pytest.mark.parametrize(
    "kwargs,match",
    [
        ({"cluster_index": torch.tensor([0, 1])}, "num_nodes entries"),
        (
            {"node_index": torch.tensor([0, 1]), "cluster_index": torch.tensor([0])},
            "same length",
        ),
        ({"node_index": torch.tensor([[0, 1]])}, "one-dimensional"),
        ({"cluster_index": torch.tensor([[0, 1, 2]])}, "one-dimensional"),
        ({"node_index": torch.tensor([0, 1, 2, 0])}, "more than num_nodes"),
    ],
)
def test_sparse_connect_rejects_invalid_index_lengths(kwargs, match):
    with pytest.raises(ValueError, match=match):
        sparse_connect(torch.tensor([[0, 1], [1, 0]]), num_nodes=3, **kwargs)


class TestConnect:
    """Test the Connect base class."""

    def test_connect_repr(self):
        """Test the __repr__ method of the Connect class."""
        connect_instance = Connect()
        repr_str = repr(connect_instance)
        assert isinstance(repr_str, str)
        assert "Connect()" in repr_str

    def test_connect_forward_not_implemented(self):
        """Test that Connect.forward raises NotImplementedError."""
        connect_instance = Connect()
        so = SelectOutput(s=torch.eye(3))
        with pytest.raises(NotImplementedError):
            connect_instance.forward(
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
                so=so,
            )


class TestSparseConnect:
    """Test the SparseConnect class."""

    def test_sparse_connect_raises_runtime_error(self):
        """Test that sparse_connect raises RuntimeError when neither node_index nor cluster_index is provided."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)

        # Neither node_index nor cluster_index provided
        with pytest.raises(RuntimeError):
            sparse_connect(
                edge_index=edge_index,
                edge_weight=edge_weight,
                node_index=None,
                cluster_index=None,
            )

        # Check if the same error is raised for SparseConnect when s is dense and no node_index or cluster_index is provided
        s = torch.randn((2, 1), dtype=torch.float)
        so = SelectOutput(s=s)
        connector = SparseConnect()
        with pytest.raises(RuntimeError):
            connector(
                edge_index=edge_index,
                edge_weight=edge_weight,
                so=so,
            )

    def test_sparse_connect_with_node_index(self, pooler_test_graph_sparse):
        """Test sparse_connect with node_index (e.g., TopK pooling)."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create node_index for top-k selection (select first half of nodes)
        k = num_nodes // 2
        node_index = torch.arange(k, dtype=torch.long)

        # Test sparse_connect function
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index,
            edge_weight=edge_weight,
            node_index=node_index,
            num_nodes=num_nodes,
            num_supernodes=k,
            remove_self_loops=True,
        )

        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2  # edge_index format
        if edge_weight_pool is not None:
            assert edge_weight_pool.size(0) == adj_pool.size(1)

        # Test SparseConnect class
        cluster_index = torch.arange(k, dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=k,
        )
        connector = SparseConnect(remove_self_loops=True)
        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index,
            edge_weight=edge_weight,
            so=so,
        )

        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.size(0) == 2

    def test_sparse_connect_with_cluster_index(self, pooler_test_graph_sparse):
        """Test sparse_connect with cluster_index (e.g., Graclus, NDP)."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)
        num_supernodes = num_nodes // 2

        # Create cluster_index (2 nodes per cluster)
        cluster_index = torch.arange(num_nodes, dtype=torch.long) // 2

        # Test sparse_connect function
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index,
            edge_weight=edge_weight,
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            remove_self_loops=True,
            reduce_op="sum",
        )

        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2
        if edge_weight_pool is not None:
            assert edge_weight_pool.size(0) == adj_pool.size(1)

        # Test SparseConnect class
        so = SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )
        connector = SparseConnect(remove_self_loops=True, reduce_op="sum")
        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index,
            edge_weight=edge_weight,
            so=so,
        )

        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.size(0) == 2

    def test_sparse_connect_degree_norm_without_edge_weight(self):
        """Test sparse_connect with degree_norm=True and edge_weight=None."""
        # Create a simple graph with 4 nodes
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        num_nodes = 4
        num_supernodes = 2

        # Create cluster_index for coarsening (2 nodes per supernode)
        cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Test with degree_norm=True and edge_weight=None
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index,
            edge_weight=None,
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            degree_norm=True,
            remove_self_loops=True,
        )

        # Verify that edge weights were created and are not None
        assert edge_weight_pool is not None
        assert isinstance(edge_weight_pool, torch.Tensor)
        assert edge_weight_pool.size(0) == adj_pool.size(1)

        # Verify that the edge weights are normalized (should be between 0 and 1)
        assert torch.all(edge_weight_pool >= 0.0)
        assert torch.all(edge_weight_pool <= 1.0)

        # Test with SparseConnect class as well
        connector = SparseConnect(degree_norm=True)
        so = SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )

        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index,
            edge_weight=None,
            so=so,
        )

        # Verify that edge weights were created and are not None
        assert edge_weight_pool_class is not None
        assert isinstance(edge_weight_pool_class, torch.Tensor)
        assert edge_weight_pool_class.size(0) == adj_pool_class.size(1)

        # Verify that the edge weights are normalized
        assert torch.all(edge_weight_pool_class >= 0.0)
        assert torch.all(edge_weight_pool_class <= 1.0)

    @pytest.mark.torch_sparse
    def test_sparse_connect_with_torch_coo(self):
        """Test sparse_connect with torch COO sparse tensor input."""
        pytest.importorskip("torch_sparse")

        # Create a simple graph with 4 nodes
        edge_index = torch.tensor(
            [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
        )
        edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0, 1.0, 1.0], dtype=torch.float)
        num_nodes = 4
        num_supernodes = 2

        # Convert to torch COO sparse tensor
        edge_index_coo = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        # Create cluster_index for coarsening (2 nodes per supernode)
        cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)

        # Test with torch COO sparse tensor input
        adj_pool, edge_weight_pool = sparse_connect(
            edge_index=edge_index_coo,
            edge_weight=None,  # Edge weights are embedded in the sparse tensor
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            degree_norm=False,
            remove_self_loops=True,
        )

        # Verify that output is a torch COO sparse tensor
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.is_sparse
        assert edge_weight_pool is None  # Should be None when output is torch COO

        # Verify the shape of the pooled adjacency
        assert adj_pool.shape == (num_supernodes, num_supernodes)

        # Test with SparseConnect class as well
        connector = SparseConnect()
        so = SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )

        adj_pool_class, edge_weight_pool_class = connector(
            edge_index=edge_index_coo,
            edge_weight=None,
            so=so,
        )

        # Verify that output is a torch COO sparse tensor
        assert isinstance(adj_pool_class, torch.Tensor)
        assert adj_pool_class.is_sparse
        assert edge_weight_pool_class is None  # Should be None when output is torch COO

        # Verify the shape of the pooled adjacency
        assert adj_pool_class.shape == (num_supernodes, num_supernodes)

    def test_sparse_connect_repr(self):
        """Test SparseConnect __repr__ method."""
        connector = SparseConnect(
            reduce_op="mean",
            remove_self_loops=False,
            edge_weight_norm=True,
            degree_norm=True,
        )
        repr_str = repr(connector)
        assert "SparseConnect" in repr_str
        assert "reduce_op=mean" in repr_str
        assert "remove_self_loops=False" in repr_str
        assert "edge_weight_norm=True" in repr_str
        assert "degree_norm=True" in repr_str

    def test_sparse_connect_edge_weight_norm_requires_batch_pooled(self):
        """Test that SparseConnect raises AssertionError when edge_weight_norm=True but batch_pooled=None."""
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
        node_index = torch.tensor([0, 1], dtype=torch.long)
        so = SelectOutput(
            node_index=node_index,
            cluster_index=torch.arange(node_index.numel(), dtype=torch.long),
            num_nodes=2,
            num_supernodes=2,
        )

        connector = SparseConnect(edge_weight_norm=True)
        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                edge_index=edge_index,
                edge_weight=edge_weight,
                so=so,
                batch_pooled=None,
            )
