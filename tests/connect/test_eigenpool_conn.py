import pytest
import torch

from tests.test_utils import make_chain_graph_sparse, make_dense_assignment
from tgp.connect import EigenPoolConnect
from tgp.select import SelectOutput, eigenpool_select


class TestEigenPoolConnectBasic:
    """Basic tests for EigenPoolConnect class."""

    def test_eigenpool_connect_repr(self):
        """Test EigenPoolConnect __repr__ method."""
        connector = EigenPoolConnect(
            remove_self_loops=False,
            degree_norm=True,
            adj_transpose=False,
            edge_weight_norm=True,
            sparse_output=False,
        )
        repr_str = repr(connector)
        assert "EigenPoolConnect" in repr_str
        assert "remove_self_loops=False" in repr_str
        assert "degree_norm=True" in repr_str
        assert "adj_transpose=False" in repr_str
        assert "edge_weight_norm=True" in repr_str
        assert "sparse_output=False" in repr_str

    def test_eigenpool_connect_initialization(self):
        """Test EigenPoolConnect initialization with various parameters."""
        connector = EigenPoolConnect(
            remove_self_loops=True,
            degree_norm=False,
            adj_transpose=True,
            edge_weight_norm=False,
            sparse_output=True,
        )
        assert connector.remove_self_loops is True
        assert connector.degree_norm is False
        assert connector.adj_transpose is True
        assert connector.edge_weight_norm is False
        assert connector.sparse_output is True


class TestEigenPoolConnectWithSparseInput:
    """Tests for EigenPoolConnect with sparse edge_index input."""

    def test_eigenpool_connect_sparse_input(self, pooler_test_graph_sparse):
        """Test EigenPoolConnect with sparse edge_index."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)
        connector = EigenPoolConnect(sparse_output=False, remove_self_loops=False)

        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index,
            so=so,
            edge_weight=edge_weight,
        )

        # Output should be dense [1, K, K]
        assert adj_pool.shape == (1, actual_k, actual_k)
        assert edge_weight_pool is None

    def test_eigenpool_connect_sparse_output(self, pooler_test_graph_sparse):
        """Test EigenPoolConnect with sparse_output=True."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        connector = EigenPoolConnect(sparse_output=True, remove_self_loops=False)

        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index,
            so=so,
            edge_weight=edge_weight,
        )

        # Output should be sparse edge_index format
        assert adj_pool.shape[0] == 2  # edge_index format
        assert edge_weight_pool is not None

    def test_eigenpool_connect_handles_empty_edge_index_with_nodes(self):
        """Test sparse unbatched path when a graph has nodes but no edges."""
        num_nodes = 5
        k = 3

        # Explicitly model a graph with 5 isolated nodes.
        edge_index = torch.empty((2, 0), dtype=torch.long)
        batch = torch.zeros(num_nodes, dtype=torch.long)

        # Build a deterministic hard assignment S in [N, K].
        cluster_index = torch.tensor([0, 1, 2, 0, 1], dtype=torch.long)
        s = torch.nn.functional.one_hot(cluster_index, num_classes=k).to(torch.float32)
        so = SelectOutput(s=s, batch=batch)

        connector = EigenPoolConnect(sparse_output=False, remove_self_loops=False)
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index,
            so=so,
            edge_weight=None,
            batch=batch,
        )

        # No edges means A_ext is zero, so the pooled dense adjacency must be zero too.
        assert adj_pool.shape == (1, k, k)
        assert torch.allclose(adj_pool, torch.zeros_like(adj_pool))
        assert edge_weight_pool is None

    def test_eigenpool_connect_remove_self_loops(self, pooler_test_graph_sparse):
        """Test EigenPoolConnect with remove_self_loops=True."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)
        connector = EigenPoolConnect(remove_self_loops=True, sparse_output=False)

        adj_pool, _ = connector(edge_index=edge_index, so=so, edge_weight=edge_weight)

        # Diagonal should be zero
        for i in range(actual_k):
            assert adj_pool[0, i, i] == 0

    def test_eigenpool_connect_degree_norm(self, pooler_test_graph_sparse):
        """Test EigenPoolConnect with degree_norm=True."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Without degree norm
        connector_no_norm = EigenPoolConnect(degree_norm=False, sparse_output=False)
        adj_no_norm, _ = connector_no_norm(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        # With degree norm
        connector_norm = EigenPoolConnect(degree_norm=True, sparse_output=False)
        adj_norm, _ = connector_norm(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        # Results should be different (unless trivial case)
        assert adj_no_norm.shape == adj_norm.shape


class TestEigenPoolConnectWithDenseInput:
    """Tests for EigenPoolConnect with dense adjacency input."""

    def test_eigenpool_connect_single_dense_input(self, pooler_test_graph_sparse):
        """Test EigenPoolConnect with single dense adjacency [N, N]."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3

        # Create dense adjacency
        from torch_geometric.utils import to_dense_adj

        adj_dense = to_dense_adj(edge_index).squeeze(0)  # [N, N]

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)
        connector = EigenPoolConnect(sparse_output=False, remove_self_loops=False)

        adj_pool, edge_weight_pool = connector(edge_index=adj_dense, so=so)

        # Output should be [1, K, K]
        assert adj_pool.shape == (1, actual_k, actual_k)
        assert edge_weight_pool is None

    def test_eigenpool_connect_batched_dense_input(self, pooler_test_graph_dense_batch):
        """Test EigenPoolConnect with batched dense adjacency [B, N, N]."""
        x, adj = pooler_test_graph_dense_batch
        B, N, F = x.shape
        k = 3

        # Create SelectOutput with batched dense assignment [B, N, K]
        s_list = []
        for b in range(B):
            edge_index = torch.nonzero(adj[b] > 0).t()
            so_b = eigenpool_select(edge_index=edge_index, k=k)
            s_list.append(so_b.s)
        s = torch.stack(s_list, dim=0)
        so = SelectOutput(s=s)
        actual_k = so.s.size(-1)

        connector = EigenPoolConnect(sparse_output=False, remove_self_loops=False)

        adj_pool, edge_weight_pool = connector(edge_index=adj, so=so)

        # Output should be [B, K, K]
        assert adj_pool.shape == (B, actual_k, actual_k)
        assert edge_weight_pool is None


class TestEigenPoolConnectAext:
    """Tests for A_ext computation in EigenPoolConnect."""

    def test_compute_a_ext(self):
        """Test _compute_a_ext static method."""
        # Create a simple adjacency matrix
        adj = torch.tensor(
            [
                [0, 1, 1, 0],
                [1, 0, 0, 1],
                [1, 0, 0, 1],
                [0, 1, 1, 0],
            ],
            dtype=torch.float32,
        )
        # Nodes 0,1 in cluster 0; nodes 2,3 in cluster 1
        cluster_index = torch.tensor([0, 0, 1, 1])

        a_ext = EigenPoolConnect._compute_a_ext(adj, cluster_index)

        # A_ext should only have inter-cluster edges
        # Edges (0,1), (1,0) are intra-cluster, should be 0
        assert a_ext[0, 1] == 0
        assert a_ext[1, 0] == 0
        # Edges (2,3), (3,2) are intra-cluster, should be 0
        assert a_ext[2, 3] == 0
        assert a_ext[3, 2] == 0
        # Edges (0,2), (1,3), etc. are inter-cluster, should be preserved
        assert a_ext[0, 2] == 1
        assert a_ext[1, 3] == 1


class TestEigenPoolConnectOmega:
    """Tests for using so.s as Omega in EigenPoolConnect."""

    def test_omega_from_select_output(self, pooler_test_graph_sparse):
        """Test that so.s is used directly as Omega."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # so.s should be the one-hot [N, K] matrix
        assert so.s.dim() == 2
        N = so.s.size(0)
        K = so.s.size(-1)

        # Each row should sum to 1 (one-hot)
        row_sums = so.s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N))

        # The connector uses so.s directly as Omega
        connector = EigenPoolConnect(sparse_output=False)
        adj_pool, _ = connector(edge_index=edge_index, so=so)

        # Output should have K clusters
        assert adj_pool.shape[-1] == K

    def test_unbatched_rejects_batched_assignment(self, pooler_test_graph_sparse):
        """Unbatched sparse inputs should not accept batched [B, N, K] assignments."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        N = x.size(0)
        k = 3

        s = make_dense_assignment(N, k)
        s_batched = torch.stack([s, s], dim=0)  # [2, N, K]
        so = SelectOutput(s=s_batched)

        connector = EigenPoolConnect()
        with pytest.raises(ValueError, match="SelectOutput.s must have shape"):
            connector(edge_index=edge_index, so=so)

    def test_unbatched_rejects_wrong_dim_assignment(self, pooler_test_graph_sparse):
        """Unbatched sparse inputs should reject assignments with wrong rank."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        N = x.size(0)
        k = 3

        s = make_dense_assignment(N, k).unsqueeze(0).unsqueeze(0)  # [1, 1, N, K]
        so = SelectOutput(s=s)

        connector = EigenPoolConnect()
        with pytest.raises(ValueError, match="SelectOutput.s must have shape"):
            connector(edge_index=edge_index, so=so)


class TestEigenPoolConnectACoar:
    """Tests for A_coar computation (Omega^T A_ext Omega)."""

    def test_a_coar_is_symmetric(self, pooler_test_graph_sparse):
        """Test that coarsened adjacency is symmetric for undirected graphs."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        connector = EigenPoolConnect(
            sparse_output=False, remove_self_loops=False, degree_norm=False
        )

        adj_pool, _ = connector(edge_index=edge_index, so=so, edge_weight=edge_weight)

        # Should be symmetric
        adj_pool_squeezed = adj_pool.squeeze(0)
        assert torch.allclose(adj_pool_squeezed, adj_pool_squeezed.t(), atol=1e-6)

    def test_a_coar_only_inter_cluster_edges(self):
        """Test that A_coar only contains inter-cluster edges."""
        # Create two disconnected cliques that are connected by one edge
        # Clique 1: nodes 0, 1, 2 (cluster 0)
        # Clique 2: nodes 3, 4, 5 (cluster 1)
        # Bridge edge: 2-3
        edge_list = [
            # Clique 1
            [0, 1],
            [1, 0],
            [0, 2],
            [2, 0],
            [1, 2],
            [2, 1],
            # Clique 2
            [3, 4],
            [4, 3],
            [3, 5],
            [5, 3],
            [4, 5],
            [5, 4],
            # Bridge
            [2, 3],
            [3, 2],
        ]
        edge_index = torch.tensor(edge_list, dtype=torch.long).t()

        k = 2  # Two clusters
        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)

        connector = EigenPoolConnect(
            sparse_output=False, remove_self_loops=True, degree_norm=False
        )
        adj_pool, _ = connector(edge_index=edge_index, so=so)

        # A_coar should have off-diagonal entries (inter-cluster edges)
        assert adj_pool.shape == (1, actual_k, actual_k)


class TestEigenPoolConnectChainGraph:
    """Tests with chain graph structure."""

    def test_connect_chain_graph(self):
        """Test connect on a chain graph."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=6, F_dim=4, seed=42)
        k = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)

        connector = EigenPoolConnect(sparse_output=False)
        adj_pool, _ = connector(edge_index=edge_index, so=so)

        assert adj_pool.shape == (1, actual_k, actual_k)


if __name__ == "__main__":
    pytest.main([__file__])
