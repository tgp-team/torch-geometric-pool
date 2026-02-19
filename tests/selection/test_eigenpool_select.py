import pytest
import torch

from tests.test_utils import make_chain_graph_sparse, make_grid_graph_sparse
from tgp.select import EigenPoolSelect, SelectOutput, eigenpool_select


class TestEigenPoolSelectFunction:
    """Tests for the eigenpool_select function."""

    def test_eigenpool_select_basic(self, pooler_test_graph_sparse):
        """Test basic eigenpool_select functionality."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N = x.shape[0]
        k = 3

        so = eigenpool_select(
            edge_index=edge_index,
            k=k,
        )

        # Check SelectOutput structure
        assert isinstance(so, SelectOutput)
        # Now s is standard [N, K] one-hot
        assert so.s.shape[0] == N
        assert so.s.shape[1] <= k  # May be fewer if graph is small
        # No extra attributes
        assert (
            not hasattr(so, "eigenpool_cluster_index")
            or so.eigenpool_cluster_index is None
        )
        assert not hasattr(so, "num_modes") or so.num_modes is None

    def test_eigenpool_select_one_hot_matrix(self, pooler_test_graph_sparse):
        """Test that s is a valid one-hot matrix."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        N = x.shape[0]
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Each row should sum to 1 (one-hot)
        row_sums = so.s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N))

        # Each row should have exactly one 1 and rest 0s
        assert torch.all((so.s == 0) | (so.s == 1))

    def test_eigenpool_select_cluster_assignment(self, pooler_test_graph_sparse):
        """Test that cluster assignments can be derived from s."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Derive cluster index from one-hot matrix
        cluster_index = so.s.argmax(dim=-1)
        num_clusters = so.s.size(-1)

        # Cluster indices should be in valid range
        assert cluster_index.min() >= 0
        assert cluster_index.max() < num_clusters

    def test_eigenpool_select_with_edge_weight(self, pooler_test_graph_sparse):
        """Test eigenpool_select with edge weights."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(
            edge_index=edge_index,
            k=k,
            edge_weight=edge_weight,
        )

        assert so is not None
        assert so.s is not None

    def test_eigenpool_select_with_batch(self, pooler_test_graph_sparse):
        """Test eigenpool_select with batch vector."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(
            edge_index=edge_index,
            k=k,
            batch=batch,
        )

        assert so.batch is batch

    def test_eigenpool_select_uses_num_nodes_for_edgeless_graph(self):
        """Test that num_nodes is honored when edge_index is empty."""
        edge_index = torch.empty((2, 0), dtype=torch.long)
        num_nodes = 4

        so = eigenpool_select(edge_index=edge_index, k=1, num_nodes=num_nodes)

        assert so.s.shape == (num_nodes, 1)
        assert hasattr(so, "theta")
        assert so.theta.shape[0] == num_nodes

    def test_eigenpool_select_batched_handles_graph_without_edges(self):
        """Test batched mode when one graph has no edges."""
        # Graph 0: two nodes with one undirected edge.
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        # Graph 1: three isolated nodes (no edges).
        batch = torch.tensor([0, 0, 1, 1, 1], dtype=torch.long)

        so = eigenpool_select(edge_index=edge_index, k=2, batch=batch)

        assert so.s.shape == (5, 2)
        assert isinstance(so.theta, list)
        assert len(so.theta) == 2
        assert so.theta[0].shape[0] == 2
        assert so.theta[1].shape[0] == 3

    def test_eigenpool_select_k_larger_than_nodes(self):
        """Test eigenpool_select when k >= num_nodes."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=4, F_dim=2, seed=42)
        k = 10  # Larger than num_nodes

        so = eigenpool_select(edge_index=edge_index, k=k)

        # k should be capped at num_nodes
        num_clusters = so.s.size(-1)
        assert num_clusters <= 4

    def test_eigenpool_select_k_equals_one(self):
        """Test eigenpool_select with k=1 (single cluster)."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=6, F_dim=2, seed=42)
        N = x.shape[0]
        k = 1

        so = eigenpool_select(edge_index=edge_index, k=k)

        assert so.s.size(-1) == 1
        assert so.s.shape == (N, 1)
        # All nodes should be in cluster 0 (all ones in first column)
        assert torch.all(so.s[:, 0] == 1)

    def test_eigenpool_select_s_inv_computed(self, pooler_test_graph_sparse):
        """Test that s_inv is properly computed."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # s_inv should be transpose for one-hot matrix
        assert so.s_inv is not None
        assert so.s_inv.shape == (so.s.size(-1), so.s.size(0))  # [K, N]


class TestEigenPoolSelectClass:
    """Tests for the EigenPoolSelect class."""

    def test_eigenpool_select_class_forward(self, pooler_test_graph_sparse):
        """Test EigenPoolSelect forward method."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N = x.shape[0]
        k = 3

        selector = EigenPoolSelect(k=k)
        so = selector(edge_index=edge_index)

        assert isinstance(so, SelectOutput)
        assert so.s.shape[0] == N
        assert so.s.shape[1] <= k

    def test_eigenpool_select_class_repr(self):
        """Test EigenPoolSelect __repr__ method."""
        selector = EigenPoolSelect(k=5)
        repr_str = repr(selector)

        assert "EigenPoolSelect" in repr_str
        assert "k=5" in repr_str

    def test_eigenpool_select_class_is_dense(self):
        """Test that EigenPoolSelect.is_dense is True."""
        selector = EigenPoolSelect(k=3)
        assert selector.is_dense is True

    def test_eigenpool_select_class_requires_edge_index(self):
        """Test that invalid edge_index input raises from connectivity parsing."""
        selector = EigenPoolSelect(k=3)

        with pytest.raises(NotImplementedError):
            selector(edge_index=None)

    def test_eigenpool_select_class_x_unused(self, pooler_test_graph_sparse):
        """Test that x parameter is unused but accepted."""
        x, edge_index, _, _ = pooler_test_graph_sparse

        selector = EigenPoolSelect(k=3)
        # Should work with x=None
        import numpy as np

        torch.manual_seed(0)
        np.random.seed(0)
        so1 = selector(edge_index=edge_index, x=None)
        # Should also work with x provided
        torch.manual_seed(0)
        np.random.seed(0)
        so2 = selector(edge_index=edge_index, x=x)

        # Results should be equivalent since x is unused (allow label permutation)
        assert torch.allclose(so1.s @ so1.s.t(), so2.s @ so2.s.t())


class TestEigenPoolSelectGridGraph:
    """Tests for EigenPoolSelect with grid graphs."""

    def test_eigenpool_select_grid_graph(self):
        """Test EigenPoolSelect on a grid graph."""
        x, edge_index, _, _ = make_grid_graph_sparse(rows=3, cols=3, F_dim=2, seed=42)
        N = x.shape[0]  # 9 nodes
        k = 4

        so = eigenpool_select(edge_index=edge_index, k=k)

        assert so.s.shape[0] == N
        assert so.s.shape[1] <= k


class TestEigenPoolSelectNumSupernodes:
    """Tests for num_supernodes property with new design."""

    def test_num_supernodes_matches_s_columns(self, pooler_test_graph_sparse):
        """Test that num_supernodes equals s.size(-1)."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        # With standard [N, K] matrix, num_supernodes = K
        assert so.num_supernodes == so.s.size(-1)


if __name__ == "__main__":
    pytest.main([__file__])
