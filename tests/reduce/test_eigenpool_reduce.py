import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.reduce import EigenPoolReduce
from tgp.select import SelectOutput, eigenpool_select


class TestEigenPoolReduceBasic:
    """Basic tests for EigenPoolReduce class."""

    def test_eigenpool_reduce_repr(self):
        """Test EigenPoolReduce __repr__ method."""
        reducer = EigenPoolReduce(num_modes=5)
        repr_str = repr(reducer)
        assert "EigenPoolReduce" in repr_str
        assert "num_modes=5" in repr_str

    def test_eigenpool_reduce_initialization(self):
        """Test EigenPoolReduce initialization."""
        reducer = EigenPoolReduce(num_modes=3)
        assert reducer.num_modes == 3


class TestEigenPoolReduceForward:
    """Tests for EigenPoolReduce forward method."""

    def test_eigenpool_reduce_forward_shape(self, pooler_test_graph_sparse):
        """Test that EigenPoolReduce produces correct output shape."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        reducer = EigenPoolReduce(num_modes=num_modes)

        x_pool, batch_pool = reducer(x=x, so=so, batch=batch)

        # Output should be [K, H*d]
        actual_k = so.s.size(-1)
        assert x_pool.shape[0] == actual_k
        # H may be capped based on cluster sizes
        assert x_pool.shape[1] % F == 0

    def test_eigenpool_reduce_forward_values(self, pooler_test_graph_sparse):
        """Test that EigenPoolReduce produces non-zero values."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        reducer = EigenPoolReduce(num_modes=num_modes)

        x_pool, _ = reducer(x=x, so=so, batch=batch)

        # Should not be all zeros
        assert x_pool.abs().sum() > 0

    def test_eigenpool_reduce_requires_theta(self, pooler_test_graph_sparse):
        """Test that missing theta fails when reduce is called directly."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        so_no_theta = SelectOutput(s=so.s)  # drop theta
        reducer = EigenPoolReduce(num_modes=2)

        with pytest.raises(AttributeError):
            reducer(x=x, so=so_no_theta, batch=batch)

    def test_eigenpool_reduce_different_num_modes(self, pooler_test_graph_sparse):
        """Test EigenPoolReduce with different num_modes values."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)

        for num_modes in [1, 2, 3, 4]:
            reducer = EigenPoolReduce(num_modes=num_modes)
            x_pool, _ = reducer(x=x, so=so, batch=batch)

            # num_modes may be capped based on cluster sizes
            assert x_pool.shape[0] == actual_k
            # Feature dim should be multiple of F
            assert x_pool.shape[1] % F == 0


class TestEigenPoolReduceBatch:
    """Tests for EigenPoolReduce reduce_batch method."""

    def test_reduce_batch_single_graph(self, pooler_test_graph_sparse):
        """Test reduce_batch for single graph (batch all zeros)."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)

        batch_pool = EigenPoolReduce.reduce_batch(so, batch)

        # Should have K elements, all with batch index 0
        assert batch_pool.shape == (actual_k,)
        assert torch.all(batch_pool == 0)

    def test_reduce_batch_none(self, pooler_test_graph_sparse):
        """Test reduce_batch when batch is None."""
        x, edge_index, _, _ = pooler_test_graph_sparse
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        batch_pool = EigenPoolReduce.reduce_batch(so, None)

        assert batch_pool is None

    def test_reduce_batch_empty(self):
        """Test reduce_batch with empty batch."""
        # Create empty SelectOutput
        s = torch.empty((0, 4))
        so = SelectOutput(s=s)
        batch = torch.empty((0,), dtype=torch.long)

        batch_pool = EigenPoolReduce.reduce_batch(so, batch)

        assert batch_pool.shape == (0,)


class TestEigenPoolReduceReshape:
    """Tests for the reshape logic in EigenPoolReduce."""

    def test_reshape_dimensions(self, pooler_test_graph_sparse):
        """Test that output has correct [K, H*d] shape."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        reducer = EigenPoolReduce(num_modes=num_modes)

        x_pool, _ = reducer(x=x, so=so, batch=batch, edge_index=edge_index)

        # The output should have K rows
        actual_k = so.s.size(-1)
        assert x_pool.shape[0] == actual_k
        # Columns = H * F (feature expansion)
        assert x_pool.shape[1] >= F  # At least 1 mode


class TestEigenPoolReduceNormalization:
    """Tests for consistency across reducer instances."""

    def test_reducer_instances_same_shape(self, pooler_test_graph_sparse):
        """Test that equivalent reducer instances produce same output shape."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        reducer_a = EigenPoolReduce(num_modes=num_modes)
        reducer_b = EigenPoolReduce(num_modes=num_modes)

        x_pool_a, _ = reducer_a(x=x, so=so, batch=batch)
        x_pool_b, _ = reducer_b(x=x, so=so, batch=batch)

        # Results should have same shape
        assert x_pool_a.shape == x_pool_b.shape


class TestEigenPoolReduceHelpers:
    """Tests for helper functions in eigenpool_reduce module."""

    def test_build_pooling_matrix_shape(self, pooler_test_graph_sparse):
        """Test _build_pooling_matrix function."""
        import numpy as np

        from tgp.select.eigenpool_select import build_pooling_matrix

        x, edge_index, _, _ = pooler_test_graph_sparse
        N = x.shape[0]
        k = 3
        num_modes = 2

        # Create simple adjacency and cluster labels
        adj = np.eye(N) + np.diag(np.ones(N - 1), 1) + np.diag(np.ones(N - 1), -1)
        cluster_labels = np.array([i % k for i in range(N)])

        theta = build_pooling_matrix(adj, cluster_labels, num_modes)

        # Theta should be [N, K*H] where H may be capped
        assert theta.shape[0] == N
        assert theta.shape[1] > 0  # At least some columns

    def test_laplacian_function(self):
        """Test _laplacian function."""
        import numpy as np

        from tgp.select.eigenpool_select import laplacian

        # Simple 3-node chain
        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)

        L_norm = laplacian(adj, normalized=True)
        L_unnorm = laplacian(adj, normalized=False)

        assert L_norm.shape == (3, 3)
        assert L_unnorm.shape == (3, 3)
        # Unnormalized diagonal = degree
        assert abs(L_unnorm[0, 0] - 1.0) < 1e-6
        assert abs(L_unnorm[1, 1] - 2.0) < 1e-6

    def test_eigenvectors_function(self):
        """Test _eigenvectors function."""
        import numpy as np

        from tgp.select.eigenpool_select import eigenvectors, laplacian

        adj = np.array([[0, 1, 0], [1, 0, 1], [0, 1, 0]], dtype=np.float32)
        L = laplacian(adj, normalized=True)
        lamb, U = eigenvectors(L)

        assert lamb.shape == (3,)
        assert U.shape == (3, 3)
        # First eigenvalue should be ~0 for connected graph
        assert abs(lamb[0]) < 1e-5


class TestEigenPoolReduceChainGraph:
    """Tests with chain graph structure."""

    def test_reduce_chain_graph(self):
        """Test reduce on a chain graph."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=6, F_dim=4, seed=42)
        k = 2
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        reducer = EigenPoolReduce(num_modes=num_modes)

        x_pool, batch_pool = reducer(x=x, so=so, batch=None)

        actual_k = so.s.size(-1)
        assert x_pool.shape[0] == actual_k
        assert batch_pool is None


if __name__ == "__main__":
    pytest.main([__file__])
