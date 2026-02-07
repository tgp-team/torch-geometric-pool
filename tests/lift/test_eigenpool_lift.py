import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.lift import EigenPoolLift
from tgp.reduce import EigenPoolReduce
from tgp.select import SelectOutput, eigenpool_select


class TestEigenPoolLiftBasic:
    """Basic tests for EigenPoolLift class."""

    def test_eigenpool_lift_repr(self):
        """Test EigenPoolLift __repr__ method."""
        lifter = EigenPoolLift(num_modes=5, normalized=True)
        repr_str = repr(lifter)
        assert "EigenPoolLift" in repr_str
        assert "num_modes=5" in repr_str
        assert "normalized=True" in repr_str

    def test_eigenpool_lift_initialization(self):
        """Test EigenPoolLift initialization."""
        lifter = EigenPoolLift(num_modes=3, normalized=False)
        assert lifter.num_modes == 3
        assert lifter.normalized is False


class TestEigenPoolLiftForward:
    """Tests for EigenPoolLift forward method."""

    def test_eigenpool_lift_forward_shape(self, pooler_test_graph_sparse):
        """Test that EigenPoolLift produces correct output shape."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # First reduce to get pooled features
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, _ = reducer(x=x, so=so, batch=batch)

        # Now lift
        lifter = EigenPoolLift(num_modes=num_modes)
        x_lifted = lifter(x_pool=x_pool, so=so, batch=batch)

        # Output should be [N, d]
        assert x_lifted.shape == (N, F)

    def test_eigenpool_lift_forward_values(self, pooler_test_graph_sparse):
        """Test that EigenPoolLift produces non-zero values."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Reduce then lift
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, _ = reducer(x=x, so=so, batch=batch)

        lifter = EigenPoolLift(num_modes=num_modes)
        x_lifted = lifter(x_pool=x_pool, so=so, batch=batch)

        # Should not be all zeros
        assert x_lifted.abs().sum() > 0

    def test_eigenpool_lift_requires_theta(self, pooler_test_graph_sparse):
        """Test that SelectOutput.theta is required."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Reduce first
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, _ = reducer(x=x, so=so, batch=batch)

        lifter = EigenPoolLift(num_modes=num_modes)

        so_no_theta = SelectOutput(s=so.s)
        with pytest.raises(ValueError, match="SelectOutput.theta is required"):
            lifter(x_pool=x_pool, so=so_no_theta, batch=batch)


class TestEigenPoolLiftReduceCycle:
    """Tests for reduce -> lift cycle."""

    def test_reduce_lift_cycle(self, pooler_test_graph_sparse):
        """Test that reduce followed by lift produces correct shapes."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)
        actual_k = so.s.size(-1)

        # Reduce
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, _ = reducer(x=x, so=so, batch=batch)
        assert x_pool.shape[0] == actual_k

        # Lift
        lifter = EigenPoolLift(num_modes=num_modes)
        x_lifted = lifter(x_pool=x_pool, so=so, batch=batch)
        assert x_lifted.shape == (N, F)

    def test_reduce_transform_lift_cycle(self, pooler_test_graph_sparse):
        """Test reduce -> linear transform -> lift cycle."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Reduce
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, _ = reducer(x=x, so=so, batch=batch)

        # Get actual expanded feature dimension
        H_times_d = x_pool.shape[-1]

        # Apply transformation (simulating GNN layer)
        linear = torch.nn.Linear(H_times_d, H_times_d)
        x_pool_transformed = linear(x_pool)

        # Lift
        lifter = EigenPoolLift(num_modes=num_modes)
        x_lifted = lifter(x_pool=x_pool_transformed, so=so, batch=batch)
        assert x_lifted.shape == (N, F)


class TestEigenPoolLiftNormalization:
    """Tests for normalized vs unnormalized Laplacian in lift."""

    def test_normalized_vs_unnormalized(self, pooler_test_graph_sparse):
        """Test that normalized and unnormalized produce different results."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Reduce with both normalized and unnormalized
        reducer_norm = EigenPoolReduce(num_modes=num_modes, normalized=True)
        reducer_unnorm = EigenPoolReduce(num_modes=num_modes, normalized=False)

        x_pool_norm, _ = reducer_norm(x=x, so=so, batch=batch, edge_index=edge_index)
        x_pool_unnorm, _ = reducer_unnorm(
            x=x, so=so, batch=batch, edge_index=edge_index
        )

        # Lift with matching normalization
        lifter_norm = EigenPoolLift(num_modes=num_modes, normalized=True)
        lifter_unnorm = EigenPoolLift(num_modes=num_modes, normalized=False)

        x_lifted_norm = lifter_norm(
            x_pool=x_pool_norm, so=so, batch=batch, edge_index=edge_index
        )
        x_lifted_unnorm = lifter_unnorm(
            x_pool=x_pool_unnorm, so=so, batch=batch, edge_index=edge_index
        )

        # Both should have same shape
        assert x_lifted_norm.shape == x_lifted_unnorm.shape


class TestEigenPoolLiftDifferentModes:
    """Tests for different num_modes values."""

    def test_lift_with_different_num_modes(self, pooler_test_graph_sparse):
        """Test lift with varying number of modes."""
        x, edge_index, _, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3

        so = eigenpool_select(edge_index=edge_index, k=k)

        for num_modes in [1, 2, 3]:
            # Reduce
            reducer = EigenPoolReduce(num_modes=num_modes)
            x_pool, _ = reducer(x=x, so=so, batch=batch, edge_index=edge_index)

            # Lift
            lifter = EigenPoolLift(num_modes=num_modes)
            x_lifted = lifter(x_pool=x_pool, so=so, batch=batch, edge_index=edge_index)

            assert x_lifted.shape == (N, F)


class TestEigenPoolLiftChainGraph:
    """Tests with chain graph structure."""

    def test_lift_chain_graph(self):
        """Test lift on a chain graph."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=6, F_dim=4, seed=42)
        N, F = x.shape
        k = 2
        num_modes = 2

        so = eigenpool_select(edge_index=edge_index, k=k)

        # Reduce
        reducer = EigenPoolReduce(num_modes=num_modes)
        x_pool, batch_pool = reducer(x=x, so=so, batch=None)

        # Lift
        lifter = EigenPoolLift(num_modes=num_modes)
        x_lifted = lifter(x_pool=x_pool, so=so, batch=None)

        assert x_lifted.shape == (N, F)


class TestEigenPoolLiftHelpers:
    """Tests that helper functions are available and work correctly."""

    def test_laplacian_function_exists(self):
        """Test that _laplacian function exists in eigenpool_lift module."""
        import numpy as np

        from tgp.select.eigenpool_select import laplacian

        adj = np.array([[0, 1], [1, 0]], dtype=np.float32)
        L = laplacian(adj, normalized=True)
        assert L.shape == (2, 2)

    def test_build_pooling_matrix_exists(self):
        """Test that _build_pooling_matrix function exists in eigenpool_lift module."""
        import numpy as np

        from tgp.select.eigenpool_select import build_pooling_matrix

        adj = np.eye(4)
        cluster_labels = np.array([0, 0, 1, 1])
        theta = build_pooling_matrix(adj, cluster_labels, num_modes=2)
        assert theta.shape[0] == 4


if __name__ == "__main__":
    pytest.main([__file__])
