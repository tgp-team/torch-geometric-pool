import pytest
import torch

from tests.test_utils import make_chain_graph_sparse
from tgp.poolers import EigenPooling, get_pooler, pooler_map


class TestEigenPoolingBasic:
    """Basic tests for EigenPooling class."""

    def test_eigenpool_in_pooler_map(self):
        """Test that EigenPooling is registered in pooler_map."""
        assert "eigen" in pooler_map
        assert pooler_map["eigen"] is EigenPooling

    def test_eigenpool_get_pooler(self):
        """Test that EigenPooling can be instantiated via get_pooler."""
        pooler = get_pooler("eigen", k=3, num_modes=2)
        assert isinstance(pooler, EigenPooling)
        assert pooler.k == 3
        assert pooler.num_modes == 2

    def test_eigenpool_repr(self):
        """Test EigenPooling __repr__ method."""
        pooler = EigenPooling(k=5, num_modes=3, normalized=True)
        repr_str = repr(pooler)
        assert "EigenPooling" in repr_str
        assert "k=5" in repr_str
        assert "num_modes=3" in repr_str

    def test_eigenpool_initialization(self):
        """Test EigenPooling initialization with various parameters."""
        pooler = EigenPooling(
            k=4,
            num_modes=5,
            normalized=True,
            cached=True,
            remove_self_loops=False,
            degree_norm=False,
            batched=False,
        )
        assert pooler.k == 4
        assert pooler.num_modes == 5
        assert pooler.normalized is True
        assert pooler.cached is True


class TestEigenPoolingUnbatched:
    """Tests for EigenPooling with unbatched (sparse) inputs."""

    def test_eigenpool_unbatched_forward(self, pooler_test_graph_sparse):
        """Test unbatched forward pass with sparse edge_index."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        out = pooler(x=x, adj=edge_index)

        # Check output shapes
        actual_k = out.so.s.size(-1)  # Now s is [N, K]
        # Output x should have shape [K, H*d] where H may be capped
        assert out.x.shape[0] == actual_k
        assert out.x.shape[1] % F == 0  # H*d is multiple of F
        assert out.so is not None

    def test_eigenpool_unbatched_precoarsening(self, pooler_test_graph_sparse):
        """Test precoarsening with unbatched inputs."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        precoarsened = pooler.precoarsening(edge_index=edge_index)

        # Check precoarsening output
        assert precoarsened.so is not None
        assert precoarsened.edge_index is not None
        actual_k = precoarsened.so.s.size(-1)
        assert precoarsened.batch.shape == (actual_k,)

    def test_eigenpool_unbatched_reduce_with_precoarsened(
        self, pooler_test_graph_sparse
    ):
        """Test reduce operation with precomputed SelectOutput."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        precoarsened = pooler.precoarsening(edge_index=edge_index)

        # Test reduce with precomputed so (must pass edge_index and batch)
        x_pooled, batch_pooled = pooler.reduce(
            x=x, so=precoarsened.so, edge_index=edge_index, batch=batch
        )
        actual_k = precoarsened.so.s.size(-1)
        assert x_pooled.shape[0] == actual_k
        assert x_pooled.shape[1] % F == 0  # H*d
        assert batch_pooled.shape == (actual_k,)

    def test_eigenpool_unbatched_lift(self, pooler_test_graph_sparse):
        """Test lift operation to recover original node features."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        out = pooler(x=x, adj=edge_index)

        # Test lifting (must pass edge_index)
        x_lifted = pooler(x=out.x, so=out.so, lifting=True, edge_index=edge_index)
        assert x_lifted.shape == (N, F)

    def test_eigenpool_unbatched_full_cycle(self, pooler_test_graph_sparse):
        """Test full pool -> GNN -> lift cycle."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)

        # Precoarsen
        precoarsened = pooler.precoarsening(edge_index=edge_index)
        actual_k = precoarsened.so.s.size(-1)

        # Reduce (pass edge_index)
        x_pooled, batch_pooled = pooler.reduce(
            x=x, so=precoarsened.so, edge_index=edge_index
        )
        assert x_pooled.shape[0] == actual_k

        # Lift (pass edge_index)
        x_lifted = pooler.lift(
            x_pool=x_pooled, so=precoarsened.so, edge_index=edge_index
        )
        assert x_lifted.shape == (N, F)


class TestEigenPoolingPrecoarsening:
    """Tests for EigenPooling precoarsening functionality."""

    def test_precoarsening_requires_edge_index(self):
        """Test that precoarsening raises ValueError when edge_index is None."""
        pooler = EigenPooling(k=3, num_modes=2, batched=False)

        with pytest.raises(ValueError, match="edge_index cannot be None"):
            pooler.precoarsening(edge_index=None)

    def test_precoarsening_with_batch(self, pooler_test_graph_sparse_batch):
        """Test precoarsening with multi-graph batch."""
        data_batch = pooler_test_graph_sparse_batch
        k = 2
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        precoarsened = pooler.precoarsening(
            edge_index=data_batch.edge_index,
            batch=data_batch.batch,
        )

        assert precoarsened.so is not None
        assert precoarsened.batch is not None

    def test_precoarsening_without_batch(self, pooler_test_graph_sparse):
        """Test precoarsening without explicit batch (single graph)."""
        x, edge_index, edge_weight, _ = pooler_test_graph_sparse
        k = 3
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        precoarsened = pooler.precoarsening(edge_index=edge_index)

        # Should treat as single graph
        actual_k = precoarsened.so.s.size(-1)
        assert precoarsened.batch is not None
        assert precoarsened.batch.shape == (actual_k,)
        assert torch.all(precoarsened.batch == 0)

    def test_precoarsening_fixed_k_for_small_graph(self):
        """Test that EigenPooling.precoarsening keeps exactly k clusters."""
        _, edge_index, _, _ = make_chain_graph_sparse(N=4, F_dim=2, seed=42)
        k = 5

        pooler = EigenPooling(k=k, num_modes=2, batched=False)
        precoarsened = pooler.precoarsening(edge_index=edge_index)

        assert precoarsened.so.s.shape == (4, k)
        assert precoarsened.batch.shape == (k,)


class TestEigenPoolingEdgeCases:
    """Tests for edge cases in EigenPooling."""

    def test_eigenpool_k_larger_than_nodes(self):
        """Test EigenPooling when k >= num_nodes."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=5, F_dim=3, seed=42)
        k = 10  # Larger than num_nodes
        num_modes = 2

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        out = pooler(x=x, adj=edge_index)

        # Should handle gracefully (k capped at num_nodes)
        assert out.x is not None
        actual_k = out.so.s.size(-1)
        assert actual_k <= 5

    def test_eigenpool_single_cluster(self):
        """Test EigenPooling with k=1."""
        x, edge_index, _, _ = make_chain_graph_sparse(N=6, F_dim=3, seed=42)
        N, F = x.shape
        k = 1
        num_modes = 3

        pooler = EigenPooling(k=k, num_modes=num_modes, batched=False)
        out = pooler(x=x, adj=edge_index)

        # Should produce single supernode with H*d features
        assert out.x.shape[0] == 1
        assert out.x.shape[1] % F == 0  # H*d is multiple of F


class TestEigenPoolingExtraRepr:
    """Tests for EigenPooling extra_repr_args method."""

    def test_extra_repr_args(self):
        """Test extra_repr_args returns expected dict."""
        pooler = EigenPooling(k=5, num_modes=3, normalized=True, cached=False)
        extra = pooler.extra_repr_args()

        assert extra["k"] == 5
        assert extra["num_modes"] == 3
        assert extra["normalized"] is True
        assert extra["cached"] is False


class TestEigenPoolingSelectOutput:
    """Tests for SelectOutput structure with new design."""

    def test_select_output_is_one_hot(self, pooler_test_graph_sparse):
        """Test that so.s is a standard one-hot [N, K] matrix."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        N, F = x.shape
        k = 3

        pooler = EigenPooling(k=k, num_modes=2, batched=False)
        out = pooler(x=x, adj=edge_index)

        # s should be [N, K] one-hot
        assert out.so.s.shape[0] == N
        assert out.so.s.shape[1] <= k

        # Each row should sum to 1
        row_sums = out.so.s.sum(dim=-1)
        assert torch.allclose(row_sums, torch.ones(N))

    def test_num_supernodes_equals_k(self, pooler_test_graph_sparse):
        """Test that num_supernodes equals s.size(-1)."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        k = 3

        pooler = EigenPooling(k=k, num_modes=2, batched=False)
        out = pooler(x=x, adj=edge_index)

        # num_supernodes should equal K (the actual number of clusters)
        assert out.so.num_supernodes == out.so.s.size(-1)


if __name__ == "__main__":
    pytest.main([__file__])
