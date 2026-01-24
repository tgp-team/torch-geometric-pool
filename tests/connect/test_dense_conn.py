"""Tests for tgp.connect.dense_conn module.

Tests the DenseConnect class for dense pooling methods.
"""

import pytest
import torch

from tgp.connect import DenseConnect
from tgp.select import SelectOutput


class TestDenseConnect:
    """Test the DenseConnect class for batched dense pooling."""

    def test_dense_connect_repr(self):
        """Test DenseConnect __repr__ method."""
        connector = DenseConnect(
            remove_self_loops=False,
            degree_norm=True,
            adj_transpose=False,
            edge_weight_norm=True,
            unbatched_output="batch",
        )
        repr_str = repr(connector)
        assert "DenseConnect" in repr_str
        assert "remove_self_loops=False" in repr_str
        assert "degree_norm=True" in repr_str
        assert "adj_transpose=False" in repr_str
        assert "edge_weight_norm=True" in repr_str
        assert 'unbatched_output="batch"' in repr_str

    def test_dense_connect_invalid_unbatched_output(self):
        """Test that DenseConnect raises ValueError for invalid unbatched_output."""
        with pytest.raises(ValueError, match="unbatched_output must be one of"):
            DenseConnect(unbatched_output="invalid")

    def test_dense_connect_with_batched_adjacency(self, pooler_test_graph_dense_batch):
        """Test DenseConnect with 3D batched adjacency matrices."""
        x, adj_dense = pooler_test_graph_dense_batch
        batch_size, num_nodes, num_features = x.shape
        num_supernodes = num_nodes // 2

        # Create dense assignment matrices (batch_size, num_nodes, num_supernodes)
        s_batched = torch.zeros(batch_size, num_nodes, num_supernodes)
        for b in range(batch_size):
            for i in range(num_nodes):
                cluster = i // 2
                if cluster < num_supernodes:
                    s_batched[b, i, cluster] = 1.0

        so = SelectOutput(s=s_batched)

        # Test without normalization
        connector = DenseConnect(edge_weight_norm=False)
        adj_orig, _ = connector(adj_dense, so)

        # Test with normalization
        connector_norm = DenseConnect(edge_weight_norm=True)
        adj_norm, _ = connector_norm(adj_dense, so)

        # Check that normalization happened per graph
        assert not torch.isnan(adj_norm).any(), "No NaN values should be present"
        assert not torch.isinf(adj_norm).any(), "No Inf values should be present"
        assert adj_norm.abs().max() <= 1.0 + 1e-6

        # Each graph (batch dimension) should be normalized independently
        for batch_idx in range(batch_size):
            graph_adj_orig = adj_orig[batch_idx]
            graph_adj_norm = adj_norm[batch_idx]
            max_val = graph_adj_orig.abs().max()
            if max_val > 0:
                expected_norm = graph_adj_orig / max_val
                torch.testing.assert_close(
                    graph_adj_norm, expected_norm, atol=1e-6, rtol=1e-6
                )

    def test_dense_connect_remove_self_loops(self, pooler_test_graph_dense):
        """Test DenseConnect with remove_self_loops=True."""
        x, adj = pooler_test_graph_dense
        B, N, F = x.shape
        k = N // 2

        # Create assignment matrix
        s = torch.zeros(B, N, k)
        for i in range(N):
            cluster = i // 2
            if cluster < k:
                s[0, i, cluster] = 1.0

        so = SelectOutput(s=s)

        # Test with remove_self_loops=True
        connector = DenseConnect(remove_self_loops=True)
        adj_pool, _ = connector(adj, so)

        # Check that diagonal is zero
        for b in range(B):
            assert torch.allclose(adj_pool[b].diagonal(), torch.zeros(k))

        # Test with remove_self_loops=False
        connector_no_remove = DenseConnect(remove_self_loops=False)
        adj_pool_no_remove, _ = connector_no_remove(adj, so)

        # Diagonal may be non-zero
        assert not torch.allclose(adj_pool_no_remove[0].diagonal(), torch.zeros(k))

    def test_dense_connect_degree_norm(self, pooler_test_graph_dense):
        """Test DenseConnect with degree_norm=True."""
        x, adj = pooler_test_graph_dense
        B, N, F = x.shape
        k = N // 2

        # Create assignment matrix
        s = torch.zeros(B, N, k)
        for i in range(N):
            cluster = i // 2
            if cluster < k:
                s[0, i, cluster] = 1.0

        so = SelectOutput(s=s)

        # Test without degree normalization
        connector = DenseConnect(degree_norm=False)
        adj_orig, _ = connector(adj, so)

        # Test with degree normalization
        connector_norm = DenseConnect(degree_norm=True)
        adj_norm, _ = connector_norm(adj, so)

        # Check that normalization was applied (values should be different)
        assert not torch.allclose(adj_orig, adj_norm)
        assert not torch.isnan(adj_norm).any()
        assert not torch.isinf(adj_norm).any()

    def test_dense_connect_adj_transpose(self, pooler_test_graph_dense):
        """Test DenseConnect with adj_transpose=True."""
        x, adj = pooler_test_graph_dense
        B, N, F = x.shape
        k = N // 2

        # Create assignment matrix
        s = torch.zeros(B, N, k)
        for i in range(N):
            cluster = i // 2
            if cluster < k:
                s[0, i, cluster] = 1.0

        so = SelectOutput(s=s)

        # Test with adj_transpose=False
        connector = DenseConnect(adj_transpose=False)
        adj_pool, _ = connector(adj, so)

        # Test with adj_transpose=True
        connector_transpose = DenseConnect(adj_transpose=True)
        adj_pool_transpose, _ = connector_transpose(adj, so)

        # Transposed version should be different (unless symmetric)
        assert adj_pool.shape == adj_pool_transpose.shape
        # For non-symmetric adj, transpose should make a difference
        if not torch.allclose(adj[0], adj[0].transpose(-2, -1)):
            assert not torch.allclose(adj_pool, adj_pool_transpose)

    def test_dense_connect_requires_select_output(self):
        """Test that DenseConnect raises ValueError when so is None."""
        connector = DenseConnect()
        adj = torch.randn(1, 4, 4)

        with pytest.raises(ValueError, match="SelectOutput is required"):
            connector(adj, so=None)

    def test_dense_connect_prepare_dense_inputs(self):
        """Test DenseConnect._prepare_dense_inputs static method."""
        # Test with 2D inputs (should add batch dimension)
        s_2d = torch.randn(4, 2)
        adj_2d = torch.randn(4, 4)
        s_prep, adj_prep = DenseConnect._prepare_dense_inputs(s_2d, adj_2d)
        assert s_prep.dim() == 3
        assert adj_prep.dim() == 3
        assert s_prep.size(0) == 1
        assert adj_prep.size(0) == 1

        # Test with 3D inputs (should pass through)
        s_3d = torch.randn(2, 4, 2)
        adj_3d = torch.randn(2, 4, 4)
        s_prep, adj_prep = DenseConnect._prepare_dense_inputs(s_3d, adj_3d)
        assert s_prep.shape == s_3d.shape
        assert adj_prep.shape == adj_3d.shape

        # Test with mismatched batch sizes
        s_mismatch = torch.randn(2, 4, 2)
        adj_mismatch = torch.randn(3, 4, 4)
        with pytest.raises(ValueError, match="batch sizes do not match"):
            DenseConnect._prepare_dense_inputs(s_mismatch, adj_mismatch)


class TestDenseConnectUnbatched:
    """Test the DenseConnect class for unbatched sparse pooling."""

    def test_dense_connect_unbatched_repr(self):
        """Test DenseConnect __repr__ method for unbatched mode."""
        connector = DenseConnect(
            remove_self_loops=False,
            degree_norm=True,
            edge_weight_norm=True,
            unbatched_output="batch",
        )
        repr_str = repr(connector)
        assert "DenseConnect" in repr_str
        assert "remove_self_loops=False" in repr_str
        assert "degree_norm=True" in repr_str
        assert "edge_weight_norm=True" in repr_str
        assert 'unbatched_output="batch"' in repr_str

    def test_dense_connect_unbatched_with_regular_tensor(
        self, pooler_test_graph_sparse
    ):
        """Test DenseConnect with regular tensor edge_index."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create sparse assignment matrix (COO format)
        k = num_nodes // 2
        eye_indices = torch.arange(k, dtype=torch.long)
        s = torch.sparse_coo_tensor(
            torch.stack([eye_indices, eye_indices]),
            torch.ones(k, dtype=torch.float),
            size=(num_nodes, k),
        ).coalesce()

        so = SelectOutput(s=s)

        connector = DenseConnect(
            remove_self_loops=False, degree_norm=False, unbatched_output="block"
        )
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index, edge_weight=edge_weight, so=so
        )

        # Verify output is regular tensor (edge_index format)
        assert isinstance(adj_pool, torch.Tensor)
        assert not adj_pool.is_sparse
        assert adj_pool.size(0) == 2
        assert edge_weight_pool is not None

    @pytest.mark.torch_sparse
    def test_dense_connect_unbatched_with_torch_coo_input(
        self, pooler_test_graph_sparse
    ):
        """Test DenseConnect with torch COO sparse tensor input."""
        pytest.importorskip("torch_sparse")

        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create sparse assignment matrix
        k = num_nodes // 2
        eye_indices = torch.arange(k, dtype=torch.long)
        s = torch.sparse_coo_tensor(
            torch.stack([eye_indices, eye_indices]),
            torch.ones(k, dtype=torch.float),
            size=(num_nodes, k),
        ).coalesce()

        so = SelectOutput(s=s)

        # Convert edge_index to torch COO sparse tensor
        edge_index_coo = torch.sparse_coo_tensor(
            edge_index, edge_weight, size=(num_nodes, num_nodes)
        ).coalesce()

        connector = DenseConnect(
            remove_self_loops=False, degree_norm=False, unbatched_output="block"
        )
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index_coo, edge_weight=None, so=so
        )

        # Verify output is torch COO sparse tensor
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.is_sparse
        assert edge_weight_pool is None  # Edge weights are embedded in sparse tensor

    def test_dense_connect_unbatched_degree_norm_empty_after_remove_self_loops(self):
        """Test DenseConnect with degree_norm=True and graph that becomes empty after remove_self_loops."""
        connector = DenseConnect(remove_self_loops=True, degree_norm=True)

        # Create assignment matrix
        num_nodes = 2
        k = 2
        eye_indices = torch.arange(k, dtype=torch.long)
        s = torch.sparse_coo_tensor(
            torch.stack([eye_indices, eye_indices]),
            torch.ones(k, dtype=torch.float),
            size=(num_nodes, k),
        ).coalesce()

        so = SelectOutput(s=s)

        # Create graph with only self-loops - after pooling and removing self-loops, becomes empty
        edge_index = torch.tensor([[0, 0], [0, 0]], dtype=torch.long)  # Only self-loops
        edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)

        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index, edge_weight=edge_weight, so=so
        )

        # After removing self-loops, graph should be empty or have no edges
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2

    def test_dense_connect_unbatched_edge_weight_norm_requires_batch_pooled(
        self, pooler_test_graph_sparse
    ):
        """Test that DenseConnect raises AssertionError when edge_weight_norm=True but batch_pooled=None."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create sparse assignment matrix
        k = num_nodes // 2
        eye_indices = torch.arange(k, dtype=torch.long)
        s = torch.sparse_coo_tensor(
            torch.stack([eye_indices, eye_indices]),
            torch.ones(k, dtype=torch.float),
            size=(num_nodes, k),
        ).coalesce()

        so = SelectOutput(s=s)

        connector = DenseConnect(edge_weight_norm=True)

        with pytest.raises(AssertionError, match="batch_pooled parameter is required"):
            connector(
                edge_index=edge_index,
                edge_weight=edge_weight,
                so=so,
                batch_pooled=None,
            )

    def test_dense_connect_unbatched_invalid_edge_index_type(self):
        """Test that DenseConnect raises ValueError for invalid edge_index type."""
        connector = DenseConnect(
            remove_self_loops=False, degree_norm=False, unbatched_output="block"
        )
        s = torch.eye(2).unsqueeze(0)
        so = SelectOutput(s=s)
        invalid_edge_index = [[0, 1], [1, 0]]

        with pytest.raises(ValueError, match="Edge index must be of type"):
            connector(edge_index=invalid_edge_index, edge_weight=None, so=so)

    def test_dense_connect_unbatched_unbatched_output_block(
        self, pooler_test_graph_sparse
    ):
        """Test DenseConnect with unbatched_output='block'."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create dense assignment matrix
        k = num_nodes // 2
        s = torch.zeros(num_nodes, k)
        for i in range(num_nodes):
            cluster = i // 2
            if cluster < k:
                s[i, cluster] = 1.0

        so = SelectOutput(s=s)

        connector = DenseConnect(unbatched_output="block", remove_self_loops=True)
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index, edge_weight=edge_weight, so=so, batch=None
        )

        # Should return sparse format (edge_index, edge_weight)
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.size(0) == 2  # edge_index format
        assert edge_weight_pool is not None

    def test_dense_connect_unbatched_unbatched_output_batch(
        self, pooler_test_graph_sparse
    ):
        """Test DenseConnect with unbatched_output='batch'."""
        x, edge_index, edge_weight, batch = pooler_test_graph_sparse
        num_nodes = x.size(0)

        # Create dense assignment matrix
        k = num_nodes // 2
        s = torch.zeros(num_nodes, k)
        for i in range(num_nodes):
            cluster = i // 2
            if cluster < k:
                s[i, cluster] = 1.0

        so = SelectOutput(s=s)

        connector = DenseConnect(unbatched_output="batch", remove_self_loops=True)
        adj_pool, edge_weight_pool = connector(
            edge_index=edge_index, edge_weight=edge_weight, so=so, batch=None
        )

        # Should return dense format [1, k, k]
        assert isinstance(adj_pool, torch.Tensor)
        assert adj_pool.dim() == 3
        assert adj_pool.size(0) == 1
        assert adj_pool.size(1) == k
        assert adj_pool.size(2) == k
        assert edge_weight_pool is None
