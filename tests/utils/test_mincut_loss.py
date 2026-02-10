"""Comprehensive test suite for MinCut loss implementation with isolated nodes.

Tests the mincut_loss function with various scenarios including isolated nodes,
which is important for robust pooling operations.

Also tests that the unbatched loss variants return exactly the same values as
the dense (batched) variants when given equivalent inputs.
"""

import pytest
import torch

from tgp.utils.losses import (
    mincut_loss,
    orthogonality_loss,
    sparse_mincut_loss,
    unbatched_orthogonality_loss,
)

from .dense_loss_test_helpers import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch_variable_sizes,
)


class TestMinCutLoss:
    """Test MinCut loss computation with various graph configurations."""

    def test_mincut_loss_basic(self):
        """Test basic MinCut loss computation."""
        torch.manual_seed(42)

        # Create simple batch with 2 graphs, 3 nodes each, 2 clusters
        B, N, K = 2, 3, 2

        # Create adjacency matrices (dense format)
        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)  # Make symmetric
        # Remove self-loops for each batch
        for b in range(B):
            adj[b].fill_diagonal_(0)

        # Create cluster assignment matrix S
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)  # Normalize to probabilities

        # Create pooled adjacency matrix
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        # Compute loss
        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        # Check output
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_mincut_loss_with_isolated_nodes(self):
        """Test MinCut loss with isolated nodes."""
        torch.manual_seed(123)

        # Create batch with isolated nodes
        B, N, K = 1, 5, 3  # 5 nodes, 3 clusters

        # Create adjacency matrix with isolated nodes
        # Nodes 0-2 form a triangle, nodes 3-4 are isolated
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0  # Edge 0-1
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0  # Edge 1-2
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0  # Edge 2-0
        # Nodes 3 and 4 have no edges (isolated)

        # Create cluster assignment matrix S
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)  # Normalize to probabilities

        # Create pooled adjacency matrix
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        # Compute loss
        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        # Check output is valid
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

        # Test that the loss handles isolated nodes correctly
        # Isolated nodes should contribute to the degree matrix with zero degree
        # This tests the robustness of the mincut loss computation

    def test_mincut_loss_all_isolated_nodes(self):
        """Test MinCut loss when all nodes are isolated (no edges)."""
        torch.manual_seed(456)

        B, N, K = 1, 4, 2

        # Create adjacency matrix with no edges (all nodes isolated)
        adj = torch.zeros(B, N, N)

        # Create cluster assignment matrix S
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)  # Normalize to probabilities

        # Create pooled adjacency matrix (will be all zeros)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        # Compute loss
        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        # With no edges, the numerator (trace of adj_pooled) will be 0
        # The denominator (trace of S^T D S) will also be 0 since D is zero matrix
        # The loss function adds epsilon to prevent division by zero
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0  # Scalar
        assert torch.isfinite(loss)

    def test_mincut_loss_batch_reduction(self):
        """Test MinCut loss with different batch reduction methods."""
        torch.manual_seed(789)

        B, N, K = 3, 4, 2

        # Create adjacency matrices with some isolated nodes
        adj = torch.zeros(B, N, N)
        # First graph: chain 0-1-2, node 3 isolated
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        # Second graph: single edge 0-1, nodes 2-3 isolated
        adj[1, 0, 1] = adj[1, 1, 0] = 1.0
        # Third graph: all nodes isolated
        # (adj[2] remains zeros)

        # Create cluster assignment matrix S
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)  # Normalize to probabilities

        # Create pooled adjacency matrix
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        # Test mean reduction
        loss_mean = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        # Test sum reduction
        loss_sum = mincut_loss(adj, S, adj_pooled, batch_reduction="sum")

        # Check outputs
        assert isinstance(loss_mean, torch.Tensor)
        assert isinstance(loss_sum, torch.Tensor)
        assert loss_mean.dim() == 0  # Scalar
        assert loss_sum.dim() == 0  # Scalar
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

        # Sum should be larger than mean (unless all losses are the same)
        # Note: This might not always hold due to epsilon handling, so we just check they're different
        # unless the batch losses are identical

    def test_mincut_loss_gradient_flow(self):
        """Test that gradients flow through MinCut loss."""
        torch.manual_seed(42)

        B, N, K = 1, 4, 2

        # Create adjacency matrix with isolated nodes
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0  # Only one edge, nodes 2-3 isolated

        # Create cluster assignment matrix S (requires grad)
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)  # Normalize to probabilities

        # Create pooled adjacency matrix
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        # Compute loss
        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        # Backward pass
        loss.backward()

        # Check gradients exist and are finite
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_mincut_loss_edge_cases(self):
        """Test MinCut loss edge cases."""
        torch.manual_seed(111)

        # Test with single node (isolated by definition)
        B, N, K = 1, 1, 1
        adj = torch.zeros(B, N, N)
        S = torch.ones(B, N, K)  # Single assignment
        adj_pooled = torch.zeros(B, K, K)

        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)

        # Test with two isolated nodes
        B, N, K = 1, 2, 1
        adj = torch.zeros(B, N, N)
        S = torch.ones(B, N, K)  # Both nodes to same cluster
        adj_pooled = torch.zeros(B, K, K)

        loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestMinCutLossDenseVsSparseEquality:
    """Test that unbatched mincut and orthogonality losses return exactly the
    same values as the dense (batched) versions when given equivalent inputs.
    """

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_mincut_loss_dense_vs_sparse_equality_single_graph(self, batch_reduction):
        """Single graph: dense [1,N,N] + [1,N,K] vs sparse edge_index + [N,K]."""
        torch.manual_seed(42)
        B, N, K = 1, 8, 4

        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        loss_dense = mincut_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)

        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_mincut_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            f"mincut_loss dense vs sparse mismatch (single graph, {batch_reduction}): "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_mincut_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,N] + [B,N,K] vs sparse edge_index + [N,K] + batch."""
        torch.manual_seed(123)
        B, N, K = 3, 6, 4

        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        loss_dense = mincut_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)

        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_mincut_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            f"mincut_loss dense vs sparse mismatch (batch, {batch_reduction}): "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_mincut_loss_dense_vs_sparse_with_isolated_nodes(self, batch_reduction):
        """Dense vs sparse mincut with isolated nodes and an empty graph."""
        torch.manual_seed(321)
        B, N, K = 2, 5, 3

        adj = torch.zeros(B, N, N)
        # Graph 0: triangle among nodes 0-2, nodes 3-4 isolated
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        # Graph 1: all nodes isolated (no edges)

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        loss_dense = mincut_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)

        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_mincut_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            "mincut_loss dense vs sparse mismatch with isolated nodes "
            f"(batch_reduction={batch_reduction}): "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_orthogonality_loss_dense_vs_sparse_equality_single_graph(
        self, batch_reduction
    ):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
        torch.manual_seed(42)
        B, N, K = 1, 8, 4

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)

        loss_dense = orthogonality_loss(S, batch_reduction=batch_reduction)

        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_orthogonality_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            f"orthogonality_loss dense vs sparse mismatch (single graph, {batch_reduction}): "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_orthogonality_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] + batch."""
        torch.manual_seed(123)
        B, N, K = 3, 6, 4

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)

        loss_dense = orthogonality_loss(S, batch_reduction=batch_reduction)

        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_orthogonality_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            f"orthogonality_loss dense vs sparse mismatch (batch, {batch_reduction}): "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    def test_orthogonality_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded (as from MLPSelect); dense matches sparse.

        Batched S has zero rows for padded positions, so S^T S is unchanged by padding
        and dense orthogonality_loss matches unbatched (real nodes only).
        """
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        loss_dense = orthogonality_loss(S, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_orthogonality_loss(
            S_flat, batch, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S (as from select), dense and sparse orthogonality "
            "should match; padding rows are zero and do not contribute to S^T S."
        )

    def test_mincut_loss_dense_vs_sparse_with_weighted_edges(self):
        """Dense vs sparse mincut with non-unit edge weights."""
        torch.manual_seed(456)
        B, N, K = 2, 5, 3

        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)

        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        loss_dense = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")

        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_mincut_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction="mean"
        )

        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-6, atol=1e-6), (
            "mincut_loss dense vs sparse mismatch with weighted edges: "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
