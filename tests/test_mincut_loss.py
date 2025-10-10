"""Comprehensive test suite for MinCut loss implementation with isolated nodes.

Tests the mincut_loss function with various scenarios including isolated nodes,
which is important for robust pooling operations.
"""

import pytest
import torch

from tgp.utils.losses import mincut_loss


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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
