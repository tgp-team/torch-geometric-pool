"""Tests for DMoN pooler losses: spectral, cluster, orthogonality.

Loss behavior (batched) and dense vs sparse/unbatched equality.
"""

import pytest
import torch

from tests.test_utils import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch,
    _make_dense_batch_variable_sizes,
)
from tgp.utils.losses import (
    cluster_loss,
    orthogonality_loss,
    sparse_spectral_loss,
    spectral_loss,
    unbatched_cluster_loss,
    unbatched_orthogonality_loss,
)


class TestSpectralLoss:
    """Test spectral loss computation (batched) in isolation."""

    def test_spectral_loss_basic(self):
        torch.manual_seed(42)
        adj, S = _make_dense_batch(2, 4, 3, seed=42)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_spectral_loss_with_isolated_nodes(self):
        torch.manual_seed(123)
        B, N, K = 1, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_spectral_loss_empty_graph(self):
        torch.manual_seed(456)
        B, N, K = 1, 4, 2
        adj = torch.zeros(B, N, N)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_spectral_loss_batch_reduction(self):
        torch.manual_seed(789)
        adj, S = _make_dense_batch(3, 4, 2, seed=789)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_mean = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        loss_sum = spectral_loss(adj, S, adj_pooled, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_spectral_loss_gradient_flow(self):
        """Test that gradients flow through spectral loss."""
        torch.manual_seed(42)
        B, N, K = 1, 4, 2
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_spectral_loss_edge_cases(self):
        """Test spectral loss with single node and two nodes."""
        torch.manual_seed(111)
        # Single node (isolated by definition)
        B, N, K = 1, 1, 1
        adj = torch.zeros(B, N, N)
        S = torch.ones(B, N, K)
        adj_pooled = torch.zeros(B, K, K)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)
        # Two nodes, one edge
        B, N, K = 1, 2, 1
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestClusterLoss:
    """Test cluster loss computation (batched) in isolation."""

    def test_cluster_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = cluster_loss(S, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_cluster_loss_batch_reduction(self):
        torch.manual_seed(123)
        B, N, K = 3, 5, 2
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_mean = cluster_loss(S, batch_reduction="mean")
        loss_sum = cluster_loss(S, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_cluster_loss_edge_cases(self):
        """Test cluster loss with single node and two nodes."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = cluster_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = cluster_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_cluster_loss_gradient_flow(self):
        """Test that gradients flow through cluster loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = cluster_loss(S, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()


class TestOrthogonalityLoss:
    """Test orthogonality loss (batched) in isolation. Used by DMoN."""

    def test_orthogonality_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = orthogonality_loss(S, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_orthogonality_loss_batch_reduction(self):
        torch.manual_seed(123)
        B, N, K = 3, 5, 2
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_mean = orthogonality_loss(S, batch_reduction="mean")
        loss_sum = orthogonality_loss(S, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_orthogonality_loss_edge_cases(self):
        """Test orthogonality loss with single node and two nodes."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = orthogonality_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = orthogonality_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_orthogonality_loss_gradient_flow(self):
        """Test that gradients flow through orthogonality loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = orthogonality_loss(S, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()


class TestSpectralLossDenseVsSparseEquality:
    """Dense (batched) vs sparse spectral loss return same values for equivalent inputs."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_spectral_loss_dense_vs_sparse_equality_single_graph(self, batch_reduction):
        """Single graph: dense [1,N,N] + [1,N,K] vs sparse edge_index + [N,K]."""
        adj, S = _make_dense_batch(1, 8, 4, seed=42)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_dense = spectral_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_spectral_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_spectral_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,N] + [B,N,K] vs sparse + batch."""
        adj, S = _make_dense_batch(3, 6, 4, seed=123)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_dense = spectral_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_spectral_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_spectral_loss_dense_vs_sparse_with_isolated_nodes(self, batch_reduction):
        """Dense vs sparse spectral with isolated nodes."""
        torch.manual_seed(321)
        B, N, K = 2, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_dense = spectral_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_spectral_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_spectral_loss_empty_graph_no_nan(self, batch_reduction):
        """Both batched and sparse spectral loss return finite (0) for graphs with no edges."""
        torch.manual_seed(321)
        B, N, K = 2, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_dense = spectral_loss(adj, S, adj_pooled, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_spectral_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.isfinite(loss_dense)
        assert torch.isfinite(loss_sparse)
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_spectral_loss_dense_vs_sparse_with_weighted_edges(self):
        """Dense vs sparse spectral with non-unit edge weights."""
        torch.manual_seed(456)
        B, N, K = 2, 5, 3
        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        loss_dense = spectral_loss(adj, S, adj_pooled, batch_reduction="mean")
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_spectral_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)


class TestClusterLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched cluster loss equality."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_cluster_loss_dense_vs_sparse_equality_single_graph(self, batch_reduction):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
        torch.manual_seed(42)
        B, N, K = 1, 8, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = cluster_loss(S, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_cluster_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_cluster_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] + batch."""
        torch.manual_seed(123)
        B, N, K = 3, 6, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = cluster_loss(S, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_cluster_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_cluster_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        loss_dense = cluster_loss(S, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_cluster_loss(S_flat, batch, batch_reduction="mean")
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched cluster loss should match."
        )


class TestOrthogonalityLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched orthogonality loss equality. Used by DMoN."""

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
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

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
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_orthogonality_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        loss_dense = orthogonality_loss(S, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_orthogonality_loss(
            S_flat, batch, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched orthogonality loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
