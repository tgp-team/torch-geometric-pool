"""Equivalence tests for DMoN pooler losses: spectral, cluster, orthogonality.

Dense (batched) vs sparse/unbatched variants, same style as test_mincut_loss.py.
"""

import pytest
import torch

from tgp.utils.losses import (
    cluster_loss,
    orthogonality_loss,
    sparse_spectral_loss,
    spectral_loss,
    unbatched_cluster_loss,
    unbatched_orthogonality_loss,
)

from .dense_loss_test_helpers import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch,
)


class TestSpectralLossDenseVsSparseEquality:
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
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
    def test_batch(self, batch_reduction):
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
    def test_with_isolated_nodes(self, batch_reduction):
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
        # Graph 0: triangle; graph 1: no edges (empty)
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
        assert torch.isfinite(loss_dense), (
            "batched spectral_loss must be finite for empty graph"
        )
        assert torch.isfinite(loss_sparse), (
            "sparse spectral_loss must be finite for empty graph"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_with_weighted_edges(self):
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
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
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
    def test_batch(self, batch_reduction):
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


class TestOrthogonalityLossDenseVsSparseEquality:
    """Orthogonality loss used by DMoN (and MinCut)."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
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
    def test_batch(self, batch_reduction):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
