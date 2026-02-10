"""Tests for AsymCheegerCut (ACC) pooler losses: totvar, asym_norm.

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
    asym_norm_loss,
    sparse_totvar_loss,
    totvar_loss,
    unbatched_asym_norm_loss,
)


class TestTotvarLoss:
    """Test total variation loss (batched) in isolation."""

    def test_totvar_loss_basic(self):
        torch.manual_seed(42)
        adj, S = _make_dense_batch(2, 4, 3, seed=42)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_totvar_loss_with_isolated_nodes(self):
        torch.manual_seed(123)
        B, N, K = 1, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_totvar_loss_batch_reduction(self):
        torch.manual_seed(789)
        adj, S = _make_dense_batch(3, 4, 2, seed=789)
        loss_mean = totvar_loss(S, adj, batch_reduction="mean")
        loss_sum = totvar_loss(S, adj, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_totvar_loss_all_isolated_nodes(self):
        """Adj all zeros (one or more graphs)."""
        torch.manual_seed(456)
        B, N, K = 1, 4, 2
        adj = torch.zeros(B, N, N)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_totvar_loss_gradient_flow(self):
        """Test that gradients flow through totvar_loss."""
        torch.manual_seed(42)
        adj, _ = _make_dense_batch(2, 4, 3, seed=42)
        S_raw = torch.randn(2, 4, 3, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_totvar_loss_edge_cases(self):
        """Single node (1,1,1) and two isolated nodes (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        adj = torch.zeros(B, N, N)
        S = torch.ones(B, N, K)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        adj = torch.zeros(B, N, N)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = totvar_loss(S, adj, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestAsymNormLoss:
    """Test asymmetric norm loss (batched) in isolation."""

    def test_asym_norm_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = asym_norm_loss(S, K, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_asym_norm_loss_batch_reduction(self):
        torch.manual_seed(123)
        B, N, K = 3, 5, 2
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_mean = asym_norm_loss(S, K, batch_reduction="mean")
        loss_sum = asym_norm_loss(S, K, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_asym_norm_loss_gradient_flow(self):
        """Test that gradients flow through asym_norm_loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = asym_norm_loss(S, K, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_asym_norm_loss_edge_cases(self):
        """Single node (1,1,1) and two nodes (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = asym_norm_loss(S, K, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = asym_norm_loss(S, K, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestTotvarLossDenseVsSparseEquality:
    """Dense (batched) vs sparse totvar loss equality."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_totvar_loss_dense_vs_sparse_equality_single_graph(self, batch_reduction):
        """Single graph: dense [1,N,N] + [1,N,K] vs sparse edge_index + [N,K]."""
        adj, S = _make_dense_batch(1, 8, 4, seed=42)
        loss_dense = totvar_loss(S, adj, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_totvar_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_totvar_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,N] + [B,N,K] vs sparse + batch."""
        adj, S = _make_dense_batch(3, 6, 4, seed=123)
        loss_dense = totvar_loss(S, adj, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_totvar_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_totvar_loss_dense_vs_sparse_with_isolated_nodes(self, batch_reduction):
        """Dense vs sparse totvar with isolated nodes."""
        torch.manual_seed(321)
        B, N, K = 2, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = totvar_loss(S, adj, batch_reduction=batch_reduction)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_totvar_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_totvar_loss_dense_vs_sparse_with_weighted_edges(self):
        """Dense vs sparse totvar with non-unit edge weights."""
        torch.manual_seed(456)
        B, N, K = 2, 5, 3
        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = totvar_loss(S, adj, batch_reduction="mean")
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_totvar_loss(
            edge_index, S_flat, edge_weight, batch, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)


class TestAsymNormLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched asym_norm loss equality."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_asym_norm_loss_dense_vs_sparse_equality_single_graph(
        self, batch_reduction
    ):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
        torch.manual_seed(42)
        B, N, K = 1, 8, 4
        k = K
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = asym_norm_loss(S, k, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_asym_norm_loss(
            S_flat, k, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_asym_norm_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] + batch."""
        torch.manual_seed(123)
        B, N, K = 3, 6, 4
        k = K
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = asym_norm_loss(S, k, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_asym_norm_loss(
            S_flat, k, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_asym_norm_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        K = 3
        loss_dense = asym_norm_loss(S, K, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_asym_norm_loss(S_flat, K, batch, batch_reduction="mean")
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched asym_norm loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
