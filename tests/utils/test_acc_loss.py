"""Equivalence tests for AsymCheegerCut (ACC) pooler losses: totvar, asym_norm.

Dense (batched) vs sparse/unbatched variants, same style as test_mincut_loss.py.
"""

import pytest
import torch

from tgp.utils.losses import (
    asym_norm_loss,
    sparse_totvar_loss,
    totvar_loss,
    unbatched_asym_norm_loss,
)

from .dense_loss_test_helpers import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch,
)


class TestTotvarLossDenseVsSparseEquality:
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
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
    def test_batch(self, batch_reduction):
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
    def test_with_isolated_nodes(self, batch_reduction):
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

    def test_with_weighted_edges(self):
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
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
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
    def test_batch(self, batch_reduction):
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


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
