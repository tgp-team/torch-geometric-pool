"""Equivalence tests for DiffPool pooler losses: link_pred, entropy.

Dense (batched) vs sparse/unbatched variants, same style as test_mincut_loss.py.
"""

import pytest
import torch

from tgp.utils.losses import (
    entropy_loss,
    link_pred_loss,
    sparse_link_pred_loss,
    unbatched_entropy_loss,
)

from .dense_loss_test_helpers import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch,
)


class TestLinkPredLossDenseVsSparseEquality:
    def test_single_graph(self):
        adj, S = _make_dense_batch(1, 8, 4, seed=42)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_batch(self):
        adj, S = _make_dense_batch(3, 6, 4, seed=123)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)


class TestEntropyLossDenseVsSparseEquality:
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
        torch.manual_seed(42)
        B, N, K = 1, 8, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = entropy_loss(S, num_nodes=B * N)
        _, _, S_flat, _ = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_entropy_loss(S_flat, num_nodes=B * N)
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_batch(self, batch_reduction):
        torch.manual_seed(123)
        B, N, K = 3, 6, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = entropy_loss(S, num_nodes=B * N)
        _, _, S_flat, _ = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_entropy_loss(S_flat, num_nodes=B * N)
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
