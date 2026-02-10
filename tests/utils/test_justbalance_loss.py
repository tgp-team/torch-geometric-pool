"""Tests for JustBalance pooler loss: just_balance.

Loss behavior (batched) and dense vs unbatched equality.
"""

import pytest
import torch

from tests.test_utils import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch_variable_sizes,
)
from tgp.utils.losses import (
    just_balance_loss,
    unbatched_just_balance_loss,
)


class TestJustBalanceLoss:
    """Test JustBalance loss (batched) in isolation."""

    def test_just_balance_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_just_balance_loss_batch_reduction(self):
        torch.manual_seed(123)
        B, N, K = 3, 5, 2
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_mean = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        loss_sum = just_balance_loss(S, normalize_loss=True, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_just_balance_loss_gradient_flow(self):
        """Test that gradients flow through just_balance_loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_just_balance_loss_edge_cases(self):
        """Single node (1,1,1) and two nodes (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestJustBalanceLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched JustBalance loss equality."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_just_balance_loss_dense_vs_sparse_equality_single_graph(
        self, batch_reduction
    ):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
        torch.manual_seed(42)
        B, N, K = 1, 8, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = just_balance_loss(
            S, normalize_loss=True, batch_reduction=batch_reduction
        )
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_just_balance_loss(
            S_flat, batch, normalize_loss=True, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_just_balance_loss_dense_vs_sparse_equality_batch(self, batch_reduction):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] + batch."""
        torch.manual_seed(123)
        B, N, K = 3, 6, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = just_balance_loss(
            S, normalize_loss=True, batch_reduction=batch_reduction
        )
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_just_balance_loss(
            S_flat, batch, normalize_loss=True, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_just_balance_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        loss_dense = just_balance_loss(S, normalize_loss=True, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_just_balance_loss(
            S_flat, batch, normalize_loss=True, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched just_balance loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
