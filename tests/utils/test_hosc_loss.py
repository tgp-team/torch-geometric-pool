"""Equivalence tests for HOSC pooler loss: hosc_orthogonality.

Dense (batched) vs unbatched variant, same style as test_mincut_loss.py.
"""

import pytest
import torch

from tgp.utils.losses import (
    hosc_orthogonality_loss,
    unbatched_hosc_orthogonality_loss,
)

from .dense_loss_test_helpers import _dense_batched_to_sparse_unbatched


class TestHOSCOrthogonalityLossDenseVsSparseEquality:
    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_single_graph(self, batch_reduction):
        torch.manual_seed(42)
        B, N, K = 1, 8, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = hosc_orthogonality_loss(S, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_hosc_orthogonality_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_batch(self, batch_reduction):
        torch.manual_seed(123)
        B, N, K = 3, 6, 4
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = hosc_orthogonality_loss(S, batch_reduction=batch_reduction)
        _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            torch.zeros(B, N, N, device=S.device), S
        )
        loss_sparse = unbatched_hosc_orthogonality_loss(
            S_flat, batch, batch_reduction=batch_reduction
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
