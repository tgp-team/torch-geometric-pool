"""Consolidated tests for tgp.utils.losses."""

# --- From test_acc_loss.py ---
# Tests for AsymCheegerCut (ACC) pooler losses: totvar, asym_norm.
# Loss behavior (batched) and dense vs sparse/unbatched equality.

import pytest
import torch

from tests.test_utils import (
    _dense_batched_to_sparse_unbatched,
    _make_dense_batch,
    _make_dense_batch_variable_sizes,
)
from tgp.utils.losses import (
    asym_norm_loss,
    cluster_loss,
    entropy_loss,
    hosc_orthogonality_loss,
    just_balance_loss,
    link_pred_loss,
    mincut_loss,
    orthogonality_loss,
    sparse_link_pred_loss,
    sparse_mincut_loss,
    sparse_spectral_loss,
    sparse_totvar_loss,
    spectral_loss,
    totvar_loss,
    unbatched_asym_norm_loss,
    unbatched_cluster_loss,
    unbatched_entropy_loss,
    unbatched_hosc_orthogonality_loss,
    unbatched_just_balance_loss,
    unbatched_orthogonality_loss,
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
        loss_dense = asym_norm_loss(S, K, mask=mask, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_asym_norm_loss(S_flat, K, batch, batch_reduction="mean")
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched asym_norm loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# --- From test_diffpool_loss.py ---
"""Tests for DiffPool pooler losses: link_pred, entropy.

Loss behavior (batched) and dense vs sparse/unbatched equality.
"""


class TestLinkPredLoss:
    """Test link prediction loss (batched) in isolation."""

    def test_link_pred_loss_basic(self):
        torch.manual_seed(42)
        adj, S = _make_dense_batch(2, 4, 3, seed=42)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_link_pred_loss_with_normalize(self):
        torch.manual_seed(42)
        adj, S = _make_dense_batch(2, 4, 3, seed=42)
        loss = link_pred_loss(S, adj, normalize_loss=True)
        assert torch.isfinite(loss)

    def test_link_pred_loss_with_isolated_nodes(self):
        """One graph: triangle + isolated nodes."""
        torch.manual_seed(123)
        B, N, K = 1, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        assert torch.isfinite(loss)

    def test_link_pred_loss_all_isolated(self):
        """Adj all zeros (empty graph)."""
        torch.manual_seed(456)
        B, N, K = 1, 4, 2
        adj = torch.zeros(B, N, N)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        assert torch.isfinite(loss)

    def test_link_pred_loss_gradient_flow(self):
        """Test that gradients flow through link_pred_loss."""
        torch.manual_seed(42)
        adj, _ = _make_dense_batch(2, 4, 3, seed=42)
        S_raw = torch.randn(2, 4, 3, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_link_pred_loss_edge_cases(self):
        """Single node (1,1,1) and two nodes with one edge (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        adj = torch.zeros(B, N, N)
        S = torch.ones(B, N, K)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = link_pred_loss(S, adj, normalize_loss=False)
        assert torch.isfinite(loss)


class TestEntropyLoss:
    """Test entropy loss (batched) in isolation."""

    def test_entropy_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = entropy_loss(S, num_nodes=B * N)
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_entropy_loss_gradient_flow(self):
        """Test that gradients flow through entropy_loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = entropy_loss(S, num_nodes=B * N)
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_entropy_loss_edge_cases(self):
        """Single node (1,1,1) and two nodes (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = entropy_loss(S, num_nodes=B * N)
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = entropy_loss(S, num_nodes=B * N)
        assert torch.isfinite(loss)


class TestLinkPredLossDenseVsSparseEquality:
    """Dense (batched) vs sparse link prediction loss equality."""

    def test_link_pred_loss_dense_vs_sparse_equality_single_graph(self):
        """Single graph: dense [1,N,N] + [1,N,K] vs sparse edge_index + [N,K]."""
        adj, S = _make_dense_batch(1, 8, 4, seed=42)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_link_pred_loss_dense_vs_sparse_equality_batch(self):
        """Multiple graphs: dense [B,N,N] + [B,N,K] vs sparse + batch."""
        adj, S = _make_dense_batch(3, 6, 4, seed=123)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5)

    def test_link_pred_loss_dense_vs_sparse_with_isolated_nodes(self):
        """Dense vs sparse link_pred with isolated nodes and an empty graph."""
        torch.manual_seed(321)
        B, N, K = 2, 5, 3
        adj = torch.zeros(B, N, N)
        adj[0, 0, 1] = adj[0, 1, 0] = 1.0
        adj[0, 1, 2] = adj[0, 2, 1] = 1.0
        adj[0, 2, 0] = adj[0, 0, 2] = 1.0
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "link_pred_loss dense vs sparse mismatch with isolated nodes: "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )

    def test_link_pred_loss_dense_vs_sparse_with_weighted_edges(self):
        """Dense vs sparse link_pred with non-unit edge weights."""
        torch.manual_seed(456)
        B, N, K = 2, 5, 3
        adj = torch.rand(B, N, N)
        adj = adj + adj.transpose(-2, -1)
        for b in range(B):
            adj[b].fill_diagonal_(0)
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_dense = link_pred_loss(S, adj, normalize_loss=False)
        edge_index, edge_weight, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S
        )
        loss_sparse = sparse_link_pred_loss(
            S_flat, edge_index, edge_weight, batch, normalize_loss=False
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "link_pred_loss dense vs sparse mismatch with weighted edges: "
            f"dense={loss_dense.item()}, sparse={loss_sparse.item()}"
        )


class TestEntropyLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched entropy loss equality."""

    def test_entropy_loss_dense_vs_sparse_equality_single_graph(self):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
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

    def test_entropy_loss_dense_vs_sparse_equality_batch(self):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] (flat)."""
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

    def test_entropy_loss_dense_vs_sparse_with_variable_sizes_zero_padding(self):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        num_nodes = 3 + 5 + 4
        loss_dense = entropy_loss(S, num_nodes=num_nodes)
        edge_index, _, S_flat, _ = _dense_batched_to_sparse_unbatched(adj, S, mask=mask)
        loss_sparse = unbatched_entropy_loss(S_flat, num_nodes=num_nodes)
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched entropy loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# --- From test_dmon_loss.py ---
"""Tests for DMoN pooler losses: spectral, cluster, orthogonality.

Loss behavior (batched) and dense vs sparse/unbatched equality.
"""


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
        loss_dense = cluster_loss(S, mask=mask, batch_reduction="mean")
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

# --- From test_hosc_loss.py ---
"""Tests for HOSC pooler loss: hosc_orthogonality.

Loss behavior (batched) and dense vs unbatched equality.
"""


class TestHOSCOrthogonalityLoss:
    """Test HOSC orthogonality loss (batched) in isolation."""

    def test_hosc_orthogonality_loss_basic(self):
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = hosc_orthogonality_loss(S, batch_reduction="mean")
        assert isinstance(loss, torch.Tensor)
        assert loss.dim() == 0
        assert torch.isfinite(loss)

    def test_hosc_orthogonality_loss_batch_reduction(self):
        torch.manual_seed(123)
        B, N, K = 3, 5, 2
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss_mean = hosc_orthogonality_loss(S, batch_reduction="mean")
        loss_sum = hosc_orthogonality_loss(S, batch_reduction="sum")
        assert torch.isfinite(loss_mean)
        assert torch.isfinite(loss_sum)

    def test_hosc_orthogonality_loss_gradient_flow(self):
        """Test that gradients flow through hosc_orthogonality_loss."""
        torch.manual_seed(42)
        B, N, K = 2, 4, 3
        S_raw = torch.randn(B, N, K, requires_grad=True)
        S = torch.softmax(S_raw, dim=-1)
        loss = hosc_orthogonality_loss(S, batch_reduction="mean")
        loss.backward()
        assert S_raw.grad is not None
        assert torch.isfinite(S_raw.grad).all()

    def test_hosc_orthogonality_loss_edge_cases(self):
        """Single node (1,1,1) and two nodes (1,2,1)."""
        torch.manual_seed(111)
        B, N, K = 1, 1, 1
        S = torch.ones(B, N, K)
        loss = hosc_orthogonality_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)
        B, N, K = 1, 2, 1
        S = torch.randn(B, N, K)
        S = torch.softmax(S, dim=-1)
        loss = hosc_orthogonality_loss(S, batch_reduction="mean")
        assert torch.isfinite(loss)


class TestHOSCOrthogonalityLossDenseVsSparseEquality:
    """Dense (batched) vs unbatched HOSC orthogonality loss equality."""

    @pytest.mark.parametrize("batch_reduction", ["mean", "sum"])
    def test_hosc_orthogonality_loss_dense_vs_sparse_equality_single_graph(
        self, batch_reduction
    ):
        """Single graph: dense [1,N,K] vs unbatched [N,K]."""
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
    def test_hosc_orthogonality_loss_dense_vs_sparse_equality_batch(
        self, batch_reduction
    ):
        """Multiple graphs: dense [B,N,K] vs unbatched [N,K] + batch."""
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

    def test_hosc_orthogonality_loss_dense_vs_sparse_with_variable_sizes_zero_padding(
        self,
    ):
        """With variable-sized graphs, S is zero-padded; dense matches unbatched."""
        adj, S, mask = _make_dense_batch_variable_sizes(K=3, seed=42)
        loss_dense = hosc_orthogonality_loss(S, mask=mask, batch_reduction="mean")
        edge_index, _, S_flat, batch = _dense_batched_to_sparse_unbatched(
            adj, S, mask=mask
        )
        loss_sparse = unbatched_hosc_orthogonality_loss(
            S_flat, batch, batch_reduction="mean"
        )
        assert torch.allclose(loss_dense, loss_sparse, rtol=1e-5, atol=1e-5), (
            "With zero-padded S, dense and unbatched HOSC orthogonality loss should match."
        )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])

# --- From test_justbalance_loss.py ---
"""Tests for JustBalance pooler loss: just_balance.

Loss behavior (batched) and dense vs unbatched equality.
"""


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
        loss_dense = just_balance_loss(
            S, mask=mask, normalize_loss=True, batch_reduction="mean"
        )
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

# --- From test_mincut_loss.py ---
"""Comprehensive test suite for MinCut loss implementation with isolated nodes.

Tests the mincut_loss function with various scenarios including isolated nodes,
which is important for robust pooling operations.

Also tests that the unbatched loss variants return exactly the same values as
the dense (batched) variants when given equivalent inputs.
"""


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


class TestLossesBranchCoverage:
    """Targeted branch tests for losses.py uncovered edges."""

    def test_batch_reduce_loss_rejects_invalid_reduction(self):
        import tgp.utils.losses as losses

        with pytest.raises(ValueError, match="Batch reduction .* not allowed"):
            losses._batch_reduce_loss(torch.tensor([1.0]), "invalid")  # type: ignore[arg-type]

    def test_sparse_mincut_loss_handles_filtered_none_edge_weight(self, monkeypatch):
        import tgp.utils.losses as losses

        monkeypatch.setattr(losses, "check_and_filter_edge_weights", lambda _: None)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.ones((2, 1), dtype=torch.float32)
        S = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = losses.sparse_mincut_loss(
            edge_index, S, edge_weight=edge_weight, batch=batch
        )
        assert torch.isfinite(loss)

    def test_sparse_link_pred_loss_handles_filtered_none_and_normalization(
        self, monkeypatch
    ):
        import tgp.utils.losses as losses

        monkeypatch.setattr(losses, "check_and_filter_edge_weights", lambda _: None)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.ones((2, 1), dtype=torch.float32)
        S = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = losses.sparse_link_pred_loss(
            S,
            edge_index,
            edge_weight=edge_weight,
            batch=batch,
            normalize_loss=True,
        )
        assert torch.isfinite(loss)

    def test_sparse_totvar_loss_handles_filtered_none_edge_weight(self, monkeypatch):
        import tgp.utils.losses as losses

        monkeypatch.setattr(losses, "check_and_filter_edge_weights", lambda _: None)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.ones((2, 1), dtype=torch.float32)
        S = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = losses.sparse_totvar_loss(
            edge_index, S, edge_weight=edge_weight, batch=batch
        )
        assert torch.isfinite(loss)

    def test_sparse_spectral_loss_handles_filtered_none_edge_weight(self, monkeypatch):
        import tgp.utils.losses as losses

        monkeypatch.setattr(losses, "check_and_filter_edge_weights", lambda _: None)
        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        edge_weight = torch.ones((2, 1), dtype=torch.float32)
        S = torch.tensor([[0.6, 0.4], [0.1, 0.9]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)

        loss = losses.sparse_spectral_loss(
            edge_index, S, edge_weight=edge_weight, batch=batch
        )
        assert torch.isfinite(loss)

    def test_unbatched_hosc_orthogonality_loss_k_le_one_and_batch_none(self):
        import tgp.utils.losses as losses

        # sqrt_k <= 1 early return.
        S_single_cluster = torch.ones((3, 1), dtype=torch.float32)
        zero_loss = losses.unbatched_hosc_orthogonality_loss(S_single_cluster)
        assert torch.isclose(zero_loss, torch.tensor(0.0))

        # batch=None fallback path for regular case.
        S = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)
        loss = losses.unbatched_hosc_orthogonality_loss(S, batch=None)
        assert torch.isfinite(loss)

    def test_unbatched_asym_norm_loss_k_le_one_and_idx_clip(self, monkeypatch):
        import tgp.utils.losses as losses

        # k <= 1 early return.
        S_single_cluster = torch.ones((4, 1), dtype=torch.float32)
        zero_loss = losses.unbatched_asym_norm_loss(S_single_cluster, k=1, batch=None)
        assert torch.isclose(zero_loss, torch.tensor(0.0))

        # Force idx >= n_nodes_g to exercise clip branch.
        monkeypatch.setattr(losses.math, "floor", lambda _: 10)
        S = torch.tensor([[0.6, 0.4], [0.2, 0.8]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)
        clipped_loss = losses.unbatched_asym_norm_loss(S, k=2, batch=batch)
        assert torch.isfinite(clipped_loss)

    def test_unbatched_just_balance_loss_without_normalization(self):
        import tgp.utils.losses as losses

        S = torch.tensor([[0.7, 0.3], [0.4, 0.6]], dtype=torch.float32)
        batch = torch.tensor([0, 0], dtype=torch.long)
        loss = losses.unbatched_just_balance_loss(
            S, batch=batch, normalize_loss=False, batch_reduction="mean"
        )
        assert torch.isfinite(loss)

    def test_asym_norm_loss_zero_node_branch(self):
        import tgp.utils.losses as losses

        S = torch.empty((2, 0, 3), dtype=torch.float32)
        loss = losses.asym_norm_loss(S, k=3, batch_reduction="mean")
        assert torch.isclose(loss, torch.tensor(0.0))

    def test_just_balance_loss_without_normalization(self):
        import tgp.utils.losses as losses

        S = torch.tensor(
            [[[0.7, 0.3], [0.4, 0.6]], [[0.2, 0.8], [0.9, 0.1]]], dtype=torch.float32
        )
        loss = losses.just_balance_loss(S, normalize_loss=False, batch_reduction="mean")
        assert torch.isfinite(loss)

    def test_spectral_and_cluster_loss_with_explicit_num_supernodes(self):
        import tgp.utils.losses as losses

        adj, S = _make_dense_batch(2, 4, 3, seed=1234)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)

        spectral = losses.spectral_loss(
            adj,
            S,
            adj_pooled,
            num_supernodes=S.size(-1),
            batch_reduction="mean",
        )
        cluster = losses.cluster_loss(
            S,
            num_supernodes=S.size(-1),
            batch_reduction="mean",
        )

        assert torch.isfinite(spectral)
        assert torch.isfinite(cluster)


class TestLossesStandaloneCoverage:
    """Standalone coverage for losses only indirectly exercised elsewhere."""

    def test_unbatched_default_branches(self):
        import tgp.utils.losses as losses

        S = torch.tensor([[0.8, 0.2], [0.3, 0.7]], dtype=torch.float32)

        ortho = losses.unbatched_orthogonality_loss(
            S, batch=None, batch_reduction="mean"
        )
        cluster = losses.unbatched_cluster_loss(S, batch=None, batch_reduction="mean")
        entropy = losses.unbatched_entropy_loss(S, num_nodes=None)
        asym = losses.unbatched_asym_norm_loss(
            S, k=2, batch=None, batch_reduction="mean"
        )
        jb = losses.unbatched_just_balance_loss(
            S, batch=None, normalize_loss=True, batch_reduction="mean"
        )

        assert torch.isfinite(ortho)
        assert torch.isfinite(cluster)
        assert torch.isfinite(entropy)
        assert torch.isfinite(asym)
        assert torch.isfinite(jb)

    def test_sparse_default_branches(self):
        import tgp.utils.losses as losses

        edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
        S = torch.tensor([[0.9, 0.1], [0.2, 0.8]], dtype=torch.float32)

        mincut = losses.sparse_mincut_loss(edge_index, S, edge_weight=None, batch=None)
        link = losses.sparse_link_pred_loss(
            S, edge_index, edge_weight=None, batch=None, normalize_loss=True
        )
        totvar = losses.sparse_totvar_loss(edge_index, S, edge_weight=None, batch=None)
        spectral = losses.sparse_spectral_loss(
            edge_index, S, edge_weight=None, batch=None
        )

        assert torch.isfinite(mincut)
        assert torch.isfinite(link)
        assert torch.isfinite(totvar)
        assert torch.isfinite(spectral)

    def test_just_balance_and_spectral_optional_inputs(self):
        import tgp.utils.losses as losses

        adj, S = _make_dense_batch(2, 4, 3, seed=2026)
        adj_pooled = torch.matmul(torch.matmul(S.transpose(-2, -1), adj), S)
        mask = torch.ones(S.size(0), S.size(1), dtype=torch.bool)

        jb = losses.just_balance_loss(
            S,
            normalize_loss=True,
            num_nodes=S.size(-2),
            num_supernodes=S.size(-1),
            batch_reduction="mean",
        )
        spectral = losses.spectral_loss(
            adj,
            S,
            adj_pooled,
            mask=mask,
            num_supernodes=S.size(-1),
            batch_reduction="mean",
        )

        assert torch.isfinite(jb)
        assert torch.isfinite(spectral)

    def test_weighted_bce_reconstruction_loss_paths(self):
        import tgp.utils.losses as losses

        rec_adj = torch.tensor(
            [[[1.0, -0.5], [0.1, 0.9]], [[0.3, -1.2], [0.0, 0.4]]], dtype=torch.float32
        )
        adj = torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.0, 1.0], [1.0, 0.0]]], dtype=torch.float32
        )
        mask = torch.tensor([[True, False], [True, True]], dtype=torch.bool)
        normalizing_const = torch.tensor([1.0, 2.0], dtype=torch.float32)

        loss_balanced = losses.weighted_bce_reconstruction_loss(
            rec_adj=rec_adj,
            adj=adj,
            mask=mask,
            balance_links=True,
            normalizing_const=normalizing_const,
            batch_reduction="mean",
        )
        loss_plain = losses.weighted_bce_reconstruction_loss(
            rec_adj=rec_adj,
            adj=adj,
            mask=None,
            balance_links=False,
            normalizing_const=None,
            batch_reduction="sum",
        )
        loss_balanced_no_mask = losses.weighted_bce_reconstruction_loss(
            rec_adj=rec_adj,
            adj=adj,
            mask=None,
            balance_links=True,
            normalizing_const=None,
            batch_reduction="mean",
        )

        assert torch.isfinite(loss_balanced)
        assert torch.isfinite(loss_plain)
        assert torch.isfinite(loss_balanced_no_mask)

    def test_kl_loss_paths(self):
        from torch.distributions import Beta

        import tgp.utils.losses as losses

        q = Beta(
            torch.tensor([[2.0, 1.5], [1.2, 2.1]], dtype=torch.float32),
            torch.tensor([[1.3, 1.8], [2.0, 1.1]], dtype=torch.float32),
        )
        p = Beta(
            torch.tensor([[1.0, 1.0], [1.0, 1.0]], dtype=torch.float32),
            torch.tensor([[2.0, 2.0], [2.0, 2.0]], dtype=torch.float32),
        )

        with pytest.raises(ValueError, match="Cannot specify both mask and batch"):
            losses.kl_loss(
                q,
                p,
                mask=torch.ones(2, dtype=torch.bool),
                batch=torch.zeros(2, dtype=torch.long),
            )

        with pytest.raises(ValueError, match="Batch size must be specified"):
            losses.kl_loss(
                q, p, batch=torch.zeros(2, dtype=torch.long), batch_size=None
            )

        mask = torch.tensor([True, False], dtype=torch.bool)
        loss_mask = losses.kl_loss(q, p, mask=mask, batch_reduction="mean")
        loss_mask_all = losses.kl_loss(
            q,
            p,
            mask=torch.ones(2, dtype=torch.bool),
            batch_reduction="mean",
        )
        loss_batch = losses.kl_loss(
            q,
            p,
            batch=torch.tensor([0, 1], dtype=torch.long),
            batch_size=2,
            normalizing_const=torch.tensor([1.0, 2.0], dtype=torch.float32),
            batch_reduction="sum",
        )
        loss_plain = losses.kl_loss(q, p, batch_reduction="mean")

        assert torch.isfinite(loss_mask)
        assert torch.isfinite(loss_mask_all)
        assert torch.isfinite(loss_batch)
        assert torch.isfinite(loss_plain)

    def test_cluster_connectivity_prior_loss_paths(self):
        import tgp.utils.losses as losses

        K = torch.tensor([[1.0, 0.2], [0.2, 0.9]], dtype=torch.float32)
        K_mu = torch.tensor([[0.8, 0.1], [0.1, 0.8]], dtype=torch.float32)
        K_var = torch.tensor(0.5, dtype=torch.float32)

        loss_plain = losses.cluster_connectivity_prior_loss(
            K, K_mu, K_var, normalizing_const=None, batch_reduction="mean"
        )
        loss_norm = losses.cluster_connectivity_prior_loss(
            K,
            K_mu,
            K_var,
            normalizing_const=torch.tensor([1.0, 2.0], dtype=torch.float32),
            batch_reduction="sum",
        )

        assert torch.isfinite(loss_plain)
        assert torch.isfinite(loss_norm)

    def test_sparse_bce_reconstruction_loss_paths(self):
        import tgp.utils.losses as losses

        logits = torch.tensor([0.8, -0.3, 0.2, -1.1], dtype=torch.float32)
        labels = torch.tensor([1.0, 0.0, 1.0, 0.0], dtype=torch.float32)

        loss_global, count_global = losses.sparse_bce_reconstruction_loss(
            logits,
            labels,
            edges_batch_id=None,
            batch_size=None,
            batch_reduction="mean",
        )
        loss_batch, count_batch = losses.sparse_bce_reconstruction_loss(
            logits,
            labels,
            edges_batch_id=torch.tensor([0, 0, 1, 1], dtype=torch.long),
            batch_size=2,
            batch_reduction="sum",
        )

        assert torch.isfinite(loss_global)
        assert torch.isfinite(loss_batch)
        assert torch.isfinite(count_global)
        assert torch.all(count_batch >= 1)

    def test_maxcut_loss_paths(self):
        import tgp.utils.losses as losses

        edge_index = torch.tensor([[0, 1, 2, 1], [1, 0, 1, 2]], dtype=torch.long)

        # [N, 1] scores path.
        scores_col = torch.tensor([[0.4], [-0.2], [0.9]], dtype=torch.float32)
        loss_default = losses.maxcut_loss(
            scores_col, edge_index, edge_weight=None, batch=None
        )
        assert torch.isfinite(loss_default)

        # edge_weight squeeze path (E,1) and explicit batch.
        scores_vec = torch.tensor([0.4, -0.2, 0.9], dtype=torch.float32)
        edge_weight_col = torch.tensor(
            [[1.0], [2.0], [1.5], [0.5]], dtype=torch.float32
        )
        batch = torch.tensor([0, 0, 0], dtype=torch.long)
        loss_weighted = losses.maxcut_loss(
            scores_vec,
            edge_index,
            edge_weight=edge_weight_col,
            batch=batch,
            batch_reduction="mean",
        )
        loss_weighted_1d = losses.maxcut_loss(
            scores_vec,
            edge_index,
            edge_weight=torch.tensor([1.0, 2.0, 1.5, 0.5], dtype=torch.float32),
            batch=batch,
            batch_reduction="mean",
        )
        assert torch.isfinite(loss_weighted)
        assert torch.isfinite(loss_weighted_1d)

        with pytest.raises(ValueError, match="Expected scores to have shape"):
            losses.maxcut_loss(
                torch.tensor([[0.1, 0.2], [0.3, 0.4]], dtype=torch.float32),
                edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
            )


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
