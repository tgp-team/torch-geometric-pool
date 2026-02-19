"""Tests for DiffPool pooler losses: link_pred, entropy.

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
    entropy_loss,
    link_pred_loss,
    sparse_link_pred_loss,
    unbatched_entropy_loss,
)


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
