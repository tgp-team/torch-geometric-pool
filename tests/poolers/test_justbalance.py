"""Tests for JustBalancePooling pooler."""

import pytest
import torch

from tgp.poolers import JustBalancePooling, get_pooler


def test_justbalance_initialization():
    """Test JustBalancePooling initialization with different parameters."""
    pooler = JustBalancePooling(in_channels=16, k=5)
    assert pooler.loss_coeff == 1.0
    assert pooler.batched is True

    pooler = JustBalancePooling(
        in_channels=16,
        k=5,
        loss_coeff=0.5,
        normalize_loss=True,
        batched=False,
    )
    assert pooler.loss_coeff == 0.5
    assert pooler.batched is False


def test_justbalance_batched_forward(pooler_test_graph_dense_batch):
    """Test JustBalancePooling with batched dense inputs."""
    x, adj = pooler_test_graph_dense_batch
    batch_size, n_nodes, n_features = x.shape
    k = 3

    pooler = JustBalancePooling(in_channels=n_features, k=k, batched=True)
    out = pooler(x=x, adj=adj)

    assert out.x.shape == (batch_size, k, n_features)
    assert out.edge_index.shape == (batch_size, k, k)
    assert out.loss is not None
    assert "balance_loss" in out.loss


def test_justbalance_unbatched_single_graph():
    """Test JustBalancePooling in unbatched mode with a single graph."""
    n_nodes, n_features = 10, 16
    k = 5

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, 30))

    pooler = JustBalancePooling(in_channels=n_features, k=k, batched=False)
    out = pooler(x=x, adj=edge_index)

    assert out.x.shape == (1, k, n_features)
    assert out.edge_index.shape == (1, k, k)
    assert out.loss is not None
    assert "balance_loss" in out.loss


def test_justbalance_unbatched_multiple_graphs():
    """Test JustBalancePooling in unbatched mode with multiple graphs."""
    torch.manual_seed(42)
    n_features = 16
    k = 5

    x = torch.randn(25, n_features)
    batch = torch.tensor([0] * 10 + [1] * 15)
    edges_g0 = torch.randint(0, 10, (2, 20))
    edges_g1 = torch.randint(10, 25, (2, 30))
    edge_index = torch.cat([edges_g0, edges_g1], dim=1)

    pooler = JustBalancePooling(in_channels=n_features, k=k, batched=False)
    out = pooler(x=x, adj=edge_index, batch=batch)

    assert out.x.shape == (2, k, n_features), (
        f"expected x shape (2, {k}, {n_features}), got {out.x.shape}"
    )
    assert out.edge_index.shape == (2, k, k), (
        f"expected edge_index shape (2, {k}, {k}), got {out.edge_index.shape}"
    )
    assert out.loss is not None


def test_justbalance_unbatched_sparse_output():
    """Test JustBalancePooling in unbatched mode with sparse output."""
    torch.manual_seed(42)
    n_features = 16
    k = 5

    x = torch.randn(25, n_features)
    batch = torch.tensor([0] * 10 + [1] * 15)
    edges_g0 = torch.randint(0, 10, (2, 20))
    edges_g1 = torch.randint(10, 25, (2, 30))
    edge_index = torch.cat([edges_g0, edges_g1], dim=1)

    pooler = JustBalancePooling(
        in_channels=n_features, k=k, batched=False, sparse_output=True
    )
    out = pooler(x=x, adj=edge_index, batch=batch)

    assert out.x.dim() == 2, f"expected x to be 2D, got shape {out.x.shape}"
    assert out.x.shape[1] == n_features, (
        f"expected x feature dim {n_features}, got {out.x.shape[1]}"
    )
    assert out.batch is not None, "expected batch vector when sparse_output=True"
    assert out.batch.shape[0] == out.x.shape[0], (
        f"batch length {out.batch.shape[0]} should match x nodes {out.x.shape[0]}"
    )
    assert out.edge_index.dim() == 2, (
        f"expected edge_index to be 2D (COO), got shape {out.edge_index.shape}"
    )
    assert out.edge_index.shape[0] == 2, (
        f"expected edge_index rows 2 (COO), got {out.edge_index.shape[0]}"
    )


@pytest.mark.parametrize("train_mode", [True, False])
def test_justbalance_training_mode(pooler_test_graph_dense_batch, train_mode):
    """Test JustBalancePooling gradient computation."""
    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]

    pooler = JustBalancePooling(in_channels=n_features, k=3)
    if train_mode:
        pooler.train()
    else:
        pooler.eval()

    x = x.detach().clone().requires_grad_(True)
    out = pooler(x=x, adj=adj)

    total_loss = sum(out.loss.values())
    total_loss.backward()

    assert x.grad is not None


def test_justbalance_get_pooler_u_suffix():
    """Test get_pooler with jb_u suffix."""
    pooler = get_pooler("jb_u", in_channels=16, k=5)
    assert pooler.batched is False

    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 30))
    out = pooler(x=x, adj=edge_index)

    assert out.x is not None
    assert out.loss is not None


def test_justbalance_lifting_operation(pooler_test_graph_dense_batch):
    """Test the lifting operation in JustBalancePooling."""
    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]

    pooler = JustBalancePooling(in_channels=n_features, k=3)
    regular_out = pooler(x=x, adj=adj)
    lifted_out = pooler(x=regular_out.x, so=regular_out.so, lifting=True)

    assert lifted_out.shape == x.shape


def test_justbalance_batched_vs_unbatched_loss_equality(pooler_test_graph_dense_batch):
    """Batched forward loss dict matches compute_sparse_loss on same data."""
    from tests.test_utils import _dense_batched_to_sparse_unbatched

    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]
    pooler = JustBalancePooling(in_channels=n_features, k=3)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    S = out.so.s
    _, _, S_flat, batch = _dense_batched_to_sparse_unbatched(adj, S)
    loss_sparse = pooler.compute_sparse_loss(S_flat, batch)
    for key in out.loss:
        assert key in loss_sparse
        assert torch.allclose(out.loss[key], loss_sparse[key], rtol=1e-5, atol=1e-5)
    for key in loss_sparse:
        assert key in out.loss


if __name__ == "__main__":
    pytest.main([__file__, "-v"])
