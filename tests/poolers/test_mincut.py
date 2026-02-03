"""Tests for MinCutPooling pooler."""

import pytest
import torch

from tgp.poolers import MinCutPooling, get_pooler


def test_mincut_initialization():
    """Test MinCutPooling initialization with different parameters."""
    # Test valid initialization
    pooler = MinCutPooling(in_channels=16, k=5)
    assert pooler.cut_loss_coeff == 1.0
    assert pooler.ortho_loss_coeff == 1.0
    assert pooler.batched is True

    # Test custom parameters
    pooler = MinCutPooling(
        in_channels=16,
        k=5,
        cut_loss_coeff=0.5,
        ortho_loss_coeff=2.0,
        batched=False,
    )
    assert pooler.cut_loss_coeff == 0.5
    assert pooler.ortho_loss_coeff == 2.0
    assert pooler.batched is False


def test_mincut_batched_forward(pooler_test_graph_dense_batch):
    """Test MinCutPooling with batched dense inputs."""
    x, adj = pooler_test_graph_dense_batch
    batch_size, n_nodes, n_features = x.shape
    k = 3

    pooler = MinCutPooling(in_channels=n_features, k=k, batched=True)
    out = pooler(x=x, adj=adj)

    # Check output shapes
    assert out.x.shape == (batch_size, k, n_features)
    assert out.edge_index.shape == (batch_size, k, k)
    assert out.loss is not None
    assert "cut_loss" in out.loss
    assert "ortho_loss" in out.loss


def test_mincut_unbatched_single_graph():
    """Test MinCutPooling in unbatched mode with a single graph."""
    n_nodes, n_features = 10, 16
    k = 5

    x = torch.randn(n_nodes, n_features)
    edge_index = torch.randint(0, n_nodes, (2, 30))

    pooler = MinCutPooling(in_channels=n_features, k=k, batched=False)
    out = pooler(x=x, adj=edge_index)

    # With sparse_output=False (default), output is batched dense [1, K, F]
    assert out.x.shape == (1, k, n_features)
    assert out.edge_index.shape == (1, k, k)
    assert out.loss is not None
    assert "cut_loss" in out.loss
    assert "ortho_loss" in out.loss


def test_mincut_unbatched_multiple_graphs():
    """Test MinCutPooling in unbatched mode with multiple graphs."""
    torch.manual_seed(42)
    n_features = 16
    k = 5

    # Two graphs: 10 nodes and 15 nodes
    x = torch.randn(25, n_features)
    batch = torch.tensor([0] * 10 + [1] * 15)

    # Create edges within each graph (ensure each graph has edges)
    edges_g0 = torch.randint(0, 10, (2, 20))
    edges_g1 = torch.randint(10, 25, (2, 30))
    edge_index = torch.cat([edges_g0, edges_g1], dim=1)

    pooler = MinCutPooling(in_channels=n_features, k=k, batched=False)
    out = pooler(x=x, adj=edge_index, batch=batch)

    # With sparse_output=False (default), output is batched dense [B, K, F]
    assert out.x.shape == (2, k, n_features), (
        f"expected x shape (2, {k}, {n_features}), got {out.x.shape}"
    )
    # edge_index can be dense [B, K, K] when sparse_output=False
    assert out.edge_index.shape == (2, k, k), (
        f"expected edge_index shape (2, {k}, {k}), got {out.edge_index.shape}"
    )
    assert out.loss is not None


def test_mincut_unbatched_sparse_output():
    """Test MinCutPooling in unbatched mode with sparse output."""
    torch.manual_seed(42)
    n_features = 16
    k = 5

    x = torch.randn(25, n_features)
    batch = torch.tensor([0] * 10 + [1] * 15)
    edges_g0 = torch.randint(0, 10, (2, 20))
    edges_g1 = torch.randint(10, 25, (2, 30))
    edge_index = torch.cat([edges_g0, edges_g1], dim=1)

    pooler = MinCutPooling(
        in_channels=n_features, k=k, batched=False, sparse_output=True
    )
    out = pooler(x=x, adj=edge_index, batch=batch)

    # With sparse_output=True, output is sparse: x is [N_pooled, F], batch is [N_pooled]
    assert out.x.dim() == 2, f"expected x to be 2D, got shape {out.x.shape}"
    assert out.x.shape[1] == n_features, (
        f"expected x feature dim {n_features}, got {out.x.shape[1]}"
    )
    assert out.batch is not None, "expected batch vector when sparse_output=True"
    assert out.batch.shape[0] == out.x.shape[0], (
        f"batch length {out.batch.shape[0]} should match x nodes {out.x.shape[0]}"
    )
    # COO format: [2, num_edges]
    assert out.edge_index.dim() == 2, (
        f"expected edge_index to be 2D (COO), got shape {out.edge_index.shape}"
    )
    assert out.edge_index.shape[0] == 2, (
        f"expected edge_index rows 2 (COO), got {out.edge_index.shape[0]}"
    )


@pytest.mark.parametrize("train_mode", [True, False])
def test_mincut_training_mode(pooler_test_graph_dense_batch, train_mode):
    """Test MinCutPooling gradient computation."""
    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]

    pooler = MinCutPooling(in_channels=n_features, k=3)
    if train_mode:
        pooler.train()
    else:
        pooler.eval()

    # Use a leaf tensor so that .grad is populated after backward()
    # (x.clone().requires_grad_(True) is non-leaf and does not get .grad by default)
    x = x.detach().clone().requires_grad_(True)
    out = pooler(x=x, adj=adj)

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None


def test_mincut_get_pooler_u_suffix():
    """Test get_pooler with mincut_u suffix."""
    pooler = get_pooler("mincut_u", in_channels=16, k=5)
    assert pooler.batched is False

    # Test that it works with unbatched input
    x = torch.randn(10, 16)
    edge_index = torch.randint(0, 10, (2, 30))
    out = pooler(x=x, adj=edge_index)

    assert out.x is not None
    assert out.loss is not None


def test_mincut_lifting_operation(pooler_test_graph_dense_batch):
    """Test the lifting operation in MinCutPooling."""
    x, adj = pooler_test_graph_dense_batch
    n_features = x.shape[-1]

    pooler = MinCutPooling(in_channels=n_features, k=3)

    # First do regular pooling to get selection output
    regular_out = pooler(x=x, adj=adj)

    # Then test lifting operation
    lifted_out = pooler(x=regular_out.x, so=regular_out.so, lifting=True)

    # Check if lifted output has same dimensions as input
    assert lifted_out.shape == x.shape
