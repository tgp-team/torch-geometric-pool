import pytest
import torch

from tgp.poolers.bnpool import BNPool


def test_bnpool_initialization():
    """Test BNPool initialization with different parameters."""
    # Test valid initialization
    pooler = BNPool(in_channels=4, k=3)
    assert pooler.k == 3
    assert pooler.alpha_DP == 1.0

    # Test custom parameters
    pooler = BNPool(
        in_channels=4, k=3, alpha_DP=2.0, K_var=0.5, K_mu=5.0, K_init=0.5, eta=0.8
    )
    assert pooler.alpha_DP == 2.0
    assert pooler.K_var_val == 0.5
    assert pooler.K_mu_val == 5.0
    assert pooler.K_init_val == 0.5
    assert pooler.eta == 0.8


def test_bnpool_invalid_parameters():
    """Test BNPool initialization with invalid parameters."""
    with pytest.raises(ValueError, match="alpha_DP must be positive"):
        BNPool(in_channels=4, k=3, alpha_DP=-1.0)

    with pytest.raises(ValueError, match="K_var must be positive"):
        BNPool(in_channels=4, k=3, K_var=-1.0)

    with pytest.raises(ValueError, match="eta must be positive"):
        BNPool(in_channels=4, k=3, eta=-1.0)

    with pytest.raises(ValueError, match="max_k must be positive"):
        BNPool(in_channels=4, k=-3)


@pytest.mark.parametrize("train_k", [True, False])
def test_bnpool_training_mode(pooler_test_graph_dense_batch, train_k):
    """Test BNPool behavior in training mode."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, train_K=train_k)
    pooler.train()

    # Enable gradient tracking
    x.requires_grad_(True)

    # Forward pass
    out = pooler(x=x, adj=adj)

    # Check if loss components are present
    assert isinstance(out.loss, dict)
    assert "quality" in out.loss
    assert "kl" in out.loss
    assert "K_prior" in out.loss

    # Check if losses are differentiable
    total_loss = sum(out.loss.values())
    total_loss.backward()

    # Check if gradients are computed
    assert x.grad is not None
    if pooler.train_K:
        assert pooler.K.grad is not None
    else:
        assert pooler.K.grad is None


def test_bnpool_eval_mode(pooler_test_graph_dense_batch):
    x, adj = pooler_test_graph_dense_batch
    batch_size, n_nodes, n_features = x.shape
    k = 2
    pooler = BNPool(in_channels=n_features, k=k)
    pooler.eval()

    out = pooler(x=x, adj=adj)

    # Check output shapes
    assert out.x.shape[0] == batch_size, "Batch dimension should be 1 for pooled x"
    assert out.x.shape[1] == k, "Number of nodes should be equal to k"
    assert out.x.shape[2] == n_features, "Feature dimension should remain unchanged"

    assert out.edge_index.shape[0] == batch_size, (
        "Batch dimension should be 1 for edge_index"
    )
    assert out.edge_index.shape[1] == out.edge_index.shape[2] == k, (
        "Adjacency matrix size should match number of clusters k"
    )


def test_bnpool_batched_forward(pooler_test_graph_dense_batch):
    """Test BNPool with batched dense inputs."""
    x, adj = pooler_test_graph_dense_batch

    pooler = BNPool(
        in_channels=x.shape[-1],
        k=3,
    )
    out = pooler(x=x, adj=adj)
    assert out.x is not None
    assert out.edge_index is not None


def test_bnpool_lifting_operation(pooler_test_graph_dense_batch):
    """Test the lifting operation in BNPool."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3)

    # First do regular pooling to get selection output
    regular_out = pooler(x=x, adj=adj)

    # Then test lifting operation
    lifted_out = pooler(x=regular_out.x, so=regular_out.so, lifting=True)

    # Check if lifted output has same dimensions as input
    assert lifted_out.shape == x.shape


def test_bnpool_batched_dense_output_mask(pooler_test_graph_dense_batch):
    """Batched dense output: out.mask equals so.out_mask, shape [B, K_max]."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=True, sparse_output=False)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)
    assert out.mask.dim() == 2
    assert out.mask.shape[0] == out.x.shape[0]
    assert out.mask.shape[1] == out.x.shape[1]
    assert torch.equal(out.mask, (out.so.s.sum(dim=-2) > 0))


def test_bnpool_batched_sparse_output_no_mask(pooler_test_graph_dense_batch):
    """Batched sparse output: out.mask equals so.out_mask (so.s is 3D so mask is not None)."""
    x, adj = pooler_test_graph_dense_batch
    pooler = BNPool(in_channels=x.shape[-1], k=3, batched=True, sparse_output=True)
    pooler.eval()
    out = pooler(x=x, adj=adj)
    assert out.so is not None
    assert out.mask is not None
    assert torch.equal(out.mask, out.so.out_mask)
