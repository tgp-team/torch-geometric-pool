import random
from itertools import product

import numpy as np
import pytest
import torch

from tgp.poolers.bnpool import BNPool


@pytest.fixture
def set_random_seed():
    torch.manual_seed(42)
    np.random.seed(42)
    random.seed(42)


@pytest.fixture(params=list(product([1, 3, 5], [1, 4, 8], [1, 2, 12])))
def small_dense_graph(set_random_seed, request):
    batch_size, n_nodes, n_features = request.param
    x = torch.randn(batch_size, n_nodes, n_features)  # Node features
    adj = torch.randint(
        0, 2, (batch_size, n_nodes, n_nodes)
    ).float()  # Adjacency matrix
    adj = (adj + adj.transpose(-1, -2)) // 2  # Make symmetric
    return x, adj


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
def test_bnpool_training_mode(small_dense_graph, train_k):
    """Test BNPool behavior in training mode."""
    x, adj = small_dense_graph
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


def test_bnpool_eval_mode(small_dense_graph):
    x, adj = small_dense_graph
    batch_size, n_nodes, n_features = x.shape

    from tgp.poolers.bnpool import BNPool

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


def test_bnpool_with_mask_patterns(small_dense_graph):
    """Test BNPool with different mask patterns."""
    x, adj = small_dense_graph
    batch_size, n_nodes = x.shape[:2]

    # Test different mask patterns
    mask_patterns = [
        torch.ones(batch_size, n_nodes, dtype=torch.bool),  # All nodes
        torch.zeros(batch_size, n_nodes, dtype=torch.bool),  # No nodes
        torch.bernoulli(torch.ones(batch_size, n_nodes) * 0.7).bool(),  # Random mask
    ]

    pooler = BNPool(in_channels=x.shape[-1], k=3)
    for mask in mask_patterns:
        out = pooler(x=x, adj=adj, mask=mask)
        assert out.x is not None
        assert out.edge_index is not None


def test_bnpool_loss_scaling(small_dense_graph):
    """Test loss scaling behavior of BNPool."""
    x, adj = small_dense_graph

    # Test with and without loss scaling
    for rescale_loss in [True, False]:
        pooler = BNPool(in_channels=x.shape[-1], k=3, rescale_loss=rescale_loss)
        out = pooler(x=x, adj=adj)

        # Verify loss components exist
        assert all(k in out.loss for k in ["quality", "kl", "K_prior"])

        # Check if losses are properly scaled
        if rescale_loss:
            n_nodes = x.shape[1]
            assert torch.all(out.loss["quality"] <= 1.0 * n_nodes)


def test_bnpool_lifting_operation(small_dense_graph):
    """Test the lifting operation in BNPool."""
    x, adj = small_dense_graph
    pooler = BNPool(in_channels=x.shape[-1], k=3)

    # First do regular pooling to get selection output
    regular_out = pooler(x=x, adj=adj)

    # Then test lifting operation
    lifted_out = pooler(x=regular_out.x, so=regular_out.so, lifting=True)

    # Check if lifted output has same dimensions as input
    assert lifted_out.shape == x.shape
