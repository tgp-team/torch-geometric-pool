import numpy as np
import pytest
import torch


@pytest.fixture
def set_random_seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_dense_graph(set_random_seed):
    x = torch.randn(5, 3)  # Node features
    adj = torch.randint(0, 2, (5, 5)).float()  # Adjacency matrix
    adj = adj.tril() + adj.tril().T  # Make symmetric
    return x, adj


def test_bnpool_forward_shapes(small_dense_graph):
    x, adj = small_dense_graph
    _, n_nodes, n_features = 1, x.size(0), x.size(1)

    from tgp.poolers.bnpool import BNPool

    pooler = BNPool(in_channels=n_features, k=2)
    pooler.eval()

    out = pooler(x=x, adj=adj)

    # Check output shapes
    assert out.x.shape[0] <= n_nodes, "Pooled nodes should be <= input nodes"
    assert out.x.shape[1] == n_features, "Feature dimension should remain unchanged"
    assert out.edge_index.shape[0] == out.x.shape[0], (
        "Adjacency matrix should match pooled nodes"
    )
    assert out.edge_index.shape[1] == out.x.shape[0], (
        "Adjacency matrix should match pooled nodes"
    )


@pytest.mark.parametrize("n_nodes", [1, 5, 10])
def test_bnpool_different_sizes(set_random_seed, n_nodes):
    x = torch.randn(n_nodes, 3)
    adj = torch.randint(0, 2, (n_nodes, n_nodes)).float()
    adj = adj.tril() + adj.tril().T

    from tgp.poolers.bnpool import BNPool

    pooler = BNPool(in_channels=3, k=2)
    pooler.eval()

    out = pooler(x=x, adj=adj)
    assert out.x is not None
    assert out.edge_index is not None


def test_bnpool_with_mask():
    x = torch.randn(5, 3)
    adj = torch.randint(0, 2, (5, 5)).float()
    mask = torch.ones(5, dtype=torch.bool)
    mask[3:] = False  # Mark last two nodes as padding

    from tgp.poolers.bnpool import BNPool

    pooler = BNPool(in_channels=3, k=2)
    pooler.eval()

    out = pooler(x=x, adj=adj, mask=mask)
    assert out.x is not None
    assert out.edge_index is not None
