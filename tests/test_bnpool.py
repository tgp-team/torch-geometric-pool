from itertools import product

import numpy as np
import pytest
import torch


@pytest.fixture
def set_random_seed():
    torch.manual_seed(42)
    np.random.seed(42)


@pytest.fixture
def small_dense_graph(set_random_seed, request):
    batch_size, n_nodes, n_features = request.param
    x = torch.randn(batch_size, n_nodes, n_features)  # Node features
    adj = torch.randint(
        0, 2, (batch_size, n_nodes, n_nodes)
    ).float()  # Adjacency matrix
    adj = (adj + adj.transpose(-1, -2)) // 2  # Make symmetric
    return x, adj


@pytest.mark.parametrize(
    "small_dense_graph", product([1, 5, 10], [1, 3, 7], [1, 9, 12]), indirect=True
)
def test_bnpool_forward_shapes(small_dense_graph):
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


@pytest.mark.parametrize("small_dense_graph", [(5, 7, 3)], indirect=True)
def test_bnpool_with_mask(small_dense_graph):
    x, adj = small_dense_graph
    batch_size, n_nodes, _ = x.shape
    mask = torch.ones(batch_size, n_nodes, dtype=torch.bool)
    mask[:, 3:] = False  # Mark last two nodes as padding

    from tgp.poolers.bnpool import BNPool

    pooler = BNPool(in_channels=3, k=2)
    pooler.eval()

    out = pooler(x=x, adj=adj, mask=mask)
    assert out.x is not None
    assert out.edge_index is not None
