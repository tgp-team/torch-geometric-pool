# tests\test_bnpool.py

import pytest
import torch
from tgp.poolers.bnpool import BNPool
from torch import Tensor


@pytest.fixture
def small_dense_graph():
    x = torch.randn(5, 3)  # Node features
    adj = torch.randint(0, 2, (5, 5)).float()  # Adjacency matrix
    adj = adj.tril() + adj.tril(1)  # Make symmetric
    return x, adj


@pytest.mark.parametrize("k", [2, 3])
@pytest.mark.parametrize("eta", [1.0, 0.5])
@pytest.mark.parametrize("alpha_DP", [0.1, 1.0])
def test_bnpool_init(small_dense_graph, k, eta, alpha_DP):
    x, adj = small_dense_graph
    pooler = BNPool(
        in_channels=3,
        k=k,
        alpha_DP=alpha_DP,
        eta=eta
    )
    assert pooler.k == k
    assert pooler.eta == eta
    assert pooler._alpha_DP == alpha_DP


def test_bnpool_forward(small_dense_graph):
    x, adj = small_dense_graph
    pooler = BNPool(in_channels=3, k=2)
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    assert out.x is not None
    assert out.edge_index is not None
    assert out.loss is not None


def test_bnpool_invalid_params():
    with pytest.raises(ValueError, match="alpha_DP must be positive"):
        BNPool(in_channels=3, k=2, alpha_DP=-1.0)

    with pytest.raises(ValueError, match="k must be positive"):
        BNPool(in_channels=3, k=0)

    with pytest.raises(ValueError, match="eta must be positive"):
        BNPool(in_channels=3, k=2, eta=0)