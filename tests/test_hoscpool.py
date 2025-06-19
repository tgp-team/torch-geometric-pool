import pytest
import torch
from torch import Tensor

from tgp.poolers.hosc import HOSCPooling


@pytest.fixture
def small_dense_graph():
    B, N, F = 1, 12, 3
    torch.manual_seed(0)
    x = torch.randn((B, N, F), dtype=torch.float)
    adj_mat = torch.zeros((N, N), dtype=torch.float)
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_list:
        adj_mat[i, j] = 1.0
        adj_mat[j, i] = 1.0
    adj_mat += torch.eye(N)
    adj = adj_mat.unsqueeze(0)
    return x, adj


@pytest.mark.parametrize("alpha", [-0.5, 1.5])
@pytest.mark.parametrize("mu", [0.0, 0.1])
def test_hosc_different_params(small_dense_graph, alpha, mu):
    x, adj = small_dense_graph
    pooler = HOSCPooling(in_channels=3, k=2, mu=mu, alpha=alpha, hosc_ortho=False)
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss
    assert "ortho_loss" in loss_dict
    ortho_loss = loss_dict["ortho_loss"]
    assert isinstance(ortho_loss, Tensor)
    assert "hosc_loss" in loss_dict
    hosc_loss = loss_dict["hosc_loss"]
    assert torch.isfinite(hosc_loss).all()


def test_hosc_ortho_true(small_dense_graph):
    x, adj = small_dense_graph
    pooler = HOSCPooling(in_channels=3, k=2, mu=0.1, alpha=0.5, hosc_ortho=True)
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss
    assert "ortho_loss" in loss_dict


if __name__ == "__main__":
    pytest.main([__file__])
