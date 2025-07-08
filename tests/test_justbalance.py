import pytest
import torch
from torch import Tensor

from tgp.data import NormalizeAdj
from tgp.poolers.just_balance import JustBalancePooling
from tgp.utils.losses import just_balance_loss


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


@pytest.mark.parametrize("normalize_loss", [True, False])
def test_just_balance_base(small_dense_graph, normalize_loss):
    x, adj = small_dense_graph
    pooler = JustBalancePooling(
        in_channels=3, k=3, loss_coeff=0.5, normalize_loss=normalize_loss
    )
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss
    assert "balance_loss" in loss_dict
    balance_loss = loss_dict["balance_loss"]
    assert isinstance(balance_loss, Tensor)


def test_loss_nan():
    pooler = JustBalancePooling(in_channels=3, k=3, loss_coeff=0.5)
    S_nan = torch.full((10, 3), float("nan"), dtype=torch.float).unsqueeze(0)
    with pytest.raises(ValueError):
        _ = pooler.compute_loss(S=S_nan, mask=None, num_nodes=10, num_supernodes=3)


def test_data_transforms_returns_normalizeadj():
    transform = JustBalancePooling.data_transforms()
    assert isinstance(transform, NormalizeAdj)
    assert hasattr(transform, "delta")


@pytest.mark.parametrize("batch_reduction", ["sum", "mean"])
def test_just_balance_loss(batch_reduction):
    S = torch.tensor(
        [[1.0, 0.0], [1.0, 0.0], [0.0, 1.0], [0.0, 1.0]], dtype=torch.float
    ).unsqueeze(0)

    loss_balanced = just_balance_loss(S, batch_reduction=batch_reduction)
    assert pytest.approx(loss_balanced.item(), rel=1e-6) == -1.0

    # Check if it raises ValueError for non valid reduction
    with pytest.raises(ValueError):
        _ = just_balance_loss(S, batch_reduction="not_implemented")


if __name__ == "__main__":
    pytest.main([__file__])
