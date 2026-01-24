import pytest
import torch
from torch import Tensor

from tgp.poolers.hosc import HOSCPooling


@pytest.mark.parametrize("alpha", [-0.5, 1.5])
@pytest.mark.parametrize("mu", [0.0, 0.1])
def test_hosc_different_params(pooler_test_graph_dense, alpha, mu):
    x, adj = pooler_test_graph_dense
    F = x.size(2)
    pooler = HOSCPooling(in_channels=F, k=2, mu=mu, alpha=alpha, hosc_ortho=False)
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss
    assert "ortho_loss" in loss_dict
    ortho_loss = loss_dict["ortho_loss"]
    assert isinstance(ortho_loss, Tensor)
    assert "hosc_loss" in loss_dict
    hosc_loss = loss_dict["hosc_loss"]
    assert torch.isfinite(hosc_loss).all()


def test_hosc_ortho_true(pooler_test_graph_dense):
    x, adj = pooler_test_graph_dense
    F = x.size(2)
    pooler = HOSCPooling(in_channels=F, k=2, mu=0.1, alpha=0.5, hosc_ortho=True)
    pooler.eval()
    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss
    assert "ortho_loss" in loss_dict


if __name__ == "__main__":
    pytest.main([__file__])
