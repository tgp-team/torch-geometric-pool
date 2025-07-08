import pytest
import torch

from tgp.connect import DenseConnect
from tgp.poolers.dmon import DMoNPooling
from tgp.utils.losses import cluster_loss, orthogonality_loss, spectral_loss


@pytest.fixture(scope="module")
def small_graph_dense():
    B, N, F = 1, 4, 3
    torch.manual_seed(0)
    x = torch.randn((B, N, F), dtype=torch.float)
    adj_mat = torch.zeros((N, N), dtype=torch.float)
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_list:
        adj_mat[i, j] = 1.0
        adj_mat[j, i] = 1.0
    adj_mat += torch.eye(N)
    adj = adj_mat.unsqueeze(0)  # Shape: (1, N, N)
    return x, adj


def test_dmon_base(small_graph_dense):
    x, adj = small_graph_dense
    B, N, F = x.shape
    k = 2  # number of clusters

    pooler = DMoNPooling(
        in_channels=F,
        k=k,
        spectral_loss_coeff=2.0,
        cluster_loss_coeff=1.5,
        ortho_loss_coeff=0.5,
        adj_transpose=False,
    )
    pooler.eval()

    out = pooler(x=x, adj=adj, so=None, mask=None, lifting=False)
    loss_dict = out.loss

    S = out.so.s  # shape (B, N, k)
    adj_pooled, _ = DenseConnect(remove_self_loops=False, degree_norm=False)(
        adj, out.so
    )

    mask = torch.ones((B, N), dtype=torch.bool)  # No mask, all nodes valid
    exp_spec = spectral_loss(adj, S, adj_pooled, mask=mask, num_supernodes=k) * 2.0
    exp_cluster = cluster_loss(S, mask=mask, num_supernodes=k) * 1.5
    exp_ortho = orthogonality_loss(S) * 0.5

    assert torch.allclose(loss_dict["spectral_loss"], exp_spec, atol=1e-6)
    assert torch.allclose(loss_dict["cluster_loss"], exp_cluster, atol=1e-6)
    assert torch.allclose(loss_dict["ortho_loss"], exp_ortho, atol=1e-6)


if __name__ == "__main__":
    pytest.main([__file__])
