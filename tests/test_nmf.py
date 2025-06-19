import pytest
import torch

from tgp.select.nmf_select import NMFSelect


@pytest.fixture(scope="module")
def small_graph_dense():
    B, N, F = 1, 4, 3
    torch.manual_seed(0)
    x = torch.randn((B, N, F), dtype=torch.float)
    adj_mat = torch.zeros((N, N))
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 0)]
    for i, j in edge_list:
        adj_mat[i, j] = 1.0
        adj_mat[j, i] = 1.0
    adj_mat += torch.eye(N)
    adj = adj_mat.unsqueeze(0)  # Shape: (1, N, N)
    return x, adj.long()


def test_nmf_select(small_graph_dense):
    _, adj = small_graph_dense
    selector = NMFSelect(k=2)
    out = selector.forward(edge_index=adj)
    assert out.s.size(1) == 4  # Should select 2 nodes


def test_nmf_select_sparse_graph(small_graph_dense):
    # create a list of edge indices for a single graph
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    selector = NMFSelect(k=2)
    out = selector.forward(edge_index=edge_index)
    assert out.s.size(0) == 4  # Should select 2 nodes


if __name__ == "__main__":
    pytest.main([__file__])
