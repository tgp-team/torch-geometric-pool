import pytest
import torch
from torch_geometric.data import Batch, Data

from tgp.poolers import NMFPooling
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


@pytest.fixture(scope="module")
def sparse_batch_graph():
    edge_index_1 = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]], dtype=torch.long)
    edge_weight_1 = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    x_1 = torch.randn((4, 3), dtype=torch.float)

    edge_index_2 = torch.tensor(
        [[1, 2, 3, 4, 2, 0], [0, 1, 2, 2, 3, 3]], dtype=torch.long
    )
    edge_weight_2 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5, 0.6], dtype=torch.float)
    x_2 = torch.randn((5, 3), dtype=torch.float)

    edge_index_3 = torch.tensor([[0, 1, 3, 3, 2], [1, 0, 1, 2, 3]], dtype=torch.long)
    edge_weight_3 = torch.tensor([0.1, 0.2, 0.3, 0.4, 0.5], dtype=torch.float)
    x_3 = torch.randn((4, 3), dtype=torch.float)

    data_batch = Batch.from_data_list(
        [
            Data(edge_index=edge_index_1, edge_attr=edge_weight_1, x=x_1),
            Data(edge_index=edge_index_2, edge_attr=edge_weight_2, x=x_2),
            Data(edge_index=edge_index_3, edge_attr=edge_weight_3, x=x_3),
        ]
    )
    return data_batch


def test_nmf_select(small_graph_dense):
    _, adj = small_graph_dense
    selector = NMFSelect(k=2)
    out = selector.forward(edge_index=adj)
    assert out.s.size(1) == 4  # Should select 2 nodes


def test_nmf_precoarsening(sparse_batch_graph):
    data_batch = sparse_batch_graph
    assert data_batch.num_graphs == 3
    assert data_batch.num_nodes == 13

    pooling_out = NMFPooling(k=3).precoarsening(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
        batch=data_batch.batch,
        num_nodes=data_batch.num_nodes,
    )

    assert pooling_out.so.s.size(0) == 13
    assert pooling_out.so.s.size(1) == 9
    assert pooling_out.batch.size(0) == 9


def test_batch_none(sparse_batch_graph):
    data_batch = sparse_batch_graph
    assert data_batch.num_graphs == 3
    assert data_batch.num_nodes == 13

    pooling_out = NMFPooling(k=3).precoarsening(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
    )

    assert pooling_out.batch.size(0) == 3


if __name__ == "__main__":
    pytest.main([__file__])
