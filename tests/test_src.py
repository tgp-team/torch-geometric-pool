import pytest
import torch

from tgp.poolers import MinCutPooling
from tgp.src import SRCPooling


@pytest.fixture(scope="module")
def make_chain_graph(N=4, F_dim=5):
    row = torch.arange(N - 1, dtype=torch.long)
    col = row + 1
    edge_index = torch.stack(
        [torch.cat([row, col]), torch.cat([col, row])], dim=0
    )  # undirected chain
    E = edge_index.size(1)
    x = torch.randn((N, F_dim), dtype=torch.float)
    edge_weight = torch.ones(E, dtype=torch.float)
    batch = torch.zeros(N, dtype=torch.long)
    return x, edge_index, edge_weight, batch


# def test_reducer_none(make_chain_graph):
#     x, edge_index, edge_weight, batch = make_chain_graph

#     pooler = SRCPooling(selector=GraclusSelect(), connector=SparseConnect())
#     out = pooler.coarsen_graph(
#         edge_index=edge_index, edge_weight=edge_weight, x=x, batch=batch
#     )
#     assert out.batch is None


def test_compute_loss_none():
    pooler = SRCPooling()
    assert pooler.compute_loss() is None


def test_preprocessing(make_chain_graph):
    x, edge_index, edge_weight, batch = make_chain_graph

    # add a trailing dimension to edge_weight to simulate a feature dimension
    edge_weight = edge_weight.unsqueeze(-1)

    pooler = MinCutPooling(
        k=2,
        in_channels=x.size(-1),
    )
    x, adj, mask = pooler.preprocessing(
        edge_index=edge_index, edge_weight=edge_weight, x=x, batch=batch
    )

    assert adj.dim() == 3


if __name__ == "__main__":
    pytest.main([__file__])
