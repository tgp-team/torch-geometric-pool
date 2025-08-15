import pytest
import torch

from tgp.poolers import LaPooling
from tgp.src import PoolingOutput


def make_chain_graph(N=4, F_dim=5):
    """Utility to create a simple undirected chain graph of N nodes with random features of dimension F_dim.
    Returns x (N x F_dim), edge_index (2 x E), edge_weight (E,), batch (N,).
    """
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


def test_forward():
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float).unsqueeze(1)  # make it 2D [N, 1]
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    pooler = LaPooling()
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_supernodes
    assert out.x.shape == (k, 1)


def test_shortest_path():
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float).unsqueeze(1)  # make it 2D [N, 1]
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    pooler = LaPooling(shortest_path_reg=True)
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x, adj=edge_index, edge_weight=edge_weight, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_supernodes
    assert out.x.shape == (k, 1)


if __name__ == "__main__":
    pytest.main([__file__])
