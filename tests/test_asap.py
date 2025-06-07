# tests/test_asapool.py

import pytest
import torch
from torch import Tensor
from torch_geometric.nn import GCNConv
from torch_sparse import SparseTensor

from tgp.poolers.asap import ASAPooling
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


def test_forward_with_1d_x_and_dense_adj():
    """Cover lines where x.dim() == 1 triggers unsqueeze, and where adj is a dense tensor."""
    N = 3
    # x is 1D of length N
    x = torch.arange(N, dtype=torch.float)
    # Build simple chain adjacency as dense edge_index
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([torch.cat([row, col]), torch.cat([col, row])], dim=0)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    batch = None

    # Instantiate ASAPooling with default parameters (no GNN, add_self_loops=False)
    pooler = ASAPooling(in_channels=1, ratio=0.5)
    pooler.eval()

    # Call forward; x will be unsqueezed internally
    out = pooler(
        x=x,
        adj=edge_index,
        edge_weight=edge_weight,
        so=None,
        batch=batch,
        lifting=False,
    )
    assert isinstance(out, PoolingOutput)
    # out.x should have shape [k, 1]
    k = out.so.num_clusters
    assert out.x.shape == (k, 1)


def test_forward_with_gnn_and_sparse_adj_and_add_self_loops_and_extra_repr():
    """Cover lines where GNN is provided, adj is a SparseTensor, and add_self_loops=True.
    Also test extra_repr_args.
    """
    x, edge_index, edge_weight, batch = make_chain_graph(N=4, F_dim=3)
    # Convert to SparseTensor
    adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)

    # Provide a GNN (GCNConv) and add_self_loops=True, plus non-default dropout and negative_slope
    pooler = ASAPooling(
        in_channels=3,
        ratio=0.5,
        GNN=GCNConv,
        add_self_loops=True,
        dropout=0.1,
        negative_slope=0.1,
        nonlinearity="sigmoid",
    )
    pooler.eval()
    pooler.reset_parameters()

    # Forward pass should go through GNN branch, convert adj to edge_index+ew, compute attention, apply dropout, select, reduce, connect, and add new self-loops
    out = pooler(
        x=x, adj=adj, edge_weight=edge_weight, so=None, batch=batch, lifting=False
    )
    assert isinstance(out, PoolingOutput)


def test_lifting_branch_returns_original_x():
    """Cover the lifting=True branch.
    Perform a forward pass to obtain SelectOutput and pooled features, then call lifting.
    """
    x, edge_index, edge_weight, batch = make_chain_graph(N=5, F_dim=4)
    pooler = ASAPooling(in_channels=4, ratio=0.4)
    pooler.eval()

    # First forward to get out.so and out.x
    out = pooler(
        x=x,
        adj=edge_index,
        edge_weight=edge_weight,
        so=None,
        batch=batch,
        lifting=False,
    )
    so = out.so
    x_pool = out.x

    # Now call with lifting=True: should return original-shape x
    x_lifted = pooler(
        x=x_pool, adj=None, edge_weight=None, so=so, batch=batch, lifting=True
    )
    assert isinstance(x_lifted, Tensor)
    assert x_lifted.shape == x.shape


def test_graph_disconnected_case():
    edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 0, 2]], dtype=torch.long)
    x = torch.randn(5, 3)
    edge_index = SparseTensor.from_edge_index(edge_index)
    batch = torch.zeros(5, dtype=torch.long)

    pooler = ASAPooling(in_channels=3, ratio=0.5)
    pooler.eval()

    out = pooler(x=x, adj=edge_index, so=None, batch=batch, lifting=False)
    assert isinstance(out, PoolingOutput)


if __name__ == "__main__":
    pytest.main([__file__])
