import pytest
import torch
from torch_geometric.utils import add_self_loops
from torch_sparse import SparseTensor

from tgp.mp.gtvconv import GTVConv, gtv_adj_weights


def test_gtv_adj_weights_source_to_target():
    # Build a simple undirected 2-node graph with edge weights [1,1]
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    num_nodes = 2

    new_ei, new_ew = gtv_adj_weights(
        edge_index=edge_index,
        edge_weight=edge_weight,
        num_nodes=num_nodes,
        flow="source_to_target",
        coeff=1.0,
    )
    # After adding self-loops, new_ei should have 4 edges: (0,1),(1,0),(0,0),(1,1)
    assert new_ei.size(1) == 4
    expected_ew = torch.tensor([1.0, 1.0, 0.0, 0.0])
    assert torch.allclose(new_ew, expected_ew)


@pytest.mark.parametrize("bias", [True, False])
def test_gtvconv_forward_dense_and_mask(bias):
    # Build a small dense adjacency (3 nodes): complete graph with self-loops
    B, N, F, out_channels = 2, 3, 2, 4
    x = torch.randn((B, N, F), dtype=torch.float)

    # Dense adjacency matrix [N,N]
    adj = torch.tensor(
        [
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
            [1.0, 1.0, 1.0],
        ],
        dtype=torch.float,
    )

    conv = GTVConv(
        in_channels=F,
        out_channels=out_channels,
        bias=bias,
        delta_coeff=0.5,
        eps=1e-3,
        act="relu",
    )
    conv.eval()

    # Forward without mask
    out = conv(x, adj)
    assert out.shape == (B, N, out_channels)
    assert torch.isfinite(out).all()

    # Test mask: mask zeros out node 1, so out[1] should be zero
    mask = torch.tensor([1, 0, 1], dtype=torch.bool)
    out_masked = conv(x, adj, mask=mask)
    assert out_masked.shape == (B, N, out_channels)
    assert torch.allclose(out_masked[:, 1], torch.zeros_like(out_masked[:, 1]))

@pytest.mark.parametrize("bias", [True, False])
def test_gtvconv_forward_sparse(bias):
    # Build a small sparse adjacency: 3-node path 0-1-2 + self-loops
    N, F, out_channels = 3, 2, 4
    x = torch.randn((N, F), dtype=torch.float)

    # Directed edges: (0,1),(1,2)
    row = torch.tensor([0, 1], dtype=torch.long)
    col = torch.tensor([1, 2], dtype=torch.long)
    edge_index = torch.stack([row, col], dim=0)
    edge_weight = torch.tensor([1.0, 1.0], dtype=torch.float)
    edge_index, edge_weight = add_self_loops(
        edge_index, edge_attr=edge_weight, num_nodes=N
    )
    # Build SparseTensor from (edge_index, edge_weight)
    sparse_adj = SparseTensor.from_edge_index(edge_index, edge_attr=edge_weight)

    conv = GTVConv(
        in_channels=F,
        out_channels=out_channels,
        bias=bias,
        delta_coeff=1.0,
        eps=1e-4,
        act="PReLU",
    )
    conv.eval()

    # Forward with sparse adjacency
    out = conv(x, sparse_adj)
    assert out.shape == (N, out_channels)
    assert torch.isfinite(out).all()

    sparse_adj2 = SparseTensor.from_edge_index(edge_index, edge_attr=None)

    conv = GTVConv(
        in_channels=F,
        out_channels=out_channels,
        bias=True,
        delta_coeff=1.0,
        eps=1e-4,
        act="PReLU",
    )
    conv.eval()

    # Forward with sparse adjacency
    out = conv(x, sparse_adj2)
    assert out.shape == (N, out_channels)
    assert torch.isfinite(out).all()

if __name__ == "__main__":
    pytest.main([__file__])
