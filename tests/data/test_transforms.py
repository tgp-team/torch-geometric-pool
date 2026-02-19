import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from tgp.data.transforms import NormalizeAdj, SortNodes


def test_normalizeadj_with_edge_attr():
    N = 3
    edge_index = torch.tensor([[0, 1], [1, 0]], dtype=torch.long)  # undirected edge
    edge_weight = torch.tensor([2.0, 2.0], dtype=torch.float)
    # Two edges (one in each direction); edge_attr shape [2,2]
    edge_attr = torch.tensor([[1.0, 10.0], [2.0, 20.0]], dtype=torch.float)
    x = torch.randn((N, 4))
    data = Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
    )
    data.num_nodes = N

    delta = 0.5
    transform = NormalizeAdj(delta=delta)
    data_t = transform(data)

    # After transform, edge_index, edge_weight, and edge_attr should exist
    assert data_t.edge_index is not None
    assert data_t.edge_weight is not None
    assert data_t.edge_attr is not None

    # Convert to dense adjacency to inspect values and attributes
    dense_aw = to_dense_adj(data_t.edge_index, edge_attr=data_t.edge_weight).squeeze(0)
    diag = torch.diagonal(dense_aw)
    assert torch.all(diag > 0)

    num_edges_after = data_t.edge_attr.size(0)
    assert num_edges_after >= (edge_index.size(1) + N)

    # Each attr should still have dimension 2
    assert data_t.edge_attr.size(1) == 2


def test_sortnodes_with_edge_attr():
    # Create a graph with 3 nodes, labels y out of order, and edge_attr
    # Node labels: [2, 0, 1] -> sorted order [0, 1, 2]
    y = torch.tensor([2, 0, 1], dtype=torch.long)
    x = torch.tensor([[1.0], [2.0], [3.0]])  # features corresponding to nodes 0,1,2
    # Edges: 0->1, 1->2, 2->0 (cycle), with edge_attr dimension 3
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)
    edge_attr = torch.tensor(
        [[1.0, 0.1, 0.01], [2.0, 0.2, 0.02], [3.0, 0.3, 0.03]], dtype=torch.float
    )

    data = Data(
        x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
    )
    data.num_nodes = 3

    transform = SortNodes()
    data_s = transform(data)

    # After sorting, y should be [0, 1, 2]
    assert torch.equal(data_s.y, torch.tensor([0, 1, 2], dtype=torch.long))

    # x should be permuted according to sort_idx = [1, 2, 0]
    expected_x = torch.tensor([[2.0], [3.0], [1.0]])
    assert torch.allclose(data_s.x, expected_x)

    # edge_index should still have 3 edges
    assert data_s.edge_index.size(1) == 3
    assert data_s.edge_weight.numel() == 3
    assert data_s.edge_attr.size(0) == 3

    # Check that no edge_attr was lost: compare sets of rows
    orig_attrs = {tuple(row) for row in edge_attr.tolist()}
    new_attrs = {tuple(row) for row in data_s.edge_attr.tolist()}
    assert orig_attrs == new_attrs


def test_sortnodes_without_edge_attr():
    # Graph with 3 nodes: labels y out-of-order, no edge_attr
    y = torch.tensor([2, 0, 1], dtype=torch.long)
    x = torch.tensor([[1.0], [2.0], [3.0]])
    edge_index = torch.tensor([[0, 1, 2], [1, 2, 0]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0], dtype=torch.float)

    data = Data(x=x, y=y, edge_index=edge_index, edge_weight=edge_weight)
    data.num_nodes = 3

    transform = SortNodes()
    data_s = transform(data)

    # After sorting, y should be [0,1,2], x permuted accordingly
    assert torch.equal(data_s.y, torch.tensor([0, 1, 2], dtype=torch.long))
    expected_x = torch.tensor([[2.0], [3.0], [1.0]])
    assert torch.allclose(data_s.x, expected_x)

    # edge_index and edge_weight should be sorted accordingly and have length 3
    assert data_s.edge_index.size(1) == 3
    assert data_s.edge_weight.numel() == 3


def test_normalizeadj_with_edge_attr_already_has_self_loops():
    # Create a graph with 3 nodes, each having a self-loop, plus one undirected edge 0–1.
    N = 3
    # edge_index includes self-loops (0->0, 1->1, 2->2) and undirected edge 0<->1
    edge_index = torch.tensor(
        [
            [0, 1, 2, 0, 1],
            [0, 1, 2, 1, 0],
        ],
        dtype=torch.long,
    )
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 2.0, 2.0], dtype=torch.float)
    # edge_attr for each edge: shape [5,2]
    edge_attr = torch.tensor(
        [
            [0.1, 0.2],  # self-loop at 0
            [0.3, 0.4],  # self-loop at 1
            [0.5, 0.6],  # self-loop at 2
            [1.0, 10.0],  # edge 0->1
            [2.0, 20.0],  # edge 1->0
        ],
        dtype=torch.float,
    )

    x = torch.randn((N, 4))
    data = Data(
        x=x, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
    )
    data.num_nodes = N

    transform = NormalizeAdj(delta=0.5)
    data_t = transform(data)

    assert data_t.edge_weight.dim() == 1
    assert data_t.edge_attr.size(1) == 2

    # Check that diagonal entries in the normalized adjacency are positive
    dense_aw = to_dense_adj(data_t.edge_index, edge_attr=data_t.edge_weight).squeeze(0)
    diag = torch.diagonal(dense_aw)
    assert torch.all(diag > 0)


if __name__ == "__main__":
    pytest.main([__file__])
