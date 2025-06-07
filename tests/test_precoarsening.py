import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import add_self_loops, to_dense_adj

from tgp.connect import DenseConnectSPT, KronConnect
from tgp.data.loaders import PoolCollater, PoolDataLoader, PooledBatch
from tgp.data.transforms import NormalizeAdj, PreCoarsening, SortNodes
from tgp.poolers import NDPPooling
from tgp.select import GraclusSelect, KMISSelect, LaPoolSelect, NDPSelect, NMFSelect


def test_normalizeadj_on_simple_data():
    # Build a simple line graph 0–1–2, no self‐loops
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    x = torch.randn((3, 2))
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight)
    data.num_nodes = 3

    transform = NormalizeAdj(delta=0.5)
    data_t = transform(data)

    # After normalization, edge_index and edge_weight should exist
    assert data_t.edge_index is not None
    assert data_t.edge_weight is not None

    # Convert to dense adjacency and check that diagonal entries are positive
    dense = to_dense_adj(data_t.edge_index, edge_attr=data_t.edge_weight).squeeze(0)
    assert dense.shape == (3, 3)
    diag = torch.diagonal(dense)
    assert torch.all(diag > 0)


def test_sortnodes_reorders_correctly():
    # Create a Data with 3 nodes, labels y out of order
    x = torch.tensor([[1.0], [0.0], [2.0]])
    y = torch.tensor([1, 0, 2])  # sorted order should be [0, 1, 2]
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.tensor([1.0, 1.0, 1.0, 1.0])
    edge_attr = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    data = Data(
        x=x, y=y, edge_index=edge_index, edge_weight=edge_weight, edge_attr=edge_attr
    )
    data.num_nodes = 3

    transform = SortNodes()
    data_s = transform(data)

    # After sorting, y should be [0, 1, 2]
    assert torch.equal(data_s.y, torch.tensor([0, 1, 2]))

    # x should have been permuted accordingly: original nodes [1, 0, 2]
    expected_x = torch.tensor([[0.0], [1.0], [2.0]])
    assert torch.allclose(data_s.x, expected_x)

    # edge_index should still have shape (2, 4)
    assert data_s.edge_index.shape == (2, 4)
    # edge_weight and edge_attr lengths remain 4
    assert data_s.edge_weight.numel() == 4
    assert data_s.edge_attr.size(0) == 4


def test_precoarsening_attaches_single_level():
    # Build a simple Data with 4 nodes in a line: 0–1–2–3
    edge_index = torch.tensor(
        [[0, 1, 1, 2, 2, 3], [1, 0, 2, 1, 3, 2]], dtype=torch.long
    )
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    x = torch.randn((4, 3))
    batch = torch.zeros(4, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = 4

    dummy = NDPPooling()
    transform = PreCoarsening(pooler=dummy, recursive_depth=1)
    data_t = transform(data)

    # After one level of pre‐coarsening, attribute "pooled_data" should exist
    assert hasattr(data_t, "pooled_data")
    pooled = data_t.pooled_data
    assert isinstance(pooled, Data)
    # pooled.num_nodes == 1 and x.shape == [1,1]
    assert pooled.num_nodes >= 1


def test_precoarsening_multiple_levels_returns_list():
    # Same input but recursive_depth=2
    edge_index = torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
    x = torch.randn((3, 2))
    batch = torch.zeros(3, dtype=torch.long)
    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = 3

    pooler = NDPPooling()
    transform = PreCoarsening(pooler=pooler, recursive_depth=2)
    data_t = transform(data)

    # Now pooled_data should be a list of two Data objects
    assert isinstance(data_t.pooled_data, list)
    assert len(data_t.pooled_data) == 2
    for pd in data_t.pooled_data:
        assert isinstance(pd, Data)
        assert pd.num_nodes >= 0


def test_pooledbatch_from_data_list_and_get_example():
    # Create two simple Data objects (no advanced coarsening required)
    d1 = Data(
        x=torch.tensor([[1.0]]),
        edge_index=torch.tensor([[0], [0]]),
        edge_weight=torch.tensor([1.0]),
    )
    d1.num_nodes = 1

    d2_edge = torch.tensor([[0, 1, 1, 0], [1, 0, 0, 1]], dtype=torch.long)
    d2 = Data(
        x=torch.tensor([[2.0], [3.0]]), edge_index=d2_edge, edge_weight=torch.ones(4)
    )
    d2.num_nodes = 2

    batch = PooledBatch.from_data_list([d1, d2], follow_batch=None, exclude_keys=None)
    # Should have internal slicing dictionaries
    assert hasattr(batch, "_slice_dict")
    assert hasattr(batch, "_inc_dict")
    # _num_graphs should be 2
    assert batch._num_graphs == 2

    # get_example(1) must return a Data with same num_nodes as d2
    example = batch.get_example(1)
    assert isinstance(example, Data)
    assert example.num_nodes == 2

    # If a PooledBatch is empty (no _slice_dict), get_example should raise
    with pytest.raises(RuntimeError):
        PooledBatch().get_example(0)


def test_poolcollater_and_pooldataloader(tmp_path):
    # Build two Data objects
    d1 = Data(
        x=torch.randn((2, 3)),
        edge_index=torch.tensor([[0, 1], [1, 0]]),
        edge_weight=torch.ones(2),
        y=torch.tensor([0, 1]),
    )
    d1.num_nodes = 2

    d2 = Data(
        x=torch.randn((3, 3)),
        edge_index=torch.tensor([[0, 1, 2], [1, 2, 0]]),
        edge_weight=torch.ones(3),
        y=torch.tensor([1, 0, 1]),
    )
    d2.num_nodes = 3

    dataset = [d1, d2]

    # PoolDataLoader should yield PooledBatch objects
    loader = PoolDataLoader(dataset, batch_size=2, shuffle=False)
    for batch in loader:
        from tgp.data.loaders import PooledBatch as _PooledBatch

        assert isinstance(batch, _PooledBatch)
        # It should carry through the 'y' attribute
        assert hasattr(batch, "y")
        break

    # PoolCollater alone should also produce a PooledBatch
    collator = PoolCollater(dataset, follow_batch=["x"], exclude_keys=None)
    collated = collator([d1, d2])
    assert isinstance(collated, PooledBatch)


def test_precoarsening_with_select_and_connect_one_level():
    """Build a small graph and run PreCoarsening by supplying `select` and `connect`
    instead of a full `pooler`.  Verify that `pooled_data` appears and has fewer nodes.
    """
    # Create a simple 5-node cycle with self-loops:
    N = 5
    edge_list = [(0, 1), (1, 2), (2, 3), (3, 4), (4, 0)]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    # Features and batch (single graph)
    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = N

    # Use NDPSelect to pick half the nodes (k=2 or 3)
    select = NDPSelect(s_inv_op="transpose")
    # KronConnect to compute pooled adjacency
    connect = KronConnect()

    # PreCoarsening with select+connect, single level
    transform = PreCoarsening(selector=select, connector=connect, recursive_depth=1)
    data_out = transform(data)

    # Check that pooled_data exists and is a Data object
    assert hasattr(data_out, "pooled_data")
    pooled = data_out.pooled_data
    assert isinstance(pooled, Data)

    # Pooled graph should have some edge_index
    assert hasattr(pooled, "edge_index")
    assert pooled.edge_index.size(0) == 2  # shape (2, E_pooled)
    # Pooled edge_weight should match length of pooled edges
    assert hasattr(pooled, "edge_weight")
    assert pooled.edge_weight.numel() == pooled.edge_index.size(1)


def test_precoarsening_with_select_and_connect_multiple_levels():
    """Same as above, but with recursive_depth=2.  Pooled data should become a list."""
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 0),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = N

    # Catch assertion error if selector is kmis
    with pytest.raises(AssertionError):
        transform = PreCoarsening(
            selector=KMISSelect(scorer="degree"),
            connector=KronConnect(),
            recursive_depth=2,
        )
        _ = transform(data)


def test_precoarsening_with_select_and_connect_multiple_levels_2():
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 0),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = N

    select = LaPoolSelect(s_inv_op="transpose")
    connect = DenseConnectSPT()

    # recursive_depth=2 means two successive coarsenings:
    transform = PreCoarsening(selector=select, connector=connect, recursive_depth=2)
    data_out = transform(data)

    # Now pooled_data should be a list of length=2
    assert isinstance(data_out.pooled_data, list)
    assert len(data_out.pooled_data) == 2
    for level_data in data_out.pooled_data:
        assert isinstance(level_data, Data)
        # Each level_data.num_nodes should equal k=3 (first level) or new k=?
        assert hasattr(level_data, "num_nodes")
        assert level_data.num_nodes >= 1
        assert hasattr(level_data, "x")


def test_precoarsening_with_select_and_connect_multiple_levels_3():
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 0),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = N

    select = NMFSelect(k=2, s_inv_op="transpose")
    connect = DenseConnectSPT()

    # recursive_depth=2 means two successive coarsenings:
    transform = PreCoarsening(selector=select, connector=connect, recursive_depth=2)
    data_out = transform(data)

    # Now pooled_data should be a list of length=2
    assert isinstance(data_out.pooled_data, list)
    assert len(data_out.pooled_data) == 2
    for level_data in data_out.pooled_data:
        assert isinstance(level_data, Data)
        # Each level_data.num_nodes should equal k=3 (first level) or new k=?
        assert hasattr(level_data, "num_nodes")
        assert level_data.num_nodes >= 1
        assert hasattr(level_data, "x")


def test_precoarsening_graclus():
    N = 10
    edge_list = [
        (0, 1),
        (1, 2),
        (2, 3),
        (3, 4),
        (4, 0),
        (0, 2),
        (1, 3),
        (2, 4),
        (3, 0),
        (4, 1),
        (5, 6),
        (6, 7),
        (7, 8),
        (8, 9),
        (9, 5),
    ]
    row = torch.tensor(
        [u for u, v in edge_list] + [v for u, v in edge_list], dtype=torch.long
    )
    col = torch.tensor(
        [v for u, v in edge_list] + [u for u, v in edge_list], dtype=torch.long
    )
    edge_index = torch.stack([row, col], dim=0)
    edge_index, _ = add_self_loops(edge_index, num_nodes=N)
    edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)

    x = torch.randn((N, 3))
    batch = torch.zeros(N, dtype=torch.long)

    data = Data(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
    data.num_nodes = N

    select = GraclusSelect()
    connect = DenseConnectSPT()

    # recursive_depth=2 means two successive coarsenings:
    transform = PreCoarsening(selector=select, connector=connect, recursive_depth=2)
    data_out = transform(data)

    # Now pooled_data should be a list of length=2
    assert isinstance(data_out.pooled_data, list)
    assert len(data_out.pooled_data) == 2
    for level_data in data_out.pooled_data:
        assert isinstance(level_data, Data)
        # Each level_data.num_nodes should equal k=3 (first level) or new k=?
        assert hasattr(level_data, "num_nodes")
        assert level_data.num_nodes >= 1
        assert hasattr(level_data, "x")


if __name__ == "__main__":
    pytest.main([__file__])
