import pytest
import torch
from torch_geometric.data import Batch, Data
from torch_geometric.utils import to_dense_adj

from tgp.data.loaders import PoolCollater, PoolDataLoader, PooledBatch
from tgp.data.transforms import NormalizeAdj, PreCoarsening
from tgp.poolers import NDPPooling, get_pooler


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


poolers = ["ndp", "kmis", "graclus", "lap"]


@pytest.mark.parametrize("pooler_name", poolers)
def test_nmf_precoarsening(sparse_batch_graph, pooler_name):
    PARAMS = {
        "scorer": "degree",
    }
    pooler = get_pooler(pooler_name, **PARAMS)

    data_batch = sparse_batch_graph
    assert data_batch.num_graphs == 3
    assert data_batch.num_nodes == 13

    pooling_out = pooler.precoarsening(
        edge_index=data_batch.edge_index,
        edge_weight=data_batch.edge_attr,
        batch=data_batch.batch,
        num_nodes=data_batch.num_nodes,
    )

    assert pooling_out.so.s.size(0) == 13
    assert pooling_out.batch is not None


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


if __name__ == "__main__":
    pytest.main([__file__])
