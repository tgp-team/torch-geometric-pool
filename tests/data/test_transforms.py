import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.utils import to_dense_adj

from tgp.data.transforms import NormalizeAdj, PreCoarsening, SortNodes
from tgp.src import SRCPooling


class _TrainablePooler(SRCPooling):
    def __init__(self):
        super().__init__()
        self.weight = torch.nn.Parameter(torch.tensor(1.0))


class _BadRunLenPooler(SRCPooling):
    def __init__(self):
        super().__init__()

    def multi_level_precoarsening(self, levels: int, **kwargs):
        # Intentionally violate contract: expected `levels` outputs.
        return []


class _DummyPooledOutput:
    def __init__(self, num_nodes: int):
        self._num_nodes = int(num_nodes)

    def as_data(self):
        if self._num_nodes <= 1:
            edge_index = torch.tensor([[0], [0]], dtype=torch.long)
            edge_weight = torch.ones(1, dtype=torch.float)
        else:
            src = torch.arange(self._num_nodes - 1, dtype=torch.long)
            dst = src + 1
            edge_index = torch.stack(
                [torch.cat([src, dst]), torch.cat([dst, src])], dim=0
            )
            edge_weight = torch.ones(edge_index.size(1), dtype=torch.float)
        return Data(
            edge_index=edge_index, edge_weight=edge_weight, num_nodes=self._num_nodes
        )


class _GoodRunPooler(SRCPooling):
    def __init__(self):
        super().__init__()
        self.calls = []

    def multi_level_precoarsening(self, levels: int, **kwargs):
        self.calls.append(levels)
        num_nodes = int(kwargs["num_nodes"])
        outputs = []
        for _ in range(levels):
            num_nodes = max(1, num_nodes - 1)
            outputs.append(_DummyPooledOutput(num_nodes))
        return outputs


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


def test_normalizeadj_without_edge_attr():
    data = Data(
        x=torch.randn(3, 2),
        edge_index=torch.tensor([[0, 1, 1, 2], [1, 0, 2, 1]], dtype=torch.long),
        edge_weight=torch.ones(4, dtype=torch.float),
        num_nodes=3,
    )

    out = NormalizeAdj(delta=0.5)(data)

    assert out.edge_index is not None
    assert out.edge_weight is not None
    assert out.edge_attr is None


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


def test_precoarsening_init_raises_on_empty_poolers():
    with pytest.raises(ValueError, match="non-empty pooler"):
        PreCoarsening(poolers=[])


def test_precoarsening_normalize_poolers_arg_variants():
    assert PreCoarsening._normalize_poolers_arg("ndp") == ["ndp"]

    cfg = {"pooler": "ndp", "k": 4}
    assert PreCoarsening._normalize_poolers_arg(cfg) == [cfg]

    tuple_cfg = ("ndp", None)
    assert PreCoarsening._normalize_poolers_arg(tuple_cfg) == [tuple_cfg]


def test_precoarsening_freeze_config_value_branches():
    list_frozen = PreCoarsening._freeze_config_value([1, {"a": 2}])
    assert isinstance(list_frozen, tuple)

    set_frozen = PreCoarsening._freeze_config_value({3, 1})
    assert isinstance(set_frozen, tuple)
    assert sorted(set_frozen) == [1, 3]

    class _Unhashable:
        __hash__ = None

    unhashable_frozen = PreCoarsening._freeze_config_value(_Unhashable())
    assert isinstance(unhashable_frozen, str)


def test_precoarsening_resolve_level_config_dict_and_errors():
    transform = PreCoarsening(poolers="ndp")

    pooler_from_dict, key_from_dict = transform._resolve_level_config_with_key(
        {"pooler": "ndp"}
    )
    assert isinstance(pooler_from_dict, SRCPooling)
    assert key_from_dict[0] == "config"

    with pytest.raises(ValueError, match="Tuple pooler configs must be"):
        transform._resolve_level_config_with_key(("ndp",))

    with pytest.raises(ValueError, match="must include a pooler name or instance"):
        transform._resolve_level_config_with_key((None, {}))

    with pytest.raises(ValueError, match="Cannot provide kwargs together"):
        transform._resolve_level_config_with_key((SRCPooling(), {"foo": 1}))

    with pytest.raises(TypeError, match="Pooler config must be"):
        transform._resolve_level_config_with_key(123)

    with pytest.raises(ValueError, match="must not be trainable"):
        transform._resolve_level_config_with_key(_TrainablePooler())


def test_precoarsening_resolve_level_config_instantiated_tuple():
    transform = PreCoarsening(poolers="ndp")
    pooler = SRCPooling()

    resolved_pooler, collapse_key = transform._resolve_level_config_with_key(
        (pooler, {})
    )

    assert resolved_pooler is pooler
    assert collapse_key == ("instance", id(pooler))


def test_precoarsening_init_raises_when_poolers_resolve_to_empty(monkeypatch):
    class _TruthyEmptyIterable:
        def __bool__(self):
            return True

        def __iter__(self):
            return iter(())

    monkeypatch.setattr(
        PreCoarsening,
        "_normalize_poolers_arg",
        staticmethod(lambda _: _TruthyEmptyIterable()),
    )

    with pytest.raises(ValueError, match="At least one pooling level is required"):
        PreCoarsening(poolers="ndp")


def test_precoarsening_collapse_consecutive_runs_empty():
    assert PreCoarsening._collapse_consecutive_runs([]) == []


def test_precoarsening_collapse_consecutive_runs_grouping():
    p1 = SRCPooling()
    p2 = SRCPooling()
    entries = [
        (p1, ("config", "ndp", ())),
        (p1, ("config", "ndp", ())),
        (p2, ("config", "kmis", ())),
        (p2, ("config", "kmis", ())),
        (p1, ("config", "ndp", ())),
    ]

    collapsed = PreCoarsening._collapse_consecutive_runs(entries)

    assert collapsed == [(p1, 2), (p2, 2), (p1, 1)]


def test_precoarsening_forward_raises_on_wrong_run_output_length():
    transform = PreCoarsening(poolers=_BadRunLenPooler())
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones(2, dtype=torch.float),
        num_nodes=2,
    )

    with pytest.raises(ValueError, match="returned 0 levels, expected 1"):
        transform(data)


def test_precoarsening_forward_success_attaches_levels():
    pooler = _GoodRunPooler()
    transform = PreCoarsening(poolers=[pooler, pooler])
    data = Data(
        edge_index=torch.tensor([[0, 1], [1, 0]], dtype=torch.long),
        edge_weight=torch.ones(2, dtype=torch.float),
        num_nodes=2,
    )

    out = transform(data)

    assert hasattr(out, "pooled_data")
    assert len(out.pooled_data) == 2
    assert all(isinstance(level, Data) for level in out.pooled_data)
    assert pooler.calls == [2]


if __name__ == "__main__":
    pytest.main([__file__])
