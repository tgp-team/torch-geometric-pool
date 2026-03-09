import builtins
import importlib
import importlib.util
from pathlib import Path
from types import SimpleNamespace

import pytest
import torch

from tgp.reduce import AggrReduce, BaseReduce
from tgp.reduce.get_aggr import resolve_reduce_op
from tgp.select import SelectOutput

get_aggr_mod = importlib.import_module("tgp.reduce.get_aggr")
aggr_reduce_mod = importlib.import_module("tgp.reduce.aggr_reduce")


def _make_sparse_select_output(num_nodes=6, num_supernodes=3):
    """Minimal sparse SelectOutput for testing."""
    # node_index (row): which nodes contribute; cluster_index (col): to which supernode
    node_index = torch.tensor([0, 1, 2, 3, 4, 5], dtype=torch.long)
    cluster_index = torch.tensor([0, 0, 1, 1, 2, 2], dtype=torch.long)
    indices = torch.stack([node_index, cluster_index], dim=0)
    values = torch.ones(6, dtype=torch.float)
    s = torch.sparse_coo_tensor(
        indices, values, size=(num_nodes, num_supernodes)
    ).coalesce()
    return SelectOutput(s=s)


class _FakeAggr:
    def __call__(self, src, index, dim_size, dim):
        assert dim == 0
        out = src.new_zeros((dim_size, src.size(-1)))
        out.index_add_(0, index, src)
        return out

    def __repr__(self):
        return "FakeAggr()"


@pytest.mark.parametrize("reduce_op", ["sum", "mean"])
def test_aggr_reduce_vs_base_reduce_sparse(reduce_op):
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    if reduce_op == "sum":
        aggr_module = aggr.SumAggregation()
    else:
        aggr_module = aggr.MeanAggregation()

    so = _make_sparse_select_output()
    x = torch.randn(6, 4)  # 6 nodes, 4 features
    batch = None

    aggr_reduce = AggrReduce(aggr_module)
    x_aggr, batch_aggr = aggr_reduce(x, so, batch=batch)

    assert x_aggr.shape == (3, 4)
    if reduce_op == "sum":
        base = BaseReduce()
        x_base, batch_base = base(x, so, batch=batch)
        assert torch.allclose(x_base, x_aggr)
        assert batch_base == batch_aggr


def test_aggr_reduce_sparse_with_weights():
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")
    node_index = torch.tensor([0, 1, 2, 3], dtype=torch.long)
    cluster_index = torch.tensor([0, 0, 1, 1], dtype=torch.long)
    values = torch.tensor([1.0, 1.0, 1.0, 1.0], dtype=torch.float)
    indices = torch.stack([node_index, cluster_index], dim=0)
    s = torch.sparse_coo_tensor(indices, values, size=(4, 2)).coalesce()
    so = SelectOutput(s=s)
    x = torch.ones(4, 2)
    base = BaseReduce()
    aggr_reduce = AggrReduce(aggr.SumAggregation())
    x_base, _ = base(x, so, batch=None)
    x_aggr, _ = aggr_reduce(x, so, batch=None)
    assert torch.allclose(x_base, x_aggr)


def test_aggr_reduce_rejects_dense_select_output():
    """AggrReduce only supports sparse SelectOutput assignments."""
    try:
        from torch_geometric.nn import aggr
    except ImportError:
        pytest.skip("PyG aggr not available")

    B, N, K = 2, 3, 2
    s = torch.zeros(B, N, K)
    s[0, 0, 0] = 1.0
    s[0, 1, 1] = 1.0
    s[1, 0, 0] = 1.0
    s[1, 2, 1] = 1.0
    so = SelectOutput(s=s)
    x = torch.randn(B, N, 2)

    reducer = AggrReduce(aggr.SumAggregation())
    with pytest.raises(
        ValueError,
        match="AggrReduce supports only sparse SelectOutput assignments",
    ):
        reducer(x, so=so, batch=None)


def test_aggr_reduce_init_raises_when_pyg_aggr_is_unavailable(monkeypatch):
    monkeypatch.setattr(aggr_reduce_mod, "has_pyg_aggregation", lambda: False)

    with pytest.raises(ImportError, match="requires torch_geometric.nn.aggr"):
        aggr_reduce_mod.AggrReduce(_FakeAggr())


def test_aggr_reduce_init_rejects_non_pyg_aggregation(monkeypatch):
    monkeypatch.setattr(aggr_reduce_mod, "has_pyg_aggregation", lambda: True)
    monkeypatch.setattr(aggr_reduce_mod, "is_pyg_aggregation", lambda _: False)

    with pytest.raises(TypeError, match="aggr must be a PyG Aggregation"):
        aggr_reduce_mod.AggrReduce(_FakeAggr())


def test_aggr_reduce_uses_select_output_batch_when_batch_is_none(monkeypatch):
    monkeypatch.setattr(aggr_reduce_mod, "has_pyg_aggregation", lambda: True)
    monkeypatch.setattr(aggr_reduce_mod, "is_pyg_aggregation", lambda _: True)

    reducer = aggr_reduce_mod.AggrReduce(_FakeAggr())
    x = torch.tensor([[1.0], [2.0], [3.0], [4.0]])
    indices = torch.tensor([[0, 1, 2, 3], [0, 0, 1, 1]], dtype=torch.long)
    s = torch.sparse_coo_tensor(indices, torch.ones(4), size=(4, 2)).coalesce()
    so = SelectOutput(s=s, batch=torch.tensor([0, 0, 1, 1], dtype=torch.long))

    x_pool, batch_pool = reducer(x, so=so, batch=None)

    assert x_pool.shape == (2, 1)
    assert torch.equal(batch_pool, torch.tensor([0, 1], dtype=torch.long))


def test_aggr_reduce_readout_rejects_invalid_ndim(monkeypatch):
    monkeypatch.setattr(aggr_reduce_mod, "has_pyg_aggregation", lambda: True)
    monkeypatch.setattr(aggr_reduce_mod, "is_pyg_aggregation", lambda _: True)
    reducer = aggr_reduce_mod.AggrReduce(_FakeAggr())

    with pytest.raises(ValueError, match="expects x to be 2D \\[N, F\\] or 3D"):
        reducer(torch.randn(1, 2, 3, 4), so=None)


def test_aggr_reduce_repr(monkeypatch):
    monkeypatch.setattr(aggr_reduce_mod, "has_pyg_aggregation", lambda: True)
    monkeypatch.setattr(aggr_reduce_mod, "is_pyg_aggregation", lambda _: True)
    reducer = aggr_reduce_mod.AggrReduce(_FakeAggr())

    assert repr(reducer) == "AggrReduce(aggr=FakeAggr())"


def test_resolve_reduce_op_rejects_invalid_type():
    with pytest.raises(TypeError, match="string alias or a PyG Aggregation instance"):
        resolve_reduce_op(123)


def test_resolve_reduce_op_string_path_calls_get_aggr(monkeypatch):
    sentinel = object()

    def fake_get_aggr(alias, **kwargs):
        assert alias == "sum"
        assert kwargs == {"foo": "bar"}
        return sentinel

    monkeypatch.setattr(get_aggr_mod, "get_aggr", fake_get_aggr)
    out = resolve_reduce_op("sum", foo="bar")
    assert out is sentinel


def test_resolve_reduce_op_returns_existing_aggregation_instance(monkeypatch):
    class DummyAggregation:
        pass

    instance = DummyAggregation()
    monkeypatch.setattr(get_aggr_mod, "_PyGAggregation", DummyAggregation)

    out = resolve_reduce_op(instance)
    assert out is instance


def test_get_aggr_rejects_unknown_alias():
    with pytest.raises(ValueError, match="Unknown aggregator alias"):
        get_aggr_mod.get_aggr("definitely_unknown_alias")


def test_get_aggr_raises_import_error_when_pyg_module_missing(monkeypatch):
    monkeypatch.setattr(get_aggr_mod, "_aggr_module", None)

    with pytest.raises(ImportError, match="requires torch_geometric.nn.aggr"):
        get_aggr_mod.get_aggr("sum")


def test_get_aggr_raises_for_missing_class_in_pyg_module(monkeypatch):
    monkeypatch.setattr(get_aggr_mod, "_aggr_module", SimpleNamespace())
    monkeypatch.setitem(
        get_aggr_mod._AGGR_ALIASES,
        "missing_class_alias",
        ("DefinitelyMissingAggregationClass", {}),
    )

    with pytest.raises(ValueError, match="not found in torch_geometric.nn.aggr"):
        get_aggr_mod.get_aggr("missing_class_alias")


def test_get_aggr_signature_introspection_fallback(monkeypatch):
    class FakeAggregation:
        def __init__(self, **kwargs):
            self.kwargs = kwargs

    fake_aggr_module = SimpleNamespace(FakeAggregation=FakeAggregation)
    monkeypatch.setattr(get_aggr_mod, "_aggr_module", fake_aggr_module)
    monkeypatch.setitem(get_aggr_mod._AGGR_ALIASES, "fake", ("FakeAggregation", {}))

    def _raise_signature_error(*args, **kwargs):
        raise RuntimeError("signature unavailable")

    monkeypatch.setattr(get_aggr_mod.inspect, "signature", _raise_signature_error)

    out = get_aggr_mod.get_aggr("fake", alpha=1, beta=2)
    assert isinstance(out, FakeAggregation)
    assert out.kwargs == {"alpha": 1, "beta": 2}


def test_get_aggr_sets_default_out_channels_for_lstm(monkeypatch):
    class FakeLSTM:
        def __init__(self, in_channels=None, out_channels=None):
            self.in_channels = in_channels
            self.out_channels = out_channels

    fake_aggr_module = SimpleNamespace(LSTMAggregation=FakeLSTM)
    monkeypatch.setattr(get_aggr_mod, "_aggr_module", fake_aggr_module)

    out = get_aggr_mod.get_aggr("lstm", in_channels=8)
    assert isinstance(out, FakeLSTM)
    assert out.in_channels == 8
    assert out.out_channels == 8


def test_get_aggr_filters_kwargs_using_signature(monkeypatch):
    class FakeSigAggregation:
        def __init__(self, keep=None):
            self.keep = keep

    fake_aggr_module = SimpleNamespace(FakeSigAggregation=FakeSigAggregation)
    monkeypatch.setattr(get_aggr_mod, "_aggr_module", fake_aggr_module)
    monkeypatch.setitem(
        get_aggr_mod._AGGR_ALIASES,
        "fake_sig",
        ("FakeSigAggregation", {}),
    )

    out = get_aggr_mod.get_aggr("fake_sig", keep=7, drop_me=99)
    assert isinstance(out, FakeSigAggregation)
    assert out.keep == 7


def test_get_aggr_import_fallback_executes_without_pyg(monkeypatch):
    module_path = Path(__file__).resolve().parents[2] / "tgp" / "reduce" / "get_aggr.py"
    real_import = builtins.__import__

    def fake_import(name, globals=None, locals=None, fromlist=(), level=0):
        if name in {"torch_geometric.nn", "torch_geometric.nn.aggr"}:
            raise ImportError("forced import failure for coverage")
        return real_import(name, globals, locals, fromlist, level)

    monkeypatch.setattr(builtins, "__import__", fake_import)

    spec = importlib.util.spec_from_file_location(
        "tgp.reduce._get_aggr_import_fallback_test",
        module_path,
    )
    module = importlib.util.module_from_spec(spec)
    assert spec is not None and spec.loader is not None
    spec.loader.exec_module(module)

    assert module._aggr_module is None
    assert module._PyGAggregation is None


if __name__ == "__main__":
    pytest.main([__file__])
