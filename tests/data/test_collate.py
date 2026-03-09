import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.typing import torch_sparse

import tgp.data.collate as tgp_collate_module
from tgp.data.loaders import PooledBatch
from tgp.select import SelectOutput


def _graph(num_nodes: int) -> Data:
    x = torch.arange(num_nodes, dtype=torch.float32).view(-1, 1)
    edge_index = torch.stack([torch.arange(num_nodes), torch.arange(num_nodes)], dim=0)
    data = Data(x=x, edge_index=edge_index)
    data.num_nodes = num_nodes
    return data


def _assert_sparse_equal(lhs: torch.Tensor, rhs: torch.Tensor) -> None:
    lhs = lhs.coalesce()
    rhs = rhs.coalesce()
    assert lhs.size() == rhs.size()
    assert torch.equal(lhs.indices(), rhs.indices())
    assert torch.allclose(lhs.values(), rhs.values())


def _assert_nested_equal(lhs, rhs) -> None:
    assert type(lhs) is type(rhs)

    if isinstance(lhs, torch.Tensor):
        if lhs.is_sparse:
            _assert_sparse_equal(lhs, rhs)
        else:
            assert torch.equal(lhs, rhs)
        return

    if isinstance(lhs, dict):
        assert lhs.keys() == rhs.keys()
        for key in lhs:
            _assert_nested_equal(lhs[key], rhs[key])
        return

    if isinstance(lhs, list):
        assert len(lhs) == len(rhs)
        for left, right in zip(lhs, rhs):
            _assert_nested_equal(left, right)
        return

    assert lhs == rhs


class _FakeSelectOutput:
    def __init__(self, s, s_inv=None, batch=None, **extra_args):
        self.s = s
        self.s_inv = s_inv
        self.batch = batch
        self._extra_args = set(extra_args.keys())
        for key, value in extra_args.items():
            setattr(self, key, value)


def test_pooledbatch_round_trip_nested_dense_select_output():
    so1 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        theta=torch.tensor([1.0, 2.0]),
        theta_list=[torch.tensor([0.1, 0.2])],
    )
    so2 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        batch=torch.zeros(3, dtype=torch.long),
        theta=torch.tensor([3.0, 4.0, 5.0]),
        theta_list=[torch.tensor([0.3, 0.4, 0.5])],
    )

    d1 = _graph(2)
    d2 = _graph(3)
    d1.pooled_data = [Data(so=so1, num_nodes=so1.num_supernodes)]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    batch = PooledBatch.from_data_list([d1, d2])
    collated_so = batch.pooled_data[0].so

    assert torch.equal(collated_so.s, torch.cat([so1.s, so2.s], dim=0))
    assert torch.equal(collated_so.batch, torch.tensor([0, 0, 1, 1, 1]))

    restored_d1 = batch.get_example(0)
    restored_d2 = batch.get_example(1)

    assert torch.equal(restored_d1.pooled_data[0].so.s, so1.s)
    assert torch.equal(restored_d2.pooled_data[0].so.s, so2.s)
    assert torch.equal(restored_d1.pooled_data[0].so.batch, so1.batch)
    assert torch.equal(restored_d2.pooled_data[0].so.batch, so2.batch)
    assert torch.equal(restored_d1.pooled_data[0].so.theta, so1.theta)
    assert torch.equal(restored_d2.pooled_data[0].so.theta, so2.theta)
    assert torch.equal(restored_d1.pooled_data[0].so.theta_list[0], so1.theta_list[0])
    assert torch.equal(restored_d2.pooled_data[0].so.theta_list[0], so2.theta_list[0])


def test_pooledbatch_round_trip_nested_sparse_select_output():
    s1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    ).coalesce()
    s2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 2], [1, 0]]),
        values=torch.tensor([0.5, 1.5]),
        size=(3, 2),
    ).coalesce()

    so1 = SelectOutput(
        s=s1,
        batch=torch.zeros(2, dtype=torch.long),
        score=torch.tensor([0.1, 0.2]),
    )
    so2 = SelectOutput(
        s=s2,
        batch=torch.zeros(3, dtype=torch.long),
        score=torch.tensor([0.3, 0.4, 0.5]),
    )

    d1 = _graph(2)
    d2 = _graph(3)
    d1.pooled_data = [Data(so=so1, num_nodes=so1.num_supernodes)]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    batch = PooledBatch.from_data_list([d1, d2])

    restored_d1 = batch.get_example(0)
    restored_d2 = batch.get_example(1)

    restored_so1 = restored_d1.pooled_data[0].so
    restored_so2 = restored_d2.pooled_data[0].so

    _assert_sparse_equal(restored_so1.s, so1.s)
    _assert_sparse_equal(restored_so2.s, so2.s)
    _assert_sparse_equal(restored_so1.s_inv, so1.s_inv)
    _assert_sparse_equal(restored_so2.s_inv, so2.s_inv)
    assert torch.equal(restored_so1.batch, so1.batch)
    assert torch.equal(restored_so2.batch, so2.batch)
    assert torch.equal(restored_so1.score, so1.score)
    assert torch.equal(restored_so2.score, so2.score)


def test_pooledbatch_round_trip_multi_level_with_nested_extra_args():
    so1_level0 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.3, 0.7], [0.0, 1.0]]),
        batch=torch.zeros(3, dtype=torch.long),
        score=torch.tensor([0.1, 0.2, 0.3]),
        meta={
            "alpha": torch.tensor([1.0, 2.0, 3.0]),
            "parts": [torch.tensor([1.0, 0.0, 1.0]), torch.tensor([0.5, 0.5, 0.0])],
        },
    )
    so1_level1 = SelectOutput(
        s=torch.tensor([[1.0], [1.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        theta_list=[torch.tensor([0.2, 0.8]), torch.tensor([0.4, 0.6])],
    )

    so2_level0 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        score=torch.tensor([0.4, 0.5]),
        meta={
            "alpha": torch.tensor([4.0, 5.0]),
            "parts": [torch.tensor([1.0, 1.0]), torch.tensor([0.2, 0.8])],
        },
    )
    so2_level1 = SelectOutput(
        s=torch.tensor([[1.0], [1.0]]),
        batch=torch.zeros(2, dtype=torch.long),
        theta_list=[torch.tensor([0.9, 0.1]), torch.tensor([0.7, 0.3])],
    )

    d1 = _graph(3)
    d2 = _graph(2)
    d1.pooled_data = [
        Data(so=so1_level0, num_nodes=so1_level0.num_supernodes),
        Data(so=so1_level1, num_nodes=so1_level1.num_supernodes),
    ]
    d2.pooled_data = [
        Data(so=so2_level0, num_nodes=so2_level0.num_supernodes),
        Data(so=so2_level1, num_nodes=so2_level1.num_supernodes),
    ]

    batch = PooledBatch.from_data_list([d1, d2])
    assert len(batch.pooled_data) == 2

    restored = [batch.get_example(0), batch.get_example(1)]
    original = [[so1_level0, so1_level1], [so2_level0, so2_level1]]

    for graph_idx, restored_data in enumerate(restored):
        for level_idx, expected_so in enumerate(original[graph_idx]):
            restored_so = restored_data.pooled_data[level_idx].so
            assert torch.equal(restored_so.s, expected_so.s)
            assert torch.equal(restored_so.s_inv, expected_so.s_inv)
            assert torch.equal(restored_so.batch, expected_so.batch)

            for extra_key in expected_so._extra_args:
                _assert_nested_equal(
                    getattr(restored_so, extra_key),
                    getattr(expected_so, extra_key),
                )


def test_pooledbatch_select_output_batch_is_normalized_on_get_example():
    so1 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch=torch.tensor([3, 3], dtype=torch.long),
    )
    so2 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.5, 0.5], [0.0, 1.0]]),
        batch=torch.tensor([7, 7, 8], dtype=torch.long),
    )

    d1 = _graph(2)
    d2 = _graph(3)
    d1.pooled_data = [Data(so=so1, num_nodes=so1.num_supernodes)]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    batch = PooledBatch.from_data_list([d1, d2])
    restored_so1 = batch.get_example(0).pooled_data[0].so
    restored_so2 = batch.get_example(1).pooled_data[0].so

    assert torch.equal(restored_so1.batch, torch.tensor([0, 0], dtype=torch.long))
    assert torch.equal(restored_so2.batch, torch.tensor([0, 0, 1], dtype=torch.long))
    assert torch.equal(restored_so1.s, so1.s)
    assert torch.equal(restored_so2.s, so2.s)


def test_pooledbatch_raises_on_inconsistent_pooled_data_lengths():
    so1 = SelectOutput(s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))
    so1_next = SelectOutput(s=torch.tensor([[1.0], [1.0]]))
    so2 = SelectOutput(s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]))

    d1 = _graph(2)
    d2 = _graph(2)
    d1.pooled_data = [
        Data(so=so1, num_nodes=so1.num_supernodes),
        Data(so=so1_next, num_nodes=so1_next.num_supernodes),
    ]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    with pytest.raises(RuntimeError, match="equal size"):
        PooledBatch.from_data_list([d1, d2])


def test_pooledbatch_raises_on_mismatched_select_output_extra_args():
    so1 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        score=torch.tensor([0.1, 0.2]),
    )
    so2 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        mass=torch.tensor([1.0, 1.0]),
    )

    d1 = _graph(2)
    d2 = _graph(2)
    d1.pooled_data = [Data(so=so1, num_nodes=so1.num_supernodes)]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    with pytest.raises(ValueError, match="different extra attributes"):
        PooledBatch.from_data_list([d1, d2])


def test_collate_guard_raises_when_pyg_collate_hooks_missing(monkeypatch):
    monkeypatch.setattr(tgp_collate_module, "_HAS_PYG_COLLATE_HOOKS", False)

    with pytest.raises(RuntimeError, match="collate_fn_map"):
        tgp_collate_module.collate(Data, [_graph(2), _graph(2)])


def test_separate_guard_raises_when_pyg_separate_hooks_missing(monkeypatch):
    batch, slice_dict, inc_dict = tgp_collate_module.collate(
        Data, [_graph(2), _graph(2)]
    )
    monkeypatch.setattr(tgp_collate_module, "_HAS_PYG_SEPARATE_HOOKS", False)

    with pytest.raises(RuntimeError, match="separate_fn_map"):
        tgp_collate_module.separate(Data, batch, 0, slice_dict, inc_dict)


def test_hook_collate_list_handles_empty_lists():
    data_list = [_graph(1), _graph(1)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    value, slices, incs = tgp_collate_module._hook_collate_list(
        key="pooled_data",
        values=[[], []],
        data_list=data_list,
        stores=stores,
        increment=True,
        collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
    )

    assert value == [[], []]
    assert torch.equal(slices, torch.tensor([0, 1, 2]))
    assert incs is None


def test_hook_separate_list_falls_back_to_direct_index():
    batch = _graph(1)
    result = tgp_collate_module._hook_separate_list(
        key="pooled_data",
        values=["left", "right"],
        idx=1,
        slices=0,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )
    assert result == "right"


def test_pooledbatch_round_trip_dense_3d_select_output():
    so1 = SelectOutput(s=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]]))
    so2 = SelectOutput(s=torch.tensor([[[0.5, 0.5], [1.0, 0.0]]]))

    d1 = _graph(2)
    d2 = _graph(2)
    d1.pooled_data = [Data(so=so1, num_nodes=so1.num_supernodes)]
    d2.pooled_data = [Data(so=so2, num_nodes=so2.num_supernodes)]

    batch = PooledBatch.from_data_list([d1, d2])
    restored1 = batch.get_example(0).pooled_data[0].so
    restored2 = batch.get_example(1).pooled_data[0].so

    assert torch.equal(restored1.s, so1.s)
    assert torch.equal(restored2.s, so2.s)
    assert torch.equal(restored1.s_inv, so1.s_inv)
    assert torch.equal(restored2.s_inv, so2.s_inv)


def test_hook_collate_select_output_raises_on_mixed_batch_presence():
    so1 = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch=torch.zeros(2, dtype=torch.long),
    )
    so2 = SelectOutput(s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]), batch=None)
    data_list = [_graph(2), _graph(2)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    with pytest.raises(ValueError, match="batch"):
        tgp_collate_module._hook_collate_select_output(
            key="so",
            values=[so1, so2],
            data_list=data_list,
            stores=stores,
            increment=True,
            collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
        )


def test_hook_collate_select_output_raises_on_non_tensor_dense_input():
    so1 = _FakeSelectOutput(s={"bad": True})
    so2 = _FakeSelectOutput(s={"bad": False})
    data_list = [_graph(2), _graph(2)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    with pytest.raises(TypeError, match="Tensor or SparseTensor"):
        tgp_collate_module._hook_collate_select_output(
            key="so",
            values=[so1, so2],
            data_list=data_list,
            stores=stores,
            increment=True,
            collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
        )


def test_hook_collate_select_output_raises_on_invalid_dense_rank():
    so1 = _FakeSelectOutput(s=torch.tensor(1.0))
    so2 = _FakeSelectOutput(s=torch.tensor(2.0))
    data_list = [_graph(2), _graph(2)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    with pytest.raises(ValueError, match="2D \\[N, K\\] or 3D"):
        tgp_collate_module._hook_collate_select_output(
            key="so",
            values=[so1, so2],
            data_list=data_list,
            stores=stores,
            increment=True,
            collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
        )


def test_hook_collate_select_output_handles_torch_sparse_tensors(monkeypatch):
    sparse1 = torch_sparse.SparseTensor.from_dense(torch.eye(2))
    sparse2 = torch_sparse.SparseTensor.from_dense(
        torch.tensor([[1.0, 0.0], [0.2, 0.8], [0.0, 1.0]])
    )
    so1 = _FakeSelectOutput(s=sparse1, s_inv=sparse1.t())
    so2 = _FakeSelectOutput(s=sparse2, s_inv=sparse2.t())
    data_list = [_graph(2), _graph(3)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    class _DummySelectOutput:
        def __init__(self, s, s_inv, batch=None, **extra_args):
            self.s = s
            self.s_inv = s_inv
            self.batch = batch
            self._extra_args = set(extra_args.keys())
            for key, value in extra_args.items():
                setattr(self, key, value)

    monkeypatch.setattr(tgp_collate_module, "SelectOutput", _DummySelectOutput)
    out, slices, incs = tgp_collate_module._hook_collate_select_output(
        key="so",
        values=[so1, so2],
        data_list=data_list,
        stores=stores,
        increment=True,
        collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
    )

    assert isinstance(out, _DummySelectOutput)
    assert isinstance(out.s, torch_sparse.SparseTensor)
    assert slices["s"].shape[1] == 2
    assert incs is None


def test_hook_collate_select_output_sparse_without_s_inv():
    s1 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 1], [0, 1]]),
        values=torch.tensor([1.0, 1.0]),
        size=(2, 2),
    ).coalesce()
    s2 = torch.sparse_coo_tensor(
        indices=torch.tensor([[0, 2], [1, 0]]),
        values=torch.tensor([0.5, 1.5]),
        size=(3, 2),
    ).coalesce()
    so1 = _FakeSelectOutput(s=s1, s_inv=None)
    so2 = _FakeSelectOutput(s=s2, s_inv=None)
    data_list = [_graph(2), _graph(3)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    out, slices, incs = tgp_collate_module._hook_collate_select_output(
        key="so",
        values=[so1, so2],
        data_list=data_list,
        stores=stores,
        increment=True,
        collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
    )

    assert out.s.is_sparse
    assert out.s_inv is not None
    assert slices["s"].shape[1] == 2
    assert incs is None


def test_hook_collate_select_output_dense_without_s_inv_and_empty_batch_chunk():
    so1 = _FakeSelectOutput(
        s=torch.empty((0, 2, 2), dtype=torch.float32),
        s_inv=None,
        batch=torch.empty((0,), dtype=torch.long),
    )
    so2 = _FakeSelectOutput(
        s=torch.tensor([[[1.0, 0.0], [0.0, 1.0]]], dtype=torch.float32),
        s_inv=None,
        batch=torch.tensor([0], dtype=torch.long),
    )
    data_list = [_graph(1), _graph(1)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    out, slices, incs = tgp_collate_module._hook_collate_select_output(
        key="so",
        values=[so1, so2],
        data_list=data_list,
        stores=stores,
        increment=True,
        collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
    )

    assert out.s.shape == (1, 2, 2)
    assert torch.equal(out.batch, torch.tensor([0], dtype=torch.long))
    assert torch.equal(slices["batch"], torch.tensor([0, 0, 1]))
    assert incs is None


def test_hook_collate_select_output_dense_2d_without_s_inv():
    so1 = _FakeSelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        s_inv=None,
    )
    so2 = _FakeSelectOutput(
        s=torch.tensor([[0.5, 0.5], [1.0, 0.0]], dtype=torch.float32),
        s_inv=None,
    )
    data_list = [_graph(2), _graph(2)]
    stores = [data_list[0].stores[0], data_list[1].stores[0]]

    out, slices, incs = tgp_collate_module._hook_collate_select_output(
        key="so",
        values=[so1, so2],
        data_list=data_list,
        stores=stores,
        increment=True,
        collate_fn_map=tgp_collate_module._TGP_COLLATE_FN_MAP,
    )

    assert torch.equal(
        out.s,
        torch.tensor(
            [[1.0, 0.0], [0.0, 1.0], [0.5, 0.5], [1.0, 0.0]], dtype=torch.float32
        ),
    )
    assert out.s_inv is not None
    assert torch.equal(slices["s"], torch.tensor([0, 2, 4]))
    assert incs is None


def test_hook_separate_select_output_handles_sparse_tensor_branch(monkeypatch):
    sparse = torch_sparse.SparseTensor.from_dense(torch.eye(2))
    values = _FakeSelectOutput(s=sparse, s_inv=sparse.t())
    slices = {"s": torch.tensor([[0, 0], [2, 2]])}
    batch = _graph(2)

    class _DummySelectOutput:
        def __init__(self, s, s_inv, batch=None, **extra_args):
            self.s = s
            self.s_inv = s_inv
            self.batch = batch
            self._extra_args = set(extra_args.keys())
            for key, value in extra_args.items():
                setattr(self, key, value)

    monkeypatch.setattr(tgp_collate_module, "SelectOutput", _DummySelectOutput)
    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert isinstance(out, _DummySelectOutput)
    assert isinstance(out.s, torch_sparse.SparseTensor)
    assert isinstance(out.s_inv, torch_sparse.SparseTensor)


def test_hook_separate_select_output_sparse_tensor_without_s_inv(monkeypatch):
    sparse = torch_sparse.SparseTensor.from_dense(torch.eye(2))
    values = _FakeSelectOutput(s=sparse, s_inv=None)
    slices = {"s": torch.tensor([[0, 0], [2, 2]])}
    batch = _graph(2)

    class _DummySelectOutput:
        def __init__(self, s, s_inv, batch=None, **extra_args):
            self.s = s
            self.s_inv = s_inv
            self.batch = batch
            self._extra_args = set(extra_args.keys())
            for key, value in extra_args.items():
                setattr(self, key, value)

    monkeypatch.setattr(tgp_collate_module, "SelectOutput", _DummySelectOutput)
    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert isinstance(out, _DummySelectOutput)
    assert isinstance(out.s, torch_sparse.SparseTensor)
    assert out.s_inv is None


def test_hook_separate_select_output_torch_sparse_without_s_inv():
    values = _FakeSelectOutput(
        s=torch.sparse_coo_tensor(
            indices=torch.tensor([[0, 1], [0, 1]]),
            values=torch.tensor([1.0, 1.0]),
            size=(2, 2),
        ).coalesce(),
        s_inv=None,
    )
    slices = {"s": torch.tensor([[0, 0], [2, 2]])}
    batch = _graph(2)

    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert out.s.is_sparse
    assert out.s_inv is not None


def test_hook_separate_select_output_dense_2d_without_s_inv():
    values = _FakeSelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]], dtype=torch.float32),
        s_inv=None,
    )
    slices = {"s": torch.tensor([0, 2])}
    batch = _graph(2)

    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert torch.equal(out.s, values.s)
    assert out.s_inv is not None


def test_hook_separate_select_output_dense_3d_without_s_inv():
    values = _FakeSelectOutput(
        s=torch.tensor(
            [[[1.0, 0.0], [0.0, 1.0]], [[0.2, 0.8], [1.0, 0.0]]], dtype=torch.float32
        ),
        s_inv=None,
    )
    slices = {"s": torch.tensor([0, 1, 2])}
    batch = _graph(2)

    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert torch.equal(out.s, values.s[:1])
    assert out.s_inv is not None


def test_hook_separate_select_output_handles_2d_slices_for_dense():
    values = SelectOutput(
        s=torch.tensor([[1.0, 0.0], [0.0, 1.0]]),
        batch=torch.tensor([5, 5], dtype=torch.long),
    )
    slices = {
        "s": torch.tensor([[0, 0], [2, 2]]),
        "batch": torch.tensor([0, 2]),
    }
    batch = _graph(2)

    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert torch.equal(out.s, values.s)
    assert torch.equal(out.batch, torch.tensor([0, 0], dtype=torch.long))


def test_hook_separate_select_output_handles_empty_batch_without_normalization():
    values = SelectOutput(
        s=torch.empty((0, 2), dtype=torch.float32),
        batch=torch.empty((0,), dtype=torch.long),
    )
    slices = {
        "s": torch.tensor([0, 0]),
        "batch": torch.tensor([0, 0]),
    }
    batch = _graph(1)

    out = tgp_collate_module._hook_separate_select_output(
        key="so",
        values=values,
        idx=0,
        slices=slices,
        incs=None,
        batch=batch,
        store=batch.stores[0],
        decrement=True,
        separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
    )

    assert out.batch.numel() == 0


def test_hook_separate_select_output_raises_on_non_tensor_dense_input():
    values = _FakeSelectOutput(s={"bad": True})
    slices = {"s": torch.tensor([0, 1])}
    batch = _graph(1)

    with pytest.raises(TypeError, match="Tensor or SparseTensor"):
        tgp_collate_module._hook_separate_select_output(
            key="so",
            values=values,
            idx=0,
            slices=slices,
            incs=None,
            batch=batch,
            store=batch.stores[0],
            decrement=True,
            separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
        )


def test_hook_separate_select_output_raises_on_invalid_dense_rank():
    values = _FakeSelectOutput(s=torch.tensor(1.0), s_inv=None)
    slices = {"s": torch.tensor([0, 1])}
    batch = _graph(1)

    with pytest.raises(ValueError, match="2D \\[N, K\\] or 3D"):
        tgp_collate_module._hook_separate_select_output(
            key="so",
            values=values,
            idx=0,
            slices=slices,
            incs=None,
            batch=batch,
            store=batch.stores[0],
            decrement=True,
            separate_fn_map=tgp_collate_module._TGP_SEPARATE_FN_MAP,
        )
