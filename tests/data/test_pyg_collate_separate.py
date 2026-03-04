import pytest
import torch
from torch_geometric.data import Data
from torch_geometric.data.collate import collate as reference_collate
from torch_geometric.data.separate import separate as reference_separate

from tgp.data import pyg_collate as hook_collate_module
from tgp.data import pyg_separate as hook_separate_module

hook_collate = hook_collate_module.collate
hook_separate = hook_separate_module.separate


class Token:
    def __init__(self, name: str):
        self.name = name


class BatchedToken:
    def __init__(self, names):
        self.names = list(names)


def _token_collate_handler(**kwargs):
    values = kwargs["values"]
    slices = torch.arange(len(values) + 1)
    return BatchedToken([value.name for value in values]), slices, None


def _token_separate_handler(**kwargs):
    values = kwargs["values"]
    idx = kwargs["idx"]
    return Token(values.names[idx])


def _list_collate_handler(**kwargs):
    key = kwargs["key"]
    values = kwargs["values"]
    data_list = kwargs["data_list"]
    stores = kwargs["stores"]
    increment = kwargs["increment"]
    collate_fn_map = kwargs["collate_fn_map"]

    elem_size = len(values[0])
    if not all(len(value) == elem_size for value in values):
        raise RuntimeError("each element in list of batch should be of equal size")

    if elem_size == 0:
        slices = torch.arange(len(values) + 1)
        return values, slices, None

    transposed = list(zip(*values))
    value_list, slice_list, inc_list = [], [], []
    for samples in transposed:
        value, slices, incs = hook_collate_module._collate(
            key=key,
            values=list(samples),
            data_list=data_list,
            stores=stores,
            increment=increment,
            collate_fn_map=collate_fn_map,
        )
        value_list.append(value)
        slice_list.append(slices)
        inc_list.append(incs)

    return value_list, slice_list, inc_list


def _list_separate_handler(**kwargs):
    key = kwargs["key"]
    values = kwargs["values"]
    idx = kwargs["idx"]
    slices = kwargs["slices"]
    incs = kwargs["incs"]
    batch = kwargs["batch"]
    store = kwargs["store"]
    decrement = kwargs["decrement"]
    separate_fn_map = kwargs["separate_fn_map"]

    if not isinstance(values, list) or not isinstance(slices, list):
        return values[idx]

    return [
        hook_separate_module._separate(
            key=key,
            values=value,
            idx=idx,
            slices=slices[i],
            incs=incs[i] if decrement and incs is not None else None,
            batch=batch,
            store=store,
            decrement=decrement,
            separate_fn_map=separate_fn_map,
        )
        for i, value in enumerate(values)
    ]


def test_collate_separate_backwards_compatible_without_map():
    data_list = [
        Data(x=torch.tensor([[1.0], [2.0]]), edge_index=torch.tensor([[0, 1], [1, 0]])),
        Data(x=torch.tensor([[3.0]]), edge_index=torch.tensor([[0], [0]])),
    ]
    data_list[0].num_nodes = 2
    data_list[1].num_nodes = 1

    out_hook, slices_hook, incs_hook = hook_collate(Data, data_list)
    out_ref, slices_ref, incs_ref = reference_collate(Data, data_list)

    assert torch.equal(out_hook.x, out_ref.x)
    assert torch.equal(out_hook.edge_index, out_ref.edge_index)
    assert torch.equal(out_hook.batch, out_ref.batch)
    assert torch.equal(slices_hook["x"], slices_ref["x"])
    assert torch.equal(incs_hook["edge_index"], incs_ref["edge_index"])

    ex_hook = hook_separate(Data, out_hook, 1, slices_hook, incs_hook)
    ex_ref = reference_separate(Data, out_ref, 1, slices_ref, incs_ref)
    assert torch.equal(ex_hook.x, ex_ref.x)
    assert torch.equal(ex_hook.edge_index, ex_ref.edge_index)


def test_collate_tensor_sequence_backwards_compatible_without_map():
    data_list = [
        Data(parts=[torch.tensor([[1.0], [2.0]]), torch.tensor([[3.0]])]),
        Data(parts=[torch.tensor([[4.0]]), torch.tensor([[5.0], [6.0]])]),
    ]

    out_hook, slices_hook, incs_hook = hook_collate(Data, data_list)
    out_ref, slices_ref, incs_ref = reference_collate(Data, data_list)

    assert isinstance(out_hook.parts, list)
    assert len(out_hook.parts) == len(out_ref.parts)
    for part_hook, part_ref in zip(out_hook.parts, out_ref.parts):
        assert torch.equal(part_hook, part_ref)
    for slice_hook, slice_ref in zip(slices_hook["parts"], slices_ref["parts"]):
        assert torch.equal(slice_hook, slice_ref)
    for inc_hook, inc_ref in zip(incs_hook["parts"], incs_ref["parts"]):
        assert torch.equal(inc_hook, inc_ref)

    ex_hook = hook_separate(Data, out_hook, 1, slices_hook, incs_hook)
    ex_ref = reference_separate(Data, out_ref, 1, slices_ref, incs_ref)
    for part_hook, part_ref in zip(ex_hook.parts, ex_ref.parts):
        assert torch.equal(part_hook, part_ref)


def test_collate_hook_exact_type_precedes_subclass_order():
    class BaseToken:
        pass

    class ChildToken(BaseToken):
        pass

    def base_handler(**kwargs):
        slices = torch.arange(len(kwargs["values"]) + 1)
        return "base", slices, None

    def child_handler(**kwargs):
        slices = torch.arange(len(kwargs["values"]) + 1)
        return "child", slices, None

    data_list = [Data(tag=ChildToken()), Data(tag=ChildToken())]
    batch, _, _ = hook_collate(
        Data,
        data_list,
        collate_fn_map={BaseToken: base_handler, ChildToken: child_handler},
    )

    assert batch.tag == "child"


def test_collate_hook_subclass_fallback_uses_insertion_order():
    class Parent:
        pass

    class Child(Parent):
        pass

    def first_handler(**kwargs):
        slices = torch.arange(len(kwargs["values"]) + 1)
        return "first", slices, None

    def second_handler(**kwargs):
        slices = torch.arange(len(kwargs["values"]) + 1)
        return "second", slices, None

    data_list = [Data(tag=Child()), Data(tag=Child())]
    batch, _, _ = hook_collate(
        Data,
        data_list,
        collate_fn_map={object: first_handler, Parent: second_handler},
    )

    assert batch.tag == "first"


def test_separate_hook_round_trip_for_custom_batched_object():
    data_list = [Data(token=Token("a")), Data(token=Token("b"))]
    batch, slices, incs = hook_collate(
        Data,
        data_list,
        collate_fn_map={Token: _token_collate_handler},
    )

    restored = hook_separate(
        Data,
        batch,
        idx=1,
        slice_dict=slices,
        inc_dict=incs,
        separate_fn_map={BatchedToken: _token_separate_handler},
    )

    assert isinstance(restored.token, Token)
    assert restored.token.name == "b"


def test_nested_custom_sequence_requires_explicit_list_hook():
    data_list = [
        Data(tokens=[Token("a0"), Token("a1")]),
        Data(tokens=[Token("b0"), Token("b1")]),
    ]

    batch_no_list_hook, _, _ = hook_collate(
        Data,
        data_list,
        collate_fn_map={Token: _token_collate_handler},
    )
    assert isinstance(batch_no_list_hook.tokens, list)
    assert isinstance(batch_no_list_hook.tokens[0], list)
    assert isinstance(batch_no_list_hook.tokens[0][0], Token)

    batch, slices, incs = hook_collate(
        Data,
        data_list,
        collate_fn_map={list: _list_collate_handler, Token: _token_collate_handler},
    )
    assert isinstance(batch.tokens, list)
    assert all(isinstance(value, BatchedToken) for value in batch.tokens)

    restored = hook_separate(
        Data,
        batch,
        idx=1,
        slice_dict=slices,
        inc_dict=incs,
        separate_fn_map={
            list: _list_separate_handler,
            BatchedToken: _token_separate_handler,
        },
    )
    assert [token.name for token in restored.tokens] == ["b0", "b1"]


def test_list_hook_raises_on_inconsistent_nested_sequence_lengths():
    data_list = [
        Data(tokens=[Token("a0"), Token("a1")]),
        Data(tokens=[Token("b0")]),
    ]

    with pytest.raises(RuntimeError, match="equal size"):
        hook_collate(
            Data,
            data_list,
            collate_fn_map={list: _list_collate_handler, Token: _token_collate_handler},
        )
