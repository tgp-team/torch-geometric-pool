from collections.abc import Sequence
from inspect import signature
from typing import Any, Callable, Dict, Iterable, List, Optional, Tuple, Type

import torch
from torch import Tensor
from torch_geometric.data.collate import IncDictType, SliceDictType, T
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage
from torch_geometric.typing import SparseTensor, torch_sparse
from torch_geometric.utils import cumsum, is_sparse, is_torch_sparse_tensor, narrow
from torch_geometric.utils.sparse import cat

from tgp.select import SelectOutput

# Temporary vendored prototypes until hook support is available in upstream PyG.
# Once merged upstream, switch these imports to torch_geometric.data.*.
from . import pyg_collate as pyg_collate_module
from . import pyg_separate as pyg_separate_module

CollateFnMapType = Dict[Any, Callable[..., Tuple[Any, Any, Any]]]
SeparateFnMapType = Dict[Any, Callable[..., Any]]


def _hook_collate_base_data(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
    *,
    collate_fn_map: CollateFnMapType,
) -> Tuple[Any, Any, Any]:
    del key, data_list, stores, increment
    return pyg_collate_module.collate(
        cls=type(values[0]),
        data_list=values,
        increment=True,
        add_batch=True,
        follow_batch=None,
        exclude_keys=None,
        collate_fn_map=collate_fn_map,
    )


def _hook_collate_list(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
    *,
    collate_fn_map: CollateFnMapType,
) -> Tuple[Any, Any, Any]:
    elem_size = len(values[0])
    if not all(len(value) == elem_size for value in values):
        raise RuntimeError("each element in list of batch should be of equal size")

    if elem_size == 0:
        slices = torch.arange(len(values) + 1)
        return values, slices, None

    transposed = list(zip(*values))
    value_list, slice_list, inc_list = [], [], []
    for samples in transposed:
        value, slices, incs = pyg_collate_module._collate(
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


def _hook_collate_select_output(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
    *,
    collate_fn_map: CollateFnMapType,
) -> Tuple[Any, Any, Any]:
    elem = values[0]

    if is_sparse(elem.s):
        cat_dims = (0, 1)
        repeats = [[value.s.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(torch.tensor(repeats))
        if is_torch_sparse_tensor(elem.s):
            s = cat([value.s for value in values], dim=cat_dims)
        else:
            s = torch_sparse.cat([value.s for value in values], dim=cat_dims)

        s_inv = None
        if elem.s_inv is not None:
            if is_torch_sparse_tensor(elem.s_inv):
                s_inv = cat([value.s_inv for value in values], dim=cat_dims)
            else:
                s_inv = torch_sparse.cat(
                    [value.s_inv for value in values], dim=cat_dims
                )
    else:
        if not isinstance(elem.s, Tensor):
            raise TypeError(
                f"SelectOutput.s must be a Tensor or SparseTensor (got {type(elem.s)})."
            )

        if elem.s.dim() == 3:
            sizes = torch.tensor([value.s.size(0) for value in values])
            slices = cumsum(sizes)
            s = torch.cat([value.s for value in values], dim=0)
            s_inv = None
            if elem.s_inv is not None:
                s_inv = torch.cat([value.s_inv for value in values], dim=0)
        elif elem.s.dim() == 2:
            sizes = torch.tensor([value.s.size(0) for value in values])
            slices = cumsum(sizes)
            s = torch.cat([value.s for value in values], dim=0)
            s_inv = None
            if elem.s_inv is not None:
                s_inv = torch.cat([value.s_inv for value in values], dim=1)
        else:
            raise ValueError(
                "SelectOutput.s must be a 2D [N, K] or 3D [B, N, K] dense tensor "
                f"(got shape={tuple(elem.s.size())})."
            )

    has_batch = [
        hasattr(value, "batch") and value.batch is not None for value in values
    ]
    if any(has_batch) and not all(has_batch):
        raise ValueError(
            "Cannot collate SelectOutput objects when only some of them have "
            "a 'batch' attribute set."
        )

    if all(has_batch):
        batch_values = [value.batch for value in values]
        batch_slices = cumsum(torch.tensor([value.size(0) for value in batch_values]))
        batch_collated_parts = []
        batch_offset = 0
        for value in batch_values:
            batch_collated_parts.append(value + batch_offset)
            if value.numel() > 0:
                batch_offset += int(value.max().item()) + 1
        batch_collated = torch.cat(batch_collated_parts, dim=0)
    else:
        batch_collated = None
        batch_slices = None

    extra_keys = set(elem._extra_args)
    for value in values[1:]:
        if set(value._extra_args) != extra_keys:
            raise ValueError(
                "Cannot collate SelectOutput objects with different extra attributes."
            )

    extra_args = {}
    extra_slices = {}
    for extra_key in sorted(extra_keys):
        attr_values = [getattr(value, extra_key) for value in values]
        attr_value, attr_slices, _ = pyg_collate_module._collate(
            key=key,
            values=attr_values,
            data_list=data_list,
            stores=stores,
            increment=increment,
            collate_fn_map=collate_fn_map,
        )
        extra_args[extra_key] = attr_value
        extra_slices[extra_key] = attr_slices

    value = SelectOutput(s, s_inv, batch=batch_collated, **extra_args)
    all_slices = dict(s=slices, **extra_slices)
    if batch_slices is not None:
        all_slices["batch"] = batch_slices

    return value, all_slices, None


def _hook_separate_base_data(
    key: str,
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
    *,
    separate_fn_map: SeparateFnMapType,
) -> Any:
    del key, batch, store, decrement
    return pyg_separate_module.separate(
        cls=type(values),
        batch=values,
        idx=idx,
        slice_dict=slices,
        inc_dict=incs,
        decrement=True,
        separate_fn_map=separate_fn_map,
    )


def _hook_separate_list(
    key: str,
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
    *,
    separate_fn_map: SeparateFnMapType,
) -> Any:
    if (
        isinstance(values, Sequence)
        and not isinstance(values, str)
        and isinstance(slices, Sequence)
    ):
        return [
            pyg_separate_module._separate(
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

    return values[idx]


def _hook_separate_select_output(
    key: str,
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
    *,
    separate_fn_map: SeparateFnMapType,
) -> Any:
    del incs
    if is_sparse(values.s):
        (start_x, start_y), (end_x, end_y) = slices["s"][idx : idx + 2].tolist()
        if isinstance(values.s, SparseTensor):
            s = values.s[start_x:end_x, start_y:end_y]
            s_inv = None
            if values.s_inv is not None:
                s_inv = values.s_inv[start_y:end_y, start_x:end_x]
        else:
            s = narrow(values.s, 0, start_x, end_x - start_x)
            s = narrow(s, 1, start_y, end_y - start_y)
            s_inv = None
            if values.s_inv is not None:
                s_inv = narrow(values.s_inv, 0, start_y, end_y - start_y)
                s_inv = narrow(s_inv, 1, start_x, end_x - start_x)
    else:
        if not isinstance(values.s, Tensor):
            raise TypeError(
                "SelectOutput.s must be a Tensor or SparseTensor "
                f"(got {type(values.s)})."
            )

        s_slices = slices["s"]
        if s_slices.dim() == 2:
            start, end = int(s_slices[idx][0]), int(s_slices[idx + 1][0])
        else:
            start, end = int(s_slices[idx]), int(s_slices[idx + 1])

        if values.s.dim() == 2:
            s = values.s[start:end]
            s_inv = None
            if values.s_inv is not None:
                s_inv = values.s_inv[:, start:end]
        elif values.s.dim() == 3:
            s = values.s[start:end]
            s_inv = None
            if values.s_inv is not None:
                s_inv = values.s_inv[start:end]
        else:
            raise ValueError(
                "SelectOutput.s must be a 2D [N, K] or 3D [B, N, K] dense tensor "
                f"(got shape={tuple(values.s.size())})."
            )

    batch_attr = None
    if hasattr(values, "batch") and values.batch is not None and "batch" in slices:
        batch_attr = pyg_separate_module._separate(
            key=key,
            values=values.batch,
            idx=idx,
            slices=slices["batch"],
            incs=None,
            batch=batch,
            store=store,
            decrement=decrement,
            separate_fn_map=separate_fn_map,
        )
        if batch_attr.numel() > 0:
            batch_attr = batch_attr - batch_attr.min()

    extra_args = {
        extra_key: pyg_separate_module._separate(
            key=key,
            values=getattr(values, extra_key),
            idx=idx,
            slices=slices[extra_key],
            incs=None,
            batch=batch,
            store=store,
            decrement=decrement,
            separate_fn_map=separate_fn_map,
        )
        for extra_key in values._extra_args
    }
    return SelectOutput(s, s_inv, batch=batch_attr, **extra_args)


_TGP_COLLATE_FN_MAP: CollateFnMapType = {
    BaseData: _hook_collate_base_data,
    list: _hook_collate_list,
    SelectOutput: _hook_collate_select_output,
}

_TGP_SEPARATE_FN_MAP: SeparateFnMapType = {
    BaseData: _hook_separate_base_data,
    list: _hook_separate_list,
    SelectOutput: _hook_separate_select_output,
}

_HAS_PYG_COLLATE_HOOKS = (
    "collate_fn_map" in signature(pyg_collate_module.collate).parameters
)
_HAS_PYG_SEPARATE_HOOKS = (
    "separate_fn_map" in signature(pyg_separate_module.separate).parameters
)


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    if not _HAS_PYG_COLLATE_HOOKS:
        raise RuntimeError(
            "PyG collate module does not expose 'collate_fn_map'. "
            "Use hook-enabled pyg_collate implementation."
        )

    return pyg_collate_module.collate(
        cls=cls,
        data_list=data_list,
        increment=increment,
        add_batch=add_batch,
        follow_batch=follow_batch,
        exclude_keys=exclude_keys,
        collate_fn_map=_TGP_COLLATE_FN_MAP,
    )


def separate(
    cls: Type[T],
    batch: Any,
    idx: int,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
) -> T:
    if not _HAS_PYG_SEPARATE_HOOKS:
        raise RuntimeError(
            "PyG separate module does not expose 'separate_fn_map'. "
            "Use hook-enabled pyg_separate implementation."
        )

    return pyg_separate_module.separate(
        cls=cls,
        batch=batch,
        idx=idx,
        slice_dict=slice_dict,
        inc_dict=inc_dict,
        decrement=decrement,
        separate_fn_map=_TGP_SEPARATE_FN_MAP,
    )
