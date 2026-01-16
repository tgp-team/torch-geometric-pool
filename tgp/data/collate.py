from collections import defaultdict
from collections.abc import Mapping, Sequence
from typing import (
    Any,
    Iterable,
    List,
    Optional,
    Tuple,
    Type,
)

import torch
import torch_geometric.typing
from torch import Tensor
from torch_geometric import EdgeIndex, Index
from torch_geometric.data import Batch
from torch_geometric.data.collate import (
    IncDictType,
    SliceDictType,
    T,
    _batch_and_ptr,
    get_incs,
    repeat_interleave,
)
from torch_geometric.data.data import BaseData
from torch_geometric.data.storage import BaseStorage, NodeStorage
from torch_geometric.edge_index import SortOrder
from torch_geometric.typing import (
    SparseTensor,
    TensorFrame,
    torch_frame,
    torch_sparse,
)
from torch_geometric.utils import cumsum, is_sparse, is_torch_sparse_tensor, narrow
from torch_geometric.utils.sparse import cat

from tgp.select import SelectOutput


def collate(
    cls: Type[T],
    data_list: List[BaseData],
    increment: bool = True,
    add_batch: bool = True,
    follow_batch: Optional[Iterable[str]] = None,
    exclude_keys: Optional[Iterable[str]] = None,
) -> Tuple[T, SliceDictType, IncDictType]:
    # Collates a list of `data` objects into a single object of type `cls`.
    # `collate` can handle both homogeneous and heterogeneous data objects by
    # individually collating all their stores.
    # In addition, `collate` can handle nested data structures such as
    # dictionaries and lists.

    if not isinstance(data_list, (list, tuple)):
        # Materialize `data_list` to keep the `_parent` weakref alive.
        data_list = list(data_list)

    if cls != data_list[0].__class__:  # Dynamic inheritance.
        out = cls(_base_cls=data_list[0].__class__)  # type: ignore
    else:
        out = cls()

    # Create empty stores:
    out.stores_as(data_list[0])  # type: ignore

    follow_batch = set(follow_batch or [])
    exclude_keys = set(exclude_keys or [])

    # Group all storage objects of every data object in the `data_list` by key,
    # i.e. `key_to_stores = { key: [store_1, store_2, ...], ... }`:
    key_to_stores = defaultdict(list)
    for data in data_list:
        for store in data.stores:
            key_to_stores[store._key].append(store)

    # With this, we iterate over each list of storage objects and recursively
    # collate all its attributes into a unified representation:

    # We maintain two additional dictionaries:
    # * `slice_dict` stores a compressed index representation of each attribute
    #    and is needed to re-construct individual elements from mini-batches.
    # * `inc_dict` stores how individual elements need to be incremented, e.g.,
    #   `edge_index` is incremented by the cumulated sum of previous elements.
    #   We also need to make use of `inc_dict` when re-constructuing individual
    #   elements as attributes that got incremented need to be decremented
    #   while separating to obtain original values.
    device: Optional[torch.device] = None
    slice_dict: SliceDictType = {}
    inc_dict: IncDictType = {}
    for out_store in out.stores:  # type: ignore
        key = out_store._key
        stores = key_to_stores[key]
        for attr in stores[0].keys():
            if attr in exclude_keys:  # Do not include top-level attribute.
                continue

            values = [store[attr] for store in stores]

            # The `num_nodes` attribute needs special treatment, as we need to
            # sum their values up instead of merging them to a list:
            if attr == "num_nodes":
                out_store._num_nodes = values
                out_store.num_nodes = sum(values)
                continue

            # Skip batching of `ptr` vectors for now:
            if attr == "ptr":
                continue

            # Collate attributes into a unified representation:
            value, slices, incs = _collate(
                attr, values, data_list, stores, increment, cls
            )

            # If parts of the data are already on GPU, make sure that auxiliary
            # data like `batch` or `ptr` are also created on GPU:
            if isinstance(value, Tensor) and value.is_cuda:
                device = value.device

            out_store[attr] = value

            if key is not None:  # Heterogeneous:
                store_slice_dict = slice_dict.get(key, {})
                assert isinstance(store_slice_dict, dict)
                store_slice_dict[attr] = slices
                slice_dict[key] = store_slice_dict

                store_inc_dict = inc_dict.get(key, {})
                assert isinstance(store_inc_dict, dict)
                store_inc_dict[attr] = incs
                inc_dict[key] = store_inc_dict
            else:  # Homogeneous:
                slice_dict[attr] = slices
                inc_dict[attr] = incs

            # Add an additional batch vector for the given attribute:
            if attr in follow_batch:
                batch, ptr = _batch_and_ptr(slices, device)
                out_store[f"{attr}_batch"] = batch
                out_store[f"{attr}_ptr"] = ptr

        # In case of node-level storages, we add a top-level batch vector it:
        if (
            add_batch
            and isinstance(stores[0], NodeStorage)
            and stores[0].can_infer_num_nodes
        ):
            repeats = [store.num_nodes or 0 for store in stores]
            out_store.batch = repeat_interleave(repeats, device=device)
            out_store.ptr = cumsum(torch.tensor(repeats, device=device))

    return out, slice_dict, inc_dict


def _collate(
    key: str,
    values: List[Any],
    data_list: List[BaseData],
    stores: List[BaseStorage],
    increment: bool,
    super_cls: Type = None,
) -> Tuple[Any, Any, Any]:
    elem = values[0]

    if isinstance(elem, Tensor) and not is_sparse(elem):
        # Concatenate a list of `torch.Tensor` along the `cat_dim`.
        # NOTE: We need to take care of incrementing elements appropriately.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        if cat_dim is None or elem.dim() == 0:
            values = [value.unsqueeze(0) for value in values]
        sizes = torch.tensor([value.size(cat_dim or 0) for value in values])
        slices = cumsum(sizes)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if incs.dim() > 1 or int(incs[-1]) != 0:
                values = [
                    value + inc.to(value.device) for value, inc in zip(values, incs)
                ]
        else:
            incs = None

        if getattr(elem, "is_nested", False):
            tensors = []
            for nested_tensor in values:
                tensors.extend(nested_tensor.unbind())
            value = torch.nested.nested_tensor(tensors)

            return value, slices, incs

        out = None
        if torch.utils.data.get_worker_info() is not None and not isinstance(
            elem, (Index, EdgeIndex)
        ):
            # Write directly into shared memory to avoid an extra copy:
            numel = sum(value.numel() for value in values)
            if torch_geometric.typing.WITH_PT20:
                storage = elem.untyped_storage()._new_shared(
                    numel * elem.element_size(), device=elem.device
                )
            elif torch_geometric.typing.WITH_PT112:
                storage = elem.storage()._new_shared(numel, device=elem.device)
            else:
                storage = elem.storage()._new_shared(numel)
            shape = list(elem.size())
            if cat_dim is None or elem.dim() == 0:
                shape = [len(values)] + shape
            else:
                shape[cat_dim] = int(slices[-1])
            out = elem.new(storage).resize_(*shape)

        value = torch.cat(values, dim=cat_dim or 0, out=out)

        if increment and isinstance(value, Index) and values[0].is_sorted:
            # Check whether the whole `Index` is sorted:
            if (value.diff() >= 0).all():
                value._is_sorted = True

        if increment and isinstance(value, EdgeIndex) and values[0].is_sorted:
            # Check whether the whole `EdgeIndex` is sorted by row:
            if values[0].is_sorted_by_row and (value[0].diff() >= 0).all():
                value._sort_order = SortOrder.ROW
            # Check whether the whole `EdgeIndex` is sorted by column:
            elif values[0].is_sorted_by_col and (value[1].diff() >= 0).all():
                value._sort_order = SortOrder.COL

        return value, slices, incs

    elif isinstance(elem, TensorFrame):
        key = str(key)
        sizes = torch.tensor([value.num_rows for value in values])
        slices = cumsum(sizes)
        value = torch_frame.cat(values, dim=0)
        return value, slices, None

    elif is_sparse(elem) and increment:
        # Concatenate a list of `SparseTensor` along the `cat_dim`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = data_list[0].__cat_dim__(key, elem, stores[0])
        cat_dims = (cat_dim,) if isinstance(cat_dim, int) else cat_dim
        repeats = [[value.size(dim) for dim in cat_dims] for value in values]
        slices = cumsum(torch.tensor(repeats))
        if is_torch_sparse_tensor(elem):
            value = cat(values, dim=cat_dim)
        else:
            value = torch_sparse.cat(values, dim=cat_dim)
        return value, slices, None

    elif isinstance(elem, BaseData):
        return collate(
            cls=super_cls or type(elem),
            data_list=values,
            increment=True,
            add_batch=True,
            follow_batch=None,
            exclude_keys=None,
        )

    elif isinstance(elem, SelectOutput):
        if isinstance(elem.s, SparseTensor):
            # Sparse case: use torch_sparse.cat for block-diagonal concatenation.
            cat_dims = (0, 1)
            repeats = [[value.s.size(dim) for dim in cat_dims] for value in values]
            slices = cumsum(torch.tensor(repeats))
            s = torch_sparse.cat([value.s for value in values], dim=cat_dims)
            s_inv = None
            if elem.s_inv is not None:
                s_inv = torch_sparse.cat(
                    [value.s_inv for value in values], dim=cat_dims
                )
        else:
            # Dense case.
            if not isinstance(elem.s, Tensor):
                raise TypeError(
                    "SelectOutput.s must be a Tensor or SparseTensor "
                    f"(got {type(elem.s)})."
                )

            if elem.s.dim() == 3:
                # Batched dense: concatenate along the batch dimension.
                sizes = torch.tensor([value.s.size(0) for value in values])
                slices = cumsum(sizes)
                s = torch.cat([value.s for value in values], dim=0)
                s_inv = None
                if elem.s_inv is not None:
                    s_inv = torch.cat([value.s_inv for value in values], dim=0)
            elif elem.s.dim() == 2:
                # Unbatched dense: concatenate along the node dimension.
                sizes = torch.tensor([value.s.size(0) for value in values])
                slices = cumsum(sizes)
                s = torch.cat([value.s for value in values], dim=0)
                s_inv = None
                if elem.s_inv is not None:
                    # s_inv is [K, N] (transpose of s), so concatenate along N (dim=1).
                    s_inv = torch.cat([value.s_inv for value in values], dim=1)
            else:
                raise ValueError(
                    "SelectOutput.s must be a 2D [N, K] or 3D [B, N, K] dense tensor "
                    f"(got shape={tuple(elem.s.size())})."
                )

        # Handle SelectOutput.batch (if present).
        has_batch = [v.batch is not None for v in values]
        if any(has_batch) and not all(has_batch):
            raise ValueError(
                "Cannot collate SelectOutput objects when only some of them have "
                "a 'batch' attribute set."
            )

        if all(has_batch):
            batch_values = [v.batch for v in values]
            batch_slices = cumsum(torch.tensor([v.size(0) for v in batch_values]))

            batch_collated_parts = []
            batch_offset = 0
            for v in batch_values:
                batch_collated_parts.append(v + batch_offset)
                if v.numel() > 0:
                    batch_offset += int(v.max().item()) + 1

            batch_collated = torch.cat(batch_collated_parts, dim=0)
        else:
            batch_collated = None
            batch_slices = None

        # Handle extra_args (excluding batch which is now a proper attribute)
        extra_args = dict()
        extra_slices = dict()
        for k in elem._extra_args:
            attr_values = [getattr(v, k) for v in values]
            k_value, k_slices, k_incs = _collate(
                key,
                attr_values,
                data_list,
                stores,
                increment,
                super_cls,
            )
            extra_args[k] = k_value
            extra_slices[k] = k_slices

        value = SelectOutput(s, s_inv, batch=batch_collated, **extra_args)
        slices = dict(s=slices, **extra_slices)
        if batch_slices is not None:
            slices["batch"] = batch_slices
        return value, slices, None

    elif isinstance(elem, (int, float)):
        # Convert a list of numerical values to a `torch.Tensor`.
        value = torch.tensor(values)
        if increment:
            incs = get_incs(key, values, data_list, stores)
            if int(incs[-1]) != 0:
                value.add_(incs)
        else:
            incs = None
        slices = torch.arange(len(values) + 1)
        return value, slices, incs

    elif isinstance(elem, Mapping):
        # Recursively collate elements of dictionaries.
        value_dict, slice_dict, inc_dict = {}, {}, {}
        for key in elem.keys():
            value_dict[key], slice_dict[key], inc_dict[key] = _collate(
                key, [v[key] for v in values], data_list, stores, increment, super_cls
            )
        return value_dict, slice_dict, inc_dict

    elif isinstance(elem, Sequence) and not isinstance(elem, str) and len(elem) > 0:
        # Recursively collate elements of lists.
        value_list, slice_list, inc_list = [], [], []
        for i in range(len(elem)):
            value, slices, incs = _collate(
                key, [v[i] for v in values], data_list, stores, increment, super_cls
            )
            value_list.append(value)
            slice_list.append(slices)
            inc_list.append(incs)
        return value_list, slice_list, inc_list

    else:
        # Other-wise, just return the list of values as it is.
        slices = torch.arange(len(values) + 1)
        return values, slices, None


def separate(
    cls: Type[T],
    batch: Any,
    idx: int,
    slice_dict: Any,
    inc_dict: Any = None,
    decrement: bool = True,
) -> T:
    # Separates the individual element from a `batch` at index `idx`.
    # `separate` can handle both homogeneous and heterogeneous data objects by
    # individually separating all their stores.
    # In addition, `separate` can handle nested data structures such as
    # dictionaries and lists.

    data = cls().stores_as(batch)

    # Iterate over each storage object and recursively separate its attributes:
    for batch_store, data_store in zip(batch.stores, data.stores):
        key = batch_store._key
        if key is not None:  # Heterogeneous:
            attrs = slice_dict[key].keys()
        else:  # Homogeneous:
            attrs = set(batch_store.keys())
            attrs = [attr for attr in slice_dict.keys() if attr in attrs]

        for attr in attrs:
            if key is not None:
                slices = slice_dict[key][attr]
                incs = inc_dict[key][attr] if decrement else None
            else:
                slices = slice_dict[attr]
                incs = inc_dict[attr] if decrement else None

            data_store[attr] = _separate(
                attr,
                batch_store[attr],
                idx,
                slices,
                incs,
                batch,
                batch_store,
                decrement,
            )

        # The `num_nodes` attribute needs special treatment, as we cannot infer
        # the real number of nodes from the total number of nodes alone:
        if hasattr(batch_store, "_num_nodes"):
            data_store.num_nodes = batch_store._num_nodes[idx]

    return data


def _separate(
    key: str,
    values: Any,
    idx: int,
    slices: Any,
    incs: Any,
    batch: BaseData,
    store: BaseStorage,
    decrement: bool,
) -> Any:
    if isinstance(values, Tensor):
        # Narrow a `torch.Tensor` based on `slices`.
        # NOTE: We need to take care of decrementing elements appropriately.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, values, store)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = narrow(values, cat_dim or 0, start, end - start)
        value = value.squeeze(0) if cat_dim is None else value

        if isinstance(values, Index) and values._cat_metadata is not None:
            # Reconstruct original `Index` metadata:
            value._dim_size = values._cat_metadata.dim_size[idx]
            value._is_sorted = values._cat_metadata.is_sorted[idx]

        if isinstance(values, EdgeIndex) and values._cat_metadata is not None:
            # Reconstruct original `EdgeIndex` metadata:
            value._sparse_size = values._cat_metadata.sparse_size[idx]
            value._sort_order = values._cat_metadata.sort_order[idx]
            value._is_undirected = values._cat_metadata.is_undirected[idx]

        if decrement and incs is not None and (incs.dim() > 1 or int(incs[idx]) != 0):
            value = value - incs[idx].to(value.device)

        return value

    elif isinstance(values, SparseTensor) and decrement:
        # Narrow a `SparseTensor` based on `slices`.
        # NOTE: `cat_dim` may return a tuple to allow for diagonal stacking.
        key = str(key)
        cat_dim = batch.__cat_dim__(key, values, store)
        cat_dims = (cat_dim,) if isinstance(cat_dim, int) else cat_dim
        for i, dim in enumerate(cat_dims):
            start, end = int(slices[idx][i]), int(slices[idx + 1][i])
            values = values.narrow(dim, start, end - start)
        return values

    elif isinstance(values, TensorFrame):
        key = str(key)
        start, end = int(slices[idx]), int(slices[idx + 1])
        value = values[start:end]
        return value

    elif isinstance(values, Batch):
        return separate(
            cls=values.__class__.__bases__[-1],
            batch=values,
            idx=idx,
            slice_dict=slices,
            inc_dict=incs,
            decrement=True,
        )

    elif isinstance(values, SelectOutput):
        if isinstance(values.s, SparseTensor):
            (start_x, start_y), (end_x, end_y) = slices["s"][idx : idx + 2].tolist()
            s = values.s[start_x:end_x, start_y:end_y]
            s_inv = None
            if values.s_inv is not None:
                # s_inv has transposed shape (K x N), so we need to swap dims.
                s_inv = values.s_inv[start_y:end_y, start_x:end_x]
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
        if values.batch is not None and "batch" in slices:
            batch_attr = _separate(
                key,
                values.batch,
                idx,
                slices=slices["batch"],
                incs=None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            if batch_attr.numel() > 0:
                batch_attr = batch_attr - batch_attr.min()

        extra_args = {
            k: _separate(
                key,
                getattr(values, k),
                idx,
                slices=slices[k],
                incs=None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for k in values._extra_args
        }
        value = SelectOutput(s, s_inv, batch=batch_attr, **extra_args)
        return value

    elif isinstance(values, Mapping):
        # Recursively separate elements of dictionaries.
        return {
            key: _separate(
                key,
                value,
                idx,
                slices=slices[key],
                incs=incs[key] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for key, value in values.items()
        }

    elif (
        isinstance(values, Sequence)
        and isinstance(values[0], Sequence)
        and not isinstance(values[0], str)
        and len(values[0]) > 0
        and isinstance(values[0][0], (Tensor, SparseTensor))
        and isinstance(slices, Sequence)
    ):
        # Recursively separate elements of lists of lists.
        return [value[idx] for value in values]

    elif (
        isinstance(values, Sequence)
        and not isinstance(values, str)
        and isinstance(slices, Sequence)
    ):
        # Recursively separate elements of lists of Tensors/SparseTensors.
        return [
            _separate(
                key,
                value,
                idx,
                slices=slices[i],
                incs=incs[i] if decrement else None,
                batch=batch,
                store=store,
                decrement=decrement,
            )
            for i, value in enumerate(values)
        ]

    else:
        return values[idx]
