from collections.abc import Mapping, Sequence
from typing import Any, Dict, Optional, Tuple, Union

import torch
from torch_geometric.data import Data
from torch_geometric.transforms import BaseTransform
from torch_geometric.utils import (
    add_self_loops,
    coalesce,
    get_laplacian,
    sort_edge_index,
)

from tgp.src import SRCPooling

PoolerLevelSpec = Union[
    SRCPooling,
    str,
    Tuple[str, Dict[str, Any]],
    Dict[str, Any],
]


class NormalizeAdj(BaseTransform):
    r"""Transforms the adjacency matrix :math:`\mathbf{A}`
    by applying the following transformation:

    .. math::
        \mathbf{A} \to \mathbf{I} - \delta \mathbf{L}

    where :math:`\mathbf{L}` is the normalized Laplacian
    of the graph and :math:`\delta` is a scaling factor.

    Args:
        delta (int, optional):
            Scaling factor for the Laplacian.
            (default: :obj:`0.85`)
    """

    def __init__(self, delta: float = 0.85) -> None:
        self.delta = delta
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        r"""Applies the normalization transform to the graph while preserving edge attributes.

        The transform computes the normalized Laplacian and rescales it with :math:`-\delta`. It also handles self-loops and
        concatenates additional edge attributes if available. Duplicate entries are coalesced by summing their values.

        Args:
            data (~torch_geometric.data.Data):
                A Data object containing graph data.

        Returns:
            ~torch_geometric.data.Data: The transformed data object with updated edge_index, edge_weight, and (optionally) edge_attr.
        """
        assert data.edge_index is not None
        N = data.num_nodes

        edge_index, edge_weight = data.edge_index, data.edge_weight

        # Check how many edges have self loops
        self_loop_mask = edge_index[0] == edge_index[1]
        initial_self_loops = self_loop_mask.sum().item()

        # Get the symmetrically normalized Laplacian (I - D^-.5 A D^-.5) in sparse format
        edge_index, edge_weight = get_laplacian(
            edge_index, edge_weight, normalization="sym", num_nodes=N
        )

        # Check if new self loops have been added
        new_self_loop_mask = edge_index[0] == edge_index[1]
        num_new_self_loops = new_self_loop_mask.sum().item() - initial_self_loops

        # Rescale the Laplacian weights by -delta
        edge_weight = -self.delta * edge_weight

        # Add self-loops representing the identity matrix
        edge_index, edge_weight = add_self_loops(
            edge_index, edge_weight, fill_value=1.0, num_nodes=N
        )

        # Prepare edge attributes for coalescing
        if data.edge_attr is not None:
            if num_new_self_loops > 0:
                num_self_loops = (
                    2 * N
                )  # self loops from Laplacian and self loops from add_self_loops
            else:
                num_self_loops = N  # self loops only from add_self_loops

            # Create zero edge attributes for the self-loops
            attr_dim = data.edge_attr.size(1)
            self_loop_attr = torch.zeros(
                num_self_loops, attr_dim, device=data.edge_attr.device
            )

            # Concatenate original edge attributes and self-loop attributes
            edge_attr = torch.cat([data.edge_attr, self_loop_attr], dim=0)

        else:
            edge_attr = None

        # Prepare edge values for coalescing
        if edge_attr is not None:
            edge_weight = edge_weight.unsqueeze(
                1
            )  # Shape: [num_edges + num_self_loops, 1]
            edge_value = torch.cat(
                [edge_weight, edge_attr], dim=1
            )  # Shape: [num_edges + num_self_loops, 1 + attr_dim]
        else:
            edge_value = edge_weight  # Shape: [num_edges + num_self_loops]

        # Coalesce the sparse matrix to remove duplicate entries and sum their values
        edge_index, edge_value = coalesce(edge_index, edge_value, N)

        # Split edge_value back into edge_weight and edge_attr
        if edge_attr is not None:
            edge_weight = edge_value[:, 0]
            edge_attr = edge_value[:, 1:].to(data.edge_attr.dtype)
            data.edge_attr = edge_attr
        else:
            edge_weight = edge_value

        data.edge_index = edge_index
        data.edge_weight = edge_weight

        return data


class SortNodes(BaseTransform):
    """Sorts the nodes of a graph based on their labels."""

    def __init__(self) -> None:
        super().__init__()

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        r"""Sorts the nodes of the graph according to their labels.

        The function sorts the node labels, reassigns node indices accordingly, and updates the associated attributes.
        If edge attributes exist, they are re-sorted based on the new node ordering.

        Args:
            data (~torch_geometric.data.Data):
                A Data object containing graph data with attributes
                :attr:`edge_index`, :attr:`y`, :attr:`x`, and optionally :attr:`edge_attr`.

        Returns:
            ~torch_geometric.data.Data: The data object with sorted nodes and updated attributes.
        """
        assert data.edge_index is not None
        assert data.y is not None
        y_sorted, sort_idx = torch.sort(data.y)
        edge_index_renamed = torch.empty_like(data.edge_index)
        for new_i in range(data.num_nodes):
            i = sort_idx[new_i]
            mask_i = data.edge_index == i
            edge_index_renamed[mask_i] = new_i

        data.x = data.x[sort_idx]
        data.y = y_sorted
        # sort edge_index_renamed in order to have edges ordered by source
        if data.edge_attr is not None:
            data.edge_index, (data.edge_weight, data.edge_attr) = sort_edge_index(
                edge_index_renamed, edge_attr=[data.edge_weight, data.edge_attr]
            )
        else:
            data.edge_index, data.edge_weight = sort_edge_index(
                edge_index_renamed, data.edge_weight
            )

        return data


class PreCoarsening(BaseTransform):
    r"""A transform that precomputes a hierarchy of pooled (coarsened) graphs
    and attaches them to the input :class:`~torch_geometric.data.Data` object.

    Takes a pooling operator from :class:`~tgp.src.SRCPooling` to build
    a multi-level pooling hierarchy. The pooling operator should implement the
    function :meth:`~tgp.src.SRCPooling.precoarsening`, which computes the pooled graph
    from the original graph. The pooling operator must not be trainable,
    i.e., it should not have learnable parameters.
    The graph is recursively coarsened for :obj:`recursive_depth` levels.
    At each level, a coarsened adjacency matrix and, optionally, a pooled batch
    is computed. The result is stored as a list of intermediate pooled subgraphs
    in :class:`~torch_geometric.data.Data`, which downstream GNN models can consume.

    Args:
        pooler (~tgp.src.SRCPooling, optional):
            A non-trainable pooling operator that implements
            :meth:`~tgp.src.SRCPooling.precoarsening`.
            Used for every level when :obj:`poolers` is not provided.
        input_key (str, optional):
            The key in the data object from which to read the graph data.
            If :obj:`None`, uses the default data object.
        output_key (str, optional):
            The key in the data object where the pooled graphs will be stored.
            Defaults to :obj:`"pooled_data"`.
        recursive_depth (int):
            Number of recursive coarsening levels when using :obj:`pooler`.
            Must be greater than 0.
        poolers (Sequence, optional):
            Optional per-level pooler specification. If provided, it overrides
            :obj:`pooler` and :obj:`recursive_depth`.
            Each entry can be one of:

            - a pre-instantiated pooler object;
            - a pooler alias string, e.g. :obj:`"ndp"`;
            - a tuple :obj:`("eigen", {"k": 5})`;
            - a dictionary with keys :obj:`{"pooler": "<name>", ...kwargs}`
              or :obj:`{"name": "<name>", ...kwargs}`.
    """

    def __init__(
        self,
        pooler: Optional[SRCPooling] = None,
        input_key: str = None,
        output_key: str = "pooled_data",
        recursive_depth: int = 1,
        poolers: Optional[Sequence[PoolerLevelSpec]] = None,
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

        if poolers is None:
            if pooler is None:
                raise ValueError(
                    "`pooler` must be provided when `poolers` is not specified."
                )
            if recursive_depth <= 0:
                raise ValueError("`recursive_depth` must be greater than 0.")
            poolers = [pooler for _ in range(recursive_depth)]

        self._pooler_entries = tuple(
            self._resolve_pooler_spec_with_key(spec) for spec in poolers
        )
        self.poolers = tuple(pooler for pooler, _ in self._pooler_entries)
        if not self.poolers:
            raise ValueError("At least one pooling level is required.")
        self._pooler_runs = tuple(self._collapse_consecutive_runs(self._pooler_entries))
        self.pooler = self.poolers[0]  # Backward compatibility.
        self.recursive_depth = len(self.poolers)

    @staticmethod
    def _build_pooler(pooler_name: str, kwargs: Optional[Dict[str, Any]] = None):
        from tgp.poolers import get_pooler

        return get_pooler(pooler_name, **(kwargs or {}))

    @staticmethod
    def _freeze_config_value(value: Any):
        """Convert nested configuration values into hashable structures."""
        if isinstance(value, Mapping):
            return tuple(
                sorted(
                    (str(k), PreCoarsening._freeze_config_value(v))
                    for k, v in value.items()
                )
            )
        if isinstance(value, (list, tuple)):
            return tuple(PreCoarsening._freeze_config_value(v) for v in value)
        if isinstance(value, set):
            return tuple(
                sorted(
                    (PreCoarsening._freeze_config_value(v) for v in value),
                    key=repr,
                )
            )
        try:
            hash(value)
            return value
        except TypeError:
            return repr(value)

    def _resolve_pooler_spec(self, spec: PoolerLevelSpec) -> SRCPooling:
        pooler, _ = self._resolve_pooler_spec_with_key(spec)
        return pooler

    def _resolve_pooler_spec_with_key(
        self, spec: PoolerLevelSpec
    ) -> tuple[SRCPooling, tuple]:
        if isinstance(spec, dict):
            spec_dict = dict(spec)
            spec = (
                spec_dict.pop("pooler", spec_dict.pop("name", None)),
                spec_dict,
            )

        if isinstance(spec, tuple):
            pooler_or_name = spec[0]
            if pooler_or_name is None:
                raise ValueError("Pooler spec must include a pooler name or instance.")
            pooler_kwargs = dict(spec[1] or {})
            if isinstance(pooler_or_name, SRCPooling):
                if pooler_kwargs:
                    raise ValueError(
                        "Cannot provide kwargs together with an instantiated pooler."
                    )
                pooler = pooler_or_name
                collapse_key = ("obj", id(pooler))
            else:
                pooler_name = str(pooler_or_name).lower()
                pooler = self._build_pooler(pooler_name, pooler_kwargs)
                collapse_key = (
                    "spec",
                    pooler_name,
                    self._freeze_config_value(pooler_kwargs),
                )
        elif isinstance(spec, str):
            pooler_name = spec.lower()
            pooler = self._build_pooler(pooler_name)
            collapse_key = ("spec", pooler_name, ())
        else:
            pooler = spec
            collapse_key = ("obj", id(pooler))

        if pooler.is_trainable:
            raise ValueError("The pooler must not be trainable.")
        return pooler, collapse_key

    @staticmethod
    def _collapse_consecutive_runs(
        entries: Sequence[tuple[SRCPooling, tuple]],
    ) -> list[tuple[SRCPooling, int]]:
        if len(entries) == 0:
            return []

        collapsed: list[tuple[SRCPooling, int]] = []
        current_pooler, current_key = entries[0]
        run_len = 1

        for pooler, key in entries[1:]:
            if key == current_key:
                run_len += 1
                continue
            collapsed.append((current_pooler, run_len))
            current_pooler, current_key = pooler, key
            run_len = 1
        collapsed.append((current_pooler, run_len))
        return collapsed

    @torch.no_grad()
    def forward(self, data: Data) -> Data:
        """Precomputes the select and connect operations for a graph."""
        data_obj = data if self.input_key is None else getattr(data, self.input_key)

        pooled_out = []
        for pooler, run_len in self._pooler_runs:
            run_outputs = pooler.multi_level_precoarsening(
                levels=run_len,
                edge_index=data_obj.edge_index,
                edge_weight=data_obj.edge_weight,
                batch=data_obj.batch,
                num_nodes=data_obj.num_nodes,
            )
            if len(run_outputs) != run_len:
                raise ValueError(
                    f"{type(pooler).__name__}.multi_level_precoarsening returned "
                    f"{len(run_outputs)} levels, expected {run_len}."
                )

            for data_pooled in run_outputs:
                data_obj = data_pooled.as_data()
                pooled_out.append(data_obj)

        setattr(data, self.output_key, pooled_out)

        return data
