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

PoolerLevelConfig = Union[
    SRCPooling,
    str,
    Tuple[str, Dict[str, Any]],
    Dict[str, Any],
]
# First argument to PreCoarsening: a single pooler/config or a sequence of them.
PoolersArg = Union[PoolerLevelConfig, Sequence[PoolerLevelConfig]]
ResolvedLevelEntry = tuple[SRCPooling, tuple[Any, ...]]
CollapsedLevelRun = tuple[SRCPooling, int]


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

    Takes one or more pooling operators from :class:`~tgp.src.SRCPooling` to
    build a multi-level pooling hierarchy. Each pooler must implement
    :meth:`~tgp.src.Precoarsenable.multi_level_precoarsening`.
    The default implementation is greedy repeated
    :meth:`~tgp.src.SRCPooling.precoarsening`.

    Some poolers override :meth:`~tgp.src.Precoarsenable.precoarsening` to
    enforce method-specific behavior at each level (e.g.,
    :meth:`~tgp.poolers.nmf.NMFPooling.precoarsening` and
    :meth:`~tgp.poolers.eigenpool.EigenPooling.precoarsening` keep a fixed
    assignment width), while others override
    :meth:`~tgp.src.Precoarsenable.multi_level_precoarsening` to implement a
    custom hierarchy rollout (e.g.,
    :meth:`~tgp.poolers.sep.SEPPooling.multi_level_precoarsening`).
    Poolers must be non-trainable, i.e., they should not have learnable
    parameters.
    The graph is recursively coarsened for as many levels as given in :obj:`poolers`.
    At each level, a coarsened adjacency matrix and, optionally, a pooled batch
    is computed. The result is stored as a list of intermediate pooled subgraphs
    in :class:`~torch_geometric.data.Data`, which downstream GNN models can consume.

    Args:
        poolers (PoolersArg):
            Per-level pooler configuration. Can be a single pooler or a
            sequence of level configs. A single value is treated as one level.
            Each entry can be one of:

            - a pre-instantiated pooler instance;
            - a pooler alias string, e.g. :obj:`"ndp"`;
            - a tuple :obj:`("eigen", {"k": 5})`;
            - a dictionary with keys :obj:`{"pooler": "<name>", ...kwargs}`
              or :obj:`{"name": "<name>", ...kwargs}`.

            To use the same pooler for multiple levels, pass a sequence
            (e.g. :obj:`[pooler, pooler, pooler]` or :obj:`["ndp", "ndp", "ndp"]`).
        input_key (str, optional):
            The key in the data object from which to read the graph data.
            If :obj:`None`, uses the default data object.
        output_key (str, optional):
            The key in the data object where the pooled graphs will be stored.
            Defaults to :obj:`"pooled_data"`.
    """

    def __init__(
        self,
        poolers: PoolersArg,
        input_key: Optional[str] = None,
        output_key: str = "pooled_data",
    ) -> None:
        super().__init__()
        self.input_key = input_key
        self.output_key = output_key

        levels_list = self._normalize_poolers_arg(poolers)
        if not levels_list:
            raise ValueError(
                "`poolers` must be a non-empty pooler, level config, or sequence."
            )

        # Resolve each level config into an instantiated pooler plus a
        # deterministic "collapse key" used to merge adjacent equal levels.
        self._resolved_level_entries = tuple(
            self._resolve_level_config_with_key(level_config)
            for level_config in levels_list
        )
        self.poolers = tuple(pooler for pooler, _ in self._resolved_level_entries)
        if not self.poolers:
            raise ValueError("At least one pooling level is required.")
        self._collapsed_level_runs = tuple(
            self._collapse_consecutive_runs(self._resolved_level_entries)
        )


    @staticmethod
    def _normalize_poolers_arg(
        poolers: PoolersArg,
    ) -> list[PoolerLevelConfig]:
        """Convert poolers (single pooler/config or sequence) to a list of level configs."""
        if isinstance(poolers, SRCPooling):
            return [poolers]
        if isinstance(poolers, str):
            return [poolers]
        if isinstance(poolers, dict):
            return [poolers]
        if (
            isinstance(poolers, tuple)
            and len(poolers) == 2
            and isinstance(poolers[1], (dict, type(None)))
        ):
            return [poolers]
        return list(poolers)

    @staticmethod
    def _build_pooler(pooler_name: str, kwargs: Optional[Dict[str, Any]] = None):
        """Instantiate a pooler from its registered alias and kwargs.

        Note:
            The import is intentionally local to avoid importing all poolers
            at module import time (which can trigger optional dependency
            imports even when not needed).
        """
        from tgp.poolers import get_pooler

        return get_pooler(pooler_name, **(kwargs or {}))

    @staticmethod
    def _freeze_config_value(value: Any) -> Any:
        """Convert nested config values into hashable, deterministic keys."""
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

    def _resolve_level_config_with_key(
        self, level_config: PoolerLevelConfig
    ) -> ResolvedLevelEntry:
        """Resolve one level config into `(pooler, collapse_key)`.

        Collapse key policy:
        - declarative configs (`str`, tuple, dict) collapse by normalized
          pooler name + kwargs;
        - instantiated objects collapse only by object identity.

        Examples:
            - ``"ndp"`` -> ``("config", "ndp", ())``
            - ``("eigen", {"k": 8})`` -> ``("config", "eigen", (("k", 8),))``
            - ``{"pooler": "sep", "s_inv_op": "inverse"}`` ->
              ``("config", "sep", (("s_inv_op", "inverse"),))``
            - ``NDPPooling()`` -> ``("instance", id(pooler_instance))``
        """
        if isinstance(level_config, dict):
            config_dict = dict(level_config)
            level_config = (
                config_dict.pop("pooler", config_dict.pop("name", None)),
                config_dict,
            )

        if isinstance(level_config, tuple):
            # Common config mistake: malformed tuples from CLI/YAML parsing.
            if len(level_config) != 2:
                raise ValueError(
                    "Tuple pooler configs must be '(pooler_or_name, kwargs_dict)'."
                )
            pooler_or_name = level_config[0]
            if pooler_or_name is None:
                raise ValueError(
                    "Pooler config must include a pooler name or instance."
                )
            pooler_kwargs = dict(level_config[1] or {})
            if isinstance(pooler_or_name, SRCPooling):
                if pooler_kwargs:
                    raise ValueError(
                        "Cannot provide kwargs together with an instantiated pooler."
                    )
                pooler = pooler_or_name
                collapse_key = ("instance", id(pooler))
            else:
                pooler_name = str(pooler_or_name).lower()
                pooler = self._build_pooler(pooler_name, pooler_kwargs)
                collapse_key = (
                    "config",
                    pooler_name,
                    self._freeze_config_value(pooler_kwargs),
                )
        elif isinstance(level_config, str):
            pooler_name = level_config.lower()
            pooler = self._build_pooler(pooler_name)
            collapse_key = ("config", pooler_name, ())
        else:
            # Keep error explicit instead of relying on obscure attribute errors
            # later in the coarsening pipeline.
            if not isinstance(level_config, SRCPooling):
                raise TypeError(
                    "Pooler config must be an SRCPooling, alias string, "
                    "('name', kwargs) tuple, or {'pooler'/'name', ...} dict."
                )
            pooler = level_config
            collapse_key = ("instance", id(pooler))

        if pooler.is_trainable:
            raise ValueError("The pooler must not be trainable.")
        return pooler, collapse_key

    @staticmethod
    def _collapse_consecutive_runs(
        entries: Sequence[ResolvedLevelEntry],
    ) -> list[CollapsedLevelRun]:
        """Collapse consecutive equal pooler configs into `(pooler, run_len)`.

        Example:
            Input keys:
            ``[("config", "ndp", ()), ("config", "ndp", ()), ("config", "sep", ())]``

            Output runs:
            ``[(ndp_pooler, 2), (sep_pooler, 1)]``
        """
        if len(entries) == 0:
            return []

        collapsed: list[CollapsedLevelRun] = []
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
        """Attach pooled levels to ``data[self.output_key]``.

        Execution is run-based: adjacent identical pooler configs are collapsed
        and each run is executed once via
        :meth:`tgp.src.Precoarsenable.multi_level_precoarsening`.
        Returned levels are still appended one-by-one, preserving the original
        external contract (`len(pooled_data) == number of requested levels`).

        Example:
            With ``poolers=["ndp", "ndp", "sep", "sep", "graclus"]``,
            internal execution collapses to runs
            ``[(ndp, 2), (sep, 2), (graclus, 1)]``.
            The output still contains five levels in order:
            ``data.pooled_data = [lvl1, lvl2, lvl3, lvl4, lvl5]``.
        """
        data_obj = data if self.input_key is None else getattr(data, self.input_key)
        pooled_levels = []
        for pooler, run_len in self._collapsed_level_runs:
            run_outputs = pooler.multi_level_precoarsening(
                levels=run_len,
                edge_index=data_obj.edge_index,
                edge_weight=getattr(data_obj, "edge_weight", None),
                batch=getattr(data_obj, "batch", None),
                num_nodes=data_obj.num_nodes,
            )
            if len(run_outputs) != run_len:
                raise ValueError(
                    f"{type(pooler).__name__}.multi_level_precoarsening returned "
                    f"{len(run_outputs)} levels, expected {run_len}."
                )

            for pooled_output in run_outputs:
                data_obj = pooled_output.as_data()
                pooled_levels.append(data_obj)

        setattr(data, self.output_key, pooled_levels)

        return data
