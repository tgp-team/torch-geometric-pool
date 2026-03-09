from typing import Optional, Tuple

import torch
from torch import Tensor

from tgp.select import SelectOutput
from tgp.utils.ops import build_pooled_batch

from .base_reduce import Reduce
from .get_aggr import has_pyg_aggregation, is_pyg_aggregation


def _sort_by_cluster_index(src: Tensor, cluster_index: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Sort ``(src, cluster_index)`` by index for aggrs that require sorted input.

    Note:
        In :class:`~tgp.select.SelectOutput`, the index of pooled nodes is
        exposed as ``cluster_index``. In this module, "cluster" and
        "supernode" refer to the same concept.
    """
    cluster_index_sorted, perm = torch.sort(cluster_index, stable=True)
    src_sorted = src[perm]
    return src_sorted, cluster_index_sorted


def _aggregate_sorted(aggr, src: Tensor, index: Tensor, dim_size: int) -> Tensor:
    """Aggregate after sorting by index for aggrs that require sorted input."""
    src_sorted, index_sorted = _sort_by_cluster_index(src, index)
    return aggr(src_sorted, index=index_sorted, dim_size=dim_size, dim=0)


class AggrReduce(Reduce):
    r"""Reduce operator that wraps a PyG :class:`torch_geometric.nn.aggr.Aggregation`.

    Aggregates node features within each supernode using the given aggregation module.
    Supports sparse assignment matrices and graph-level readout mode (``so=None``).
    Dense :class:`~tgp.select.SelectOutput` assignments are not supported:
    use :class:`~tgp.reduce.BaseReduce` for dense/soft reductions.

    In :class:`~tgp.select.SelectOutput`, pooled-node indices are named
    ``cluster_index`` for historical reasons; here this is equivalent to
    "supernode index".

    ``so=None`` is supported for graph-level readout: all nodes are assigned to one
    supernode per graph (using ``batch`` as supernode index), or to a single supernode
    when ``batch`` is :obj:`None`.

    Args:
        aggr: A PyG Aggregation instance (e.g. :class:`torch_geometric.nn.aggr.SumAggregation`,
            :class:`torch_geometric.nn.aggr.MeanAggregation`).
    """

    def __init__(self, aggr):
        super().__init__()
        if not has_pyg_aggregation():
            raise ImportError(
                "AggrReduce requires torch_geometric.nn.aggr. "
                "Install PyTorch Geometric to use PyG aggregations."
            )
        if not is_pyg_aggregation(aggr):
            raise TypeError(f"aggr must be a PyG Aggregation, got {type(aggr)}")
        self.aggr = aggr

    def forward(
        self,
        x: Tensor,
        so: Optional[SelectOutput] = None,
        *,
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Aggregate node features according to the supernode assignments.

        Args:
            x (~torch.Tensor): Node features of shape :math:`[N, F]` or
                :math:`[B, N, F]`.
            so (~tgp.select.SelectOutput, optional): Select output containing
                assignment information. If :obj:`None`, performs graph-level
                readout using ``batch``.
            batch (~torch.Tensor, optional): Batch vector assigning each node
                to a graph.
            size (int, optional): Expected number of pooled nodes (readout
                groups). If :obj:`None`, inferred from ``so`` or
                ``batch``.

        Returns:
            tuple: A pair ``(x_pool, batch_pool)`` with pooled features and
            pooled batch indices.
        """
        # Path 1: readout mode (`so=None`) aggregates directly to one output per graph.
        if so is None:
            return self._readout_without_select_output(x, batch=batch, size=size)

        if batch is None and so.batch is not None:
            batch = so.batch

        # Path 2: sparse assignment matrix.
        if so.s.is_sparse:
            src = x[so.node_index] * so.weight.view(-1, 1)
            x_pool = _aggregate_sorted(
                self.aggr, src, so.cluster_index, dim_size=so.num_supernodes
            )
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        raise ValueError(
            "AggrReduce supports only sparse SelectOutput assignments. "
            "Dense assignments are not supported; use BaseReduce for dense/soft reductions."
        )

    def _readout_without_select_output(
        self,
        x: Tensor,
        *,
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Readout fast-path for ``so=None`` (one pooled supernode per graph)."""
        if x.dim() == 3:
            B, N, _ = x.shape
            num_supernodes = size if size is not None else B
            x_pool = _aggregate_sorted(
                self.aggr,
                x.reshape(-1, x.size(-1)),
                build_pooled_batch(B, N, x.device),
                dim_size=num_supernodes,
            )
            batch_pool = torch.arange(num_supernodes, device=x.device)
            return x_pool, batch_pool

        if x.dim() != 2:
            raise ValueError(
                "Readout mode expects x to be 2D [N, F] or 3D [B, N, F], "
                f"got ndim={x.dim()}."
            )

        if batch is None:
            cluster_index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
            x_pool = _aggregate_sorted(self.aggr, x, cluster_index, dim_size=1)
            return x_pool, None

        if batch.numel() > 0:
            inferred_num_supernodes = int(batch.max().item()) + 1
        else:
            # Preserve explicit graph cardinality (dim_size) even when there
            # are no real nodes (e.g. dense readout with all-false mask).
            inferred_num_supernodes = 1

        num_supernodes = size if size is not None else inferred_num_supernodes
        x_pool = _aggregate_sorted(self.aggr, x, batch, dim_size=num_supernodes)
        batch_pool = torch.arange(num_supernodes, device=batch.device)
        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr})"
