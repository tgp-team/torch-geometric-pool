import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from tgp.select import SelectOutput
from tgp.utils.ops import build_pooled_batch, is_multi_graph_batch

from .base_reduce import Reduce
from .get_aggr import has_pyg_aggregation, is_pyg_aggregation


def _sort_by_cluster_index(src: Tensor, cluster_index: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Sort ``(src, cluster_index)`` by index for aggrs that require sorted input.

    Note:
        In :class:`~tgp.select.SelectOutput`, the index of pooled nodes is
        exposed as :obj:`cluster_index`. In this module, "cluster" and
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

    Aggregates node features within each supernode using the given aggregation
    module. Supports sparse assignment matrices; for dense assignments only
    hard (one supernode per node) is supported via an index built from
    :obj:`so.s.argmax(dim=-1)` (a warning is raised).

    In :class:`~tgp.select.SelectOutput`, pooled-node indices are named
    :obj:`cluster_index` for historical reasons; here this is equivalent to
    "supernode index".

    When assignment is dense, output shape follows input: 3D in :math:`\Rightarrow` 3D out,
    2D in :math:`\Rightarrow` 2D out by default. For dense unbatched
    :math:`[N, K]` multi-graph batches, :obj:`return_batched=True` returns
    :math:`[B, K, F]` instead of :math:`[B \cdot K, F]`.

    :obj:`so=None` is supported for graph-level readout: all nodes are assigned to one
    supernode per graph (using :obj:`batch` as supernode index), or to a single supernode
    when :obj:`batch` is :obj:`None`.

    Node validity masking is provided through :obj:`so.in_mask` (inside
    :class:`~tgp.select.SelectOutput`), not as a direct :obj:`mask` argument.
    For dense batched assignments, :obj:`so.in_mask` has shape ``[B, N]``.

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
        return_batched: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        # Path 0: readout mode (`so=None`) aggregates directly to one output per graph.
        if so is None:
            if return_batched:
                raise ValueError(
                    "return_batched=True is only supported for dense assignment matrices."
                )
            return self._readout_without_select_output(x, batch=batch, size=size)

        if batch is None and so.batch is not None:
            batch = so.batch

        # Path 1: sparse assignment matrix (native sparse reduce behavior).
        if so.s.is_sparse:
            if return_batched:
                raise ValueError(
                    "return_batched=True is only supported for dense assignment matrices."
                )
            src = x[so.node_index] * so.weight.view(-1, 1)
            x_pool = _aggregate_sorted(
                self.aggr, src, so.cluster_index, dim_size=so.num_supernodes
            )
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        warnings.warn(
            "AggrReduce received a dense SelectOutput; using argmax to obtain "
            "hard supernode assignment. For soft assignments use BaseReduce instead.",
            UserWarning,
            stacklevel=2,
        )

        # Path 2: dense batched assignment [B, N, K].
        if so.s.dim() == 3:
            B, N, F_in = x.shape
            K = so.s.size(-1)

            supernode_flat = so.s.argmax(dim=-1).reshape(-1)  # [B*N]
            x_flat = x.reshape(B * N, F_in)  # [B*N, F]
            batch_flat = build_pooled_batch(B, N, x.device)  # [B*N]

            if so.in_mask is not None:
                valid = so.in_mask.reshape(-1).nonzero(as_tuple=True)[0]
                x_flat = x_flat[valid]
                batch_flat = batch_flat[valid]
                supernode_flat = supernode_flat[valid]

            global_supernode = batch_flat * K + supernode_flat  # [B*N_valid]
            x_pool_flat = _aggregate_sorted(
                self.aggr,
                x_flat,
                global_supernode,
                dim_size=B * K,
            )
            F_out = x_pool_flat.size(-1)
            x_pool = x_pool_flat.reshape(B, K, F_out)  # dense in => dense out

            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        if so.s.dim() != 2:
            raise ValueError(
                "Dense SelectOutput.s must be 2D [N, K] or 3D [B, N, K], "
                f"got ndim={so.s.dim()}."
            )

        supernode = so.s.argmax(dim=-1)  # [N]
        K = so.s.size(-1)

        # Path 3: dense unbatched assignment [N, K] with multi-graph batch.
        if is_multi_graph_batch(batch):
            B = int(batch.max().item()) + 1
            global_supernode = batch * K + supernode
            x_pool_flat = _aggregate_sorted(
                self.aggr, x, global_supernode, dim_size=B * K
            )
            F_out = x_pool_flat.size(-1)
            x_pool = x_pool_flat.reshape(B, K, F_out) if return_batched else x_pool_flat
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        # Path 4: dense unbatched assignment [N, K] for a single graph.
        x_pool = _aggregate_sorted(self.aggr, x, supernode, dim_size=K)
        if return_batched:
            x_pool = x_pool.unsqueeze(0)

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def _readout_without_select_output(
        self,
        x: Tensor,
        *,
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Readout fast-path for :obj:`so=None` (one pooled supernode per graph)."""
        if x.dim() == 3:
            B, N, _ = x.shape
            cluster_index = build_pooled_batch(B, N, x.device)
            num_supernodes = size if size is not None else B
            x_flat = x.reshape(-1, x.size(-1))
            x_pool = _aggregate_sorted(
                self.aggr,
                x_flat,
                cluster_index,
                dim_size=num_supernodes,
            )
            batch_pool = torch.arange(num_supernodes, device=x.device)
            return x_pool, batch_pool

        if x.dim() != 2:
            raise ValueError(
                "Readout mode expects x to be 2D [N, F] or 3D [B, N, F], "
                f"got ndim={x.dim()}."
            )

        if batch is not None:
            cluster_index = batch
            if batch.numel() > 0:
                inferred = int(batch.max().item()) + 1
                num_supernodes = size if size is not None else inferred
            else:
                # Preserve explicit graph cardinality (dim_size) even when there
                # are no real nodes (e.g. dense readout with all-false mask).
                num_supernodes = size if size is not None else 1
            x_pool = _aggregate_sorted(
                self.aggr,
                x,
                cluster_index,
                dim_size=num_supernodes,
            )
            batch_pool = torch.arange(num_supernodes, device=batch.device)
            return x_pool, batch_pool

        num_nodes = x.size(0)
        cluster_index = torch.zeros(num_nodes, dtype=torch.long, device=x.device)
        x_pool = _aggregate_sorted(self.aggr, x, cluster_index, dim_size=1)
        return x_pool, None

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr})"
