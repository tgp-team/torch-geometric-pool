import warnings
from typing import Optional, Tuple

import torch
from torch import Tensor

from tgp.select import SelectOutput

from .base_reduce import Reduce

try:
    from torch_geometric.nn.aggr import Aggregation as PyGAggregation
except Exception:
    PyGAggregation = None


def _sort_by_cluster_index(src: Tensor, cluster_index: Tensor) -> Tuple[Tensor, Tensor]:
    """Sort (src, cluster_index) by cluster_index for aggrs that require sorted index (e.g. LSTM)."""
    cluster_index_sorted, perm = torch.sort(cluster_index, stable=True)
    src_sorted = src[perm]
    return src_sorted, cluster_index_sorted


def _apply_mask(
    x: Tensor,
    mask: Tensor,
) -> Tuple[Tensor, Tensor]:
    r"""Apply a node mask to dense inputs.

    Assumes:
        - :obj:`x` has shape ``[B, N, F]``
        - :obj:`mask` has shape ``[B, N]``

    Returns:
        ``(x_valid, batch_valid)`` where both are flattened over valid (unmasked)
        nodes only, suitable for sparse-style aggregation.
    """
    if x.dim() != 3:
        raise ValueError(
            f"_apply_mask expects x to be 3D [B, N, F], got ndim={x.dim()}"
        )
    if mask.dim() != 2 or tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError(
            f"_apply_mask expects mask shape [B, N]={tuple(x.shape[:2])}, "
            f"got {tuple(mask.shape)}"
        )

    B, N, F = x.shape
    mask_flat = mask.reshape(-1)
    valid = mask_flat.nonzero(as_tuple=True)[0]

    x_flat = x.reshape(B * N, F)
    batch_flat = torch.arange(B, device=x.device, dtype=torch.long).repeat_interleave(N)

    return x_flat[valid], batch_flat[valid]


class AggrReduce(Reduce):
    r"""Reduce operator that wraps a PyG :class:`torch_geometric.nn.aggr.Aggregation`.

    Aggregates node features within each cluster using the given aggregation
    module. Supports sparse assignment matrices; for dense assignments only
    hard (one cluster per node) is supported via an index built from
    :obj:`so.s.argmax(dim=-1)` (a warning is raised).

    When assignment is dense, output shape follows input: 3D in :math:`\Rightarrow` 3D out,
    2D in :math:`\Rightarrow` 2D out. :obj:`return_batched` applies only to the sparse
    (unbatched) path.

    :obj:`so=None` is supported for graph-level readout: all nodes are assigned to one
    cluster per graph (using :obj:`batch` as cluster index), or to a single cluster
    when :obj:`batch` is :obj:`None`.

    Node-level :obj:`mask` is only supported for batched (dense) representations:
    use :obj:`mask=None` for 2D ``x``; for 3D ``x``, mask has shape ``[B, N]``.

    Args:
        aggr: A PyG Aggregation instance (e.g. :class:`torch_geometric.nn.aggr.SumAggregation`,
            :class:`torch_geometric.nn.aggr.MeanAggregation`).
    """

    def __init__(self, aggr: "PyGAggregation"):
        super().__init__()
        if PyGAggregation is None:
            raise ImportError(
                "AggrReduce requires torch_geometric.nn.aggr. "
                "Install PyTorch Geometric to use PyG aggregations."
            )
        if not isinstance(aggr, PyGAggregation):
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
        # so=None is treated as global readout: one cluster per graph.
        if so is None:
            so = self._one_cluster_per_graph_so(x, batch=batch, size=size)
            # For dense [B, N, F] inputs we flatten to sparse-style [B*N, F]
            # and let cluster_index encode graph membership.
            if x.dim() == 3:
                x = x.reshape(-1, x.size(-1))
                batch = so.cluster_index

        if batch is None and hasattr(so, "batch") and so.batch is not None:
            batch = so.batch

        if so.s.is_sparse:
            if return_batched:
                raise ValueError(
                    "return_batched=True is only supported for dense assignment matrices."
                )
            src = x[so.node_index]
            values = so.s.values()
            if values is not None:
                src = src * values.view(-1, 1)
            # Sort by cluster_index for aggrs that require it (e.g. LSTMAggregation)
            src_sorted, index_sorted = _sort_by_cluster_index(src, so.cluster_index)
            x_pool = self.aggr(
                src_sorted, index=index_sorted, dim_size=so.num_supernodes, dim=0
            )
        else:
            warnings.warn(
                "AggrReduce received a dense SelectOutput; using argmax to obtain "
                "hard cluster assignment. For soft assignments use BaseReduce instead.",
                UserWarning,
                stacklevel=2,
            )
            if so.s.dim() == 3:
                # Batched dense assignment: s [B, N, K], x [B, N, F]
                B, N, F_in = x.shape
                K = so.s.size(-1)
                cluster = so.s.argmax(dim=-1)  # [B, N]

                # Build per-node graph ids for global indexing.
                batch_idx = (
                    torch.arange(B, device=x.device, dtype=torch.long)
                    .view(-1, 1)
                    .expand(-1, N)
                )

                in_mask = getattr(so, "in_mask", None)
                if in_mask is not None:
                    # Use shared helper to restrict x to valid nodes.
                    x_flat, batch_flat = _apply_mask(x, in_mask)
                    # Rebuild cluster indices on the same valid positions.
                    cluster_flat = cluster.reshape(-1)
                    mask_flat = in_mask.reshape(-1)
                    valid = mask_flat.nonzero(as_tuple=True)[0]
                    cluster_flat = cluster_flat[valid]
                else:
                    x_flat = x.reshape(B * N, F_in)
                    batch_flat = batch_idx.reshape(-1)
                    cluster_flat = cluster.reshape(-1)

                # Global cluster index encodes (graph, cluster) pairs.
                global_cluster = batch_flat * K + cluster_flat

                # Sort by index for aggrs that require it (e.g. LSTMAggregation)
                src_sorted, index_sorted = _sort_by_cluster_index(
                    x_flat, global_cluster
                )
                x_pool_flat = self.aggr(
                    src_sorted, index=index_sorted, dim_size=B * K, dim=0
                )
                F_out = x_pool_flat.size(-1)
                x_pool = x_pool_flat.reshape(B, K, F_out)
                # Dense in => dense out: return [B, K, F]
            elif batch is not None and batch.numel() > 0:
                # Dense [N, K] with batch: flatten to global cluster index, one aggr.
                cluster = so.s.argmax(dim=-1)  # [N]
                K = so.s.size(-1)
                B = int(batch.max().item()) + 1
                global_cluster = batch * K + cluster
                src_sorted, index_sorted = _sort_by_cluster_index(x, global_cluster)
                x_pool_flat = self.aggr(
                    src_sorted,
                    index=index_sorted,
                    dim_size=B * K,
                    dim=0,
                )
                F_out = x_pool_flat.size(-1)
                if return_batched:
                    x_pool = x_pool_flat.reshape(B, K, F_out)
                else:
                    x_pool = x_pool_flat
            else:
                cluster = so.s.argmax(dim=-1)
                src_sorted, index_sorted = _sort_by_cluster_index(x, cluster)
                x_pool = self.aggr(
                    src_sorted,
                    index=index_sorted,
                    dim_size=so.s.size(-1),
                    dim=0,
                )
            if return_batched and x_pool.dim() == 2:
                x_pool = x_pool.unsqueeze(0)

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def _one_cluster_per_graph_so(
        self,
        x: Tensor,
        *,
        batch: Optional[Tensor] = None,
        size: Optional[int] = None,
    ) -> SelectOutput:
        """Build a sparse SelectOutput that assigns every node to one cluster per graph."""
        if x.dim() == 3:
            B, N, _ = x.shape
            num_nodes = B * N
            cluster_index = torch.arange(
                B, dtype=torch.long, device=x.device
            ).repeat_interleave(N)
            num_supernodes = size if size is not None else B
        else:
            num_nodes = x.size(0)
            if batch is not None:
                cluster_index = batch
                if batch.numel() > 0:
                    inferred = int(batch.max().item()) + 1
                    num_supernodes = size if size is not None else inferred
                else:
                    # Preserve explicit graph cardinality (dim_size) even when
                    # there are no valid nodes (e.g. dense readout with all-false mask).
                    num_supernodes = size if size is not None else 1
            else:
                cluster_index = torch.zeros(
                    num_nodes, dtype=torch.long, device=x.device
                )
                num_supernodes = 1
        return SelectOutput(
            cluster_index=cluster_index,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr})"
