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
        if so is None:
            so = self._one_cluster_per_graph_so(x, batch=batch, size=size)
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
                src_sorted,
                index=index_sorted,
                dim_size=so.num_supernodes,
                dim=0,
            )
        else:
            warnings.warn(
                "AggrReduce received a dense SelectOutput; using argmax to obtain "
                "hard cluster assignment. For soft assignments use BaseReduce instead.",
                UserWarning,
                stacklevel=2,
            )
            if so.s.dim() == 3:
                cluster = so.s.argmax(dim=-1)
                B, N, F_in = x.shape
                K = so.s.size(-1)
                x_flat = x.reshape(B * N, F_in)
                batch_idx = torch.arange(B, device=x.device).view(-1, 1).expand(-1, N)
                index = batch_idx.reshape(-1) * K + cluster.reshape(-1)
                # Sort by index for aggrs that require it (e.g. LSTMAggregation)
                x_flat_sorted, index_sorted = _sort_by_cluster_index(x_flat, index)
                x_pool = self.aggr(
                    x_flat_sorted, index=index_sorted, dim_size=B * K, dim=-2
                )
                F_out = x_pool.size(-1)
                x_pool = x_pool.reshape(B, K, F_out)
                # Dense in => dense out: do not flatten when input was 3D
            elif batch is not None and batch.numel() > 0:
                from torch_geometric.utils import unbatch

                batch_min = int(batch.min().item())
                batch_max = int(batch.max().item())
                if batch_min != batch_max:
                    unbatched_s = unbatch(so.s, batch)
                    unbatched_x = unbatch(x, batch)
                    x_pool_list = []
                    for s_i, x_i in zip(unbatched_s, unbatched_x):
                        cluster_i = s_i.argmax(dim=-1)
                        src_i = x_i
                        src_sorted, index_sorted = _sort_by_cluster_index(
                            src_i, cluster_i
                        )
                        x_pool_i = self.aggr(
                            src_sorted,
                            index=index_sorted,
                            dim_size=s_i.size(-1),
                            dim=0,
                        )
                        x_pool_list.append(x_pool_i)
                    x_pool = torch.stack(x_pool_list, dim=0)
                else:
                    cluster = so.s.argmax(dim=-1)
                    src_sorted, index_sorted = _sort_by_cluster_index(x, cluster)
                    x_pool = self.aggr(
                        src_sorted,
                        index=index_sorted,
                        dim_size=so.s.size(-1),
                        dim=0,
                    )
            else:
                cluster = so.s.argmax(dim=-1)
                src_sorted, index_sorted = _sort_by_cluster_index(x, cluster)
                x_pool = self.aggr(
                    src_sorted,
                    index=index_sorted,
                    dim_size=so.s.size(-1),
                    dim=0,
                )
            if x.dim() == 3 and x_pool.dim() == 2:
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
            if batch is not None and batch.numel() > 0:
                cluster_index = batch
                num_supernodes = (
                    size if size is not None else int(batch.max().item()) + 1
                )
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
