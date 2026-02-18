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


class AggrReduce(Reduce):
    r"""Reduce operator that wraps a PyG :class:`torch_geometric.nn.aggr.Aggregation`.

    Aggregates node features within each cluster using the given aggregation
    module. Supports sparse assignment matrices; for dense assignments only
    hard (one cluster per node) is supported via an index built from
    :obj:`so.s.argmax(dim=-1)`.

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
        so: SelectOutput,
        *,
        batch: Optional[Tensor] = None,
        return_batched: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
            x_pool = self.aggr(
                src,
                index=so.cluster_index,
                dim_size=so.num_supernodes,
                dim=0,
            )
        else:
            # Dense: support only hard assignment (argmax) so we can build an index
            warnings.warn(
                "AggrReduce received a dense SelectOutput; using argmax to obtain "
                "hard cluster assignment. For soft assignments use BaseReduce instead.",
                UserWarning,
                stacklevel=2,
            )
            if so.s.dim() == 3:
                # [B, N, K] -> cluster index per node per graph
                cluster = so.s.argmax(dim=-1)  # [B, N]
                B, N, F = x.shape
                x_flat = x.reshape(B * N, F)
                # Global cluster index: graph b, node n -> output index b*K + cluster[b,n]
                K = so.s.size(-1)
                batch_idx = torch.arange(B, device=x.device).view(-1, 1).expand(-1, N)
                cluster_flat = cluster.reshape(-1)
                batch_flat = batch_idx.reshape(-1)
                index = batch_flat * K + cluster_flat
                x_pool = self.aggr(x_flat, index=index, dim_size=B * K, dim=-2)
                x_pool = x_pool.reshape(B, K, F)
                if not return_batched:
                    x_pool = x_pool.reshape(B * K, F)
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
                        x_pool_i = self.aggr(
                            x_i,
                            index=cluster_i,
                            dim_size=s_i.size(-1),
                            dim=0,
                        )
                        x_pool_list.append(x_pool_i)
                    if return_batched:
                        x_pool = torch.stack(x_pool_list, dim=0)
                    else:
                        x_pool = torch.cat(x_pool_list, dim=0)
                else:
                    cluster = so.s.argmax(dim=-1)
                    x_pool = self.aggr(x, index=cluster, dim_size=so.s.size(-1), dim=0)
            else:
                cluster = so.s.argmax(dim=-1)
                x_pool = self.aggr(x, index=cluster, dim_size=so.s.size(-1), dim=0)
            if return_batched and x_pool.dim() == 2:
                x_pool = x_pool.unsqueeze(0)

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(aggr={self.aggr})"
