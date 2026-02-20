from typing import Optional, Tuple, Union

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from tgp.utils.typing import ReduceType

from .aggr_reduce import AggrReduce
from .get_aggr import get_aggr

try:
    from torch_geometric.nn.aggr import Aggregation as PyGAggregation
except Exception:
    PyGAggregation = None


def _is_pyg_aggregation(reduce_op) -> bool:
    """Return True if reduce_op is a PyG Aggregation module."""
    if PyGAggregation is None:
        return False
    return isinstance(reduce_op, nn.Module) and isinstance(reduce_op, PyGAggregation)


def _apply_mask_sparse(
    x: Tensor,
    batch: Optional[Tensor],
    mask: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply mask to sparse x; return (x_masked, batch_masked) with invalid rows removed."""
    if mask is None:
        return x, batch
    x_masked = x * mask.unsqueeze(-1).to(x.dtype)
    valid = mask.nonzero(as_tuple=True)[0]
    x_masked = x_masked[valid]
    batch_masked = batch[valid] if batch is not None else None
    return x_masked, batch_masked


def _apply_mask_dense(
    x: Tensor,
    mask: Optional[Tensor],
) -> Tuple[Tensor, Optional[Tensor]]:
    """Apply mask to dense x; return (x_flat_valid, batch_flat_valid) for valid nodes only."""
    B, N, F = x.shape
    if mask is None:
        x_flat = x.reshape(B * N, F)
        batch_flat = torch.arange(
            B, device=x.device, dtype=torch.long
        ).repeat_interleave(N)
        return x_flat, batch_flat
    mask_flat = mask.reshape(-1)
    valid = mask_flat.nonzero(as_tuple=True)[0]
    x_flat = x.reshape(B * N, F)
    batch_flat = torch.arange(B, device=x.device, dtype=torch.long).repeat_interleave(N)
    return x_flat[valid], batch_flat[valid]


def readout(
    x: Tensor,
    reduce_op: Union[ReduceType, str, "PyGAggregation"] = "sum",
    batch: Optional[Tensor] = None,
    size: Optional[int] = None,
    mask: Optional[Tensor] = None,
    node_dim: int = -2,
    **aggr_kwargs,
) -> Tensor:
    r"""Graph-level readout: aggregate node features to one vector per graph.

    Infers sparse vs dense from ``x.ndim``: 2D ``[N, F]`` is sparse (use ``batch``
    for grouping); 3D ``[B, N, F]`` is dense (reduce over node dimension).
    When ``reduce_op`` is a string or a PyG
    :class:`torch_geometric.nn.aggr.Aggregation` instance, readout uses
    :class:`~tgp.reduce.AggrReduce` internally with :obj:`so=None` (one cluster per graph).

    Args:
        x: Node features. Shape ``[N, F]`` (sparse) or ``[B, N, F]`` (dense).
        reduce_op: Aggregation: string (e.g. ``"sum"``, ``"mean"``, ``"max"``,
            ``"min"``, ``"lstm"``, ``"set2set"``) or a PyG Aggregation module.
            Strings are resolved via :func:`~tgp.reduce.get_aggr`.
        batch: Batch vector for sparse ``x``, shape ``[N]``. Ignored for dense.
        size: Number of graphs (for sparse). Optional.
        mask: Valid-node mask. Sparse: shape ``[N]``; dense: shape ``[B, N]``.
            Only valid nodes are aggregated when provided.
        node_dim: Dimension along which nodes are aggregated (default ``-2``).
        **aggr_kwargs: Passed to :func:`~tgp.reduce.get_aggr` when ``reduce_op``
            is a string (e.g. ``in_channels``, ``out_channels``, ``processing_steps``).

    Returns:
        Tensor of shape ``[B, F]`` (or ``[1, F]`` for single graph sparse).
    """
    if node_dim != -2:
        x = x.transpose(node_dim, -2)
    if x.dim() != 2 and x.dim() != 3:
        raise ValueError(
            f"readout expects x to be 2D [N, F] or 3D [B, N, F], got ndim={x.dim()}"
        )

    # "any" has no PyG Aggregation equivalent; use scatter
    if isinstance(reduce_op, str) and reduce_op.strip().lower() == "any":
        if x.dim() == 2:
            if mask is not None:
                x = x * mask.unsqueeze(-1).to(x.dtype)
            if batch is None:
                return x.any(dim=node_dim, keepdim=True)
            return scatter(x.bool(), batch, dim=node_dim, dim_size=size, reduce="any")
        else:
            if mask is not None:
                x = x.masked_fill(~mask.unsqueeze(-1), False)
            return x.any(dim=node_dim)

    if isinstance(reduce_op, str):
        aggr = get_aggr(reduce_op, **aggr_kwargs)
    elif _is_pyg_aggregation(reduce_op):
        aggr = reduce_op
    else:
        raise TypeError(
            f"reduce_op must be a string or a PyG Aggregation, got {type(reduce_op)}"
        )

    reducer = AggrReduce(aggr)

    if x.dim() == 2:
        x_agg, batch_agg = _apply_mask_sparse(x, batch, mask)
        if batch_agg is None and x_agg.size(0) > 0:
            batch_agg = torch.zeros(x_agg.size(0), dtype=torch.long, device=x.device)
        x_pool, _ = reducer(x_agg, so=None, batch=batch_agg, size=size)
    else:
        x_flat, batch_flat = _apply_mask_dense(x, mask)
        x_pool, _ = reducer(x_flat, so=None, batch=batch_flat, size=x.size(0))

    return x_pool
