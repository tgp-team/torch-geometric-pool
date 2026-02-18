from typing import Optional, Union

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter

from tgp.utils.typing import ReduceType

try:
    from torch_geometric.nn.aggr import Aggregation as PyGAggregation
except Exception:
    PyGAggregation = None


def _is_pyg_aggregation(reduce_op) -> bool:
    """Return True if reduce_op is a PyG Aggregation module."""
    if PyGAggregation is None:
        return False
    return isinstance(reduce_op, nn.Module) and isinstance(reduce_op, PyGAggregation)


def _readout_sparse(
    x: Tensor,
    reduce_op: Union[ReduceType, "PyGAggregation"],
    batch: Optional[Tensor] = None,
    size: Optional[int] = None,
    node_dim: int = -2,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Sparse readout: x has shape [N, F]. Aggregate by batch (or single graph)."""
    if _is_pyg_aggregation(reduce_op):
        index = batch
        if index is None:
            index = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
        dim_size = size if size is not None else (int(index.max().item()) + 1)
        if mask is not None:
            x = x * mask.unsqueeze(-1).to(x.dtype)
        return reduce_op(x, index=index, dim_size=dim_size, dim=node_dim)

    if mask is not None:
        if reduce_op == "mean":
            x = x * mask.unsqueeze(-1).to(x.dtype)
            if batch is not None:
                count = scatter(
                    mask.to(x.dtype), batch, dim=0, dim_size=size, reduce="sum"
                )
                out = scatter(x, batch, dim=node_dim, dim_size=size, reduce="sum")
                return out / count.clamp(min=1).unsqueeze(-1)
        elif reduce_op in ("max", "min"):
            fill_val = float("-inf") if reduce_op == "max" else float("inf")
            x = x.masked_fill(~mask.unsqueeze(-1), fill_val)
        else:
            x = x * mask.unsqueeze(-1).to(x.dtype)
    if batch is None:
        return x.sum(dim=node_dim, keepdim=True)
    return scatter(x, batch, dim=node_dim, dim_size=size, reduce=reduce_op)


def _readout_dense(
    x: Tensor,
    reduce_op: Union[ReduceType, "PyGAggregation"],
    node_dim: int = -2,
    mask: Optional[Tensor] = None,
) -> Tensor:
    """Dense readout: x has shape [B, N, F]. Aggregate over node dimension."""
    if _is_pyg_aggregation(reduce_op):
        B, N, F = x.shape
        x_flat = x.reshape(B * N, F)
        index = torch.arange(B, device=x.device, dtype=torch.long).repeat_interleave(N)
        if mask is not None:
            mask_flat = mask.reshape(-1)
            valid = mask_flat.nonzero(as_tuple=True)[0]
            x_flat = x_flat[valid]
            index = index[valid]
        return reduce_op(x_flat, index=index, dim_size=B, dim=-2)

    if mask is None:
        if reduce_op == "sum":
            return x.sum(dim=node_dim)
        elif reduce_op == "mean":
            return x.mean(dim=node_dim)
        elif reduce_op == "max":
            return x.max(dim=node_dim).values
        elif reduce_op == "min":
            return x.min(dim=node_dim).values
        elif reduce_op == "any":
            return x.any(dim=node_dim)
        else:
            raise ValueError(f"Unsupported aggregation method: {reduce_op}")

    assert mask.ndim == 2, (
        f"dense readout expects mask with 2 dims [B, N], got ndim={mask.ndim}"
    )
    mask_expand = mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * mask_expand

    if reduce_op == "sum":
        return x_masked.sum(dim=node_dim)
    elif reduce_op == "mean":
        n_valid = mask.sum(dim=1, keepdim=True).clamp(min=1).to(x.dtype)
        return x_masked.sum(dim=node_dim) / n_valid
    elif reduce_op == "max":
        x_fill = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        out = x_fill.max(dim=node_dim).values
        return out.masked_fill(out == float("-inf"), 0)
    elif reduce_op == "min":
        x_fill = x.masked_fill(~mask.unsqueeze(-1), float("inf"))
        out = x_fill.min(dim=node_dim).values
        return out.masked_fill(out == float("inf"), 0)
    elif reduce_op == "any":
        x_fill = x.masked_fill(~mask.unsqueeze(-1), False)
        return x_fill.any(dim=node_dim)
    else:
        raise ValueError(f"Unsupported aggregation method: {reduce_op}")


def readout(
    x: Tensor,
    reduce_op: Union[ReduceType, "PyGAggregation"] = "sum",
    batch: Optional[Tensor] = None,
    size: Optional[int] = None,
    mask: Optional[Tensor] = None,
    node_dim: int = -2,
) -> Tensor:
    r"""Graph-level readout: aggregate node features to one vector per graph.

    Infers sparse vs dense from ``x.ndim``: 2D ``[N, F]`` is sparse (use ``batch``
    for grouping); 3D ``[B, N, F]`` is dense (reduce over node dimension).
    Supports both string reduce ops (e.g. ``"sum"``, ``"mean"``) and PyG
    :class:`torch_geometric.nn.aggr.Aggregation` instances.

    Args:
        x: Node features. Shape ``[N, F]`` (sparse) or ``[B, N, F]`` (dense).
        reduce_op: Aggregation: string (e.g. ``"sum"``, ``"mean"``, ``"max"``,
            ``"min"``, ``"any"``) or a PyG Aggregation module.
        batch: Batch vector for sparse ``x``, shape ``[N]``. Ignored for dense.
        size: Number of graphs (for sparse). Optional.
        mask: Valid-node mask. Sparse: shape ``[N]``; dense: shape ``[B, N]``.
        node_dim: Dimension along which nodes are aggregated (default ``-2``).

    Returns:
        Tensor of shape ``[B, F]`` (or ``[1, F]`` for single graph sparse).
    """
    if x.dim() == 2:
        return _readout_sparse(x, reduce_op, batch, size, node_dim, mask)
    if x.dim() == 3:
        return _readout_dense(x, reduce_op, node_dim, mask)
    raise ValueError(
        f"readout expects x to be 2D [N, F] or 3D [B, N, F], got ndim={x.dim()}"
    )
