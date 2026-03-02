import warnings
from typing import Optional, Union

import torch
from torch import Tensor, nn

from .aggr_reduce import AggrReduce, _apply_mask
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


def readout(
    x: Tensor,
    reduce_op: Union[str, "PyGAggregation"] = "sum",
    batch: Optional[Tensor] = None,
    size: Optional[int] = None,
    mask: Optional[Tensor] = None,
    **aggr_kwargs,
) -> Tensor:
    r"""Graph-level readout: aggregate node features to one vector per graph.

    Infers sparse vs dense from ``x.ndim``: 2D ``[N, F]`` is sparse (use ``batch``
    for grouping); 3D ``[B, N, F]`` is dense (reduce over node dimension).
    Nodes must be on the second-to-last dimension.

    Args:
        x: Node features. Shape ``[N, F]`` (sparse) or ``[B, N, F]`` (dense).
        reduce_op: Aggregation: string (e.g. ``"sum"``, ``"mean"``, ``"max"``,
            ``"min"``, ``"lstm"``, ``"set2set"``) or a PyG Aggregation module.
            Strings are resolved via :func:`~tgp.reduce.get_aggr`.
        batch: Batch vector for sparse ``x``, shape ``[N]``. Ignored for dense.
        size: Number of graphs (for sparse). Optional.
        mask: Valid-node mask for batched (dense) ``x`` only, shape ``[B, N]``.
        **aggr_kwargs: Passed to :func:`~tgp.reduce.get_aggr` when ``reduce_op``
            is a string (e.g. ``in_channels``, ``out_channels``, ``processing_steps``).

    Returns:
        Tensor of shape ``[B, F]`` (or ``[1, F]`` for single graph sparse).
    """
    if x.dim() != 2 and x.dim() != 3:
        raise ValueError(
            f"readout expects x to be 2D [N, F] or 3D [B, N, F], got ndim={x.dim()}"
        )

    # Mask is only meaningful for batched (dense) representations; warn and ignore otherwise.
    if mask is not None:
        if x.dim() == 2:
            warnings.warn(
                "mask is only supported for batched (dense) representations; ignoring "
                "mask for 2D x (all nodes considered valid).",
                UserWarning,
                stacklevel=2,
            )
            mask = None
        elif x.dim() == 3 and (mask.dim() != 2 or mask.shape != x.shape[:2]):
            warnings.warn(
                "mask must be 2D with shape [B, N] matching x.shape[:2]; ignoring "
                "invalid mask (all nodes considered valid).",
                UserWarning,
                stacklevel=2,
            )
            mask = None

    if isinstance(reduce_op, str):
        aggr = get_aggr(reduce_op, **aggr_kwargs)
    elif _is_pyg_aggregation(reduce_op):
        aggr = reduce_op
    else:
        raise TypeError(
            f"reduce_op must be a string or a PyG Aggregation, got {type(reduce_op)}"
        )

    reducer = AggrReduce(aggr)
    reducer.to(x.device)

    # Dense masked readout: apply mask once here and delegate to sparse-style readout.
    if x.dim() == 3 and mask is not None:
        x_valid, batch_valid = _apply_mask(x, mask)
        B = mask.size(0)
        x_pool, _ = reducer(x_valid, so=None, batch=batch_valid, size=B)
        return x_pool

    # Unmasked or sparse-style paths.
    batch_in = batch
    if x.dim() == 2 and batch_in is None and x.size(0) > 0:
        batch_in = torch.zeros(x.size(0), dtype=torch.long, device=x.device)
    size_in = size if x.dim() == 2 else x.size(0)

    x_pool, _ = reducer(x, so=None, batch=batch_in, size=size_in)
    return x_pool
