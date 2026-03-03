from typing import Optional, Union

from torch import Tensor

from tgp.utils.ops import apply_dense_node_mask

from .aggr_reduce import AggrReduce
from .get_aggr import resolve_reduce_op


def _validate_dense_mask(mask: Optional[Tensor], x: Tensor) -> None:
    """Validate dense readout mask shape against :obj:`x`."""
    if mask is None:
        return
    if mask.dim() != 2 or tuple(mask.shape) != tuple(x.shape[:2]):
        raise ValueError(
            "mask must have shape [B, N] matching x.shape[:2] for dense readout."
        )


def readout(
    x: Tensor,
    reduce_op: Union[str, object] = "sum",
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
        size: Number of graphs for sparse readout when :obj:`batch` is provided.
            Passing :obj:`size` with sparse ``x`` and :obj:`batch=None` raises
            :class:`ValueError`.
        mask: Input-node validity mask for batched (dense) ``x`` only, shape ``[B, N]``.
            Passing :obj:`mask` with sparse ``x`` or with a mismatched shape raises
            :class:`ValueError`.
        **aggr_kwargs: Passed to :func:`~tgp.reduce.get_aggr` when ``reduce_op``
            is a string (e.g. ``in_channels``, ``out_channels``, ``processing_steps``).

    Returns:
        Tensor of shape ``[B, F]`` (or ``[1, F]`` for single graph sparse).
    """
    if x.dim() not in (2, 3):
        raise ValueError(
            f"readout expects x to be 2D [N, F] or 3D [B, N, F], got ndim={x.dim()}"
        )

    aggr = resolve_reduce_op(reduce_op, **aggr_kwargs)
    reducer = AggrReduce(aggr)
    reducer.to(x.device)

    # Path 1: dense masked readout [B, N, F] + [B, N].
    if x.dim() == 3 and mask is not None:
        _validate_dense_mask(mask, x)
        x_valid, batch_valid = apply_dense_node_mask(x, mask)
        B = mask.size(0)
        x_pool, _ = reducer(x_valid, so=None, batch=batch_valid, size=B)
        return x_pool

    # Path 2: dense unmasked readout [B, N, F].
    if x.dim() == 3:
        x_pool, _ = reducer(x, so=None, batch=None, size=x.size(0))
        return x_pool

    # Path 3: sparse-style readout [N, F] (+ optional batch vector).
    if mask is not None:
        raise ValueError("mask is only supported for dense x with shape [B, N, F].")

    if batch is None and size is not None:
        raise ValueError(
            "size is only supported for sparse readout when batch is provided."
        )

    x_pool, _ = reducer(x, so=None, batch=batch, size=size)
    return x_pool
