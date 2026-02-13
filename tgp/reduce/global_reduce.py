from typing import Optional

from torch import Tensor
from torch_geometric.utils import scatter

from tgp.utils.typing import ReduceType


def global_reduce(
    x: Tensor,
    reduce_op: ReduceType = "sum",
    batch: Optional[Tensor] = None,
    size: Optional[int] = None,
    node_dim: int = -2,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""Global pooling operation for sparse methods.

    Args:
        x (~torch.Tensor):
            The input tensor of shape :math:`[N, F]`,
            where :math:`N` is the number of nodes in the batch and
            :math:`F` is the number of node features.
        reduce_op (~tgp.utils.typing.ReduceType, optional):
            The aggregation method to use.
            (default: :obj:`"sum"`)
        batch (~torch.Tensor, optional):
            The batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
            to which graph in the batch each node belongs. (default: :obj:`None`)
        size (int, optional):
            The number of nodes in the graph.
            (default: :obj:`None`)
        node_dim (int, optional): The node dimension.
            (default: :obj:`-2`)
        mask (~torch.Tensor, optional): Boolean mask of shape :math:`[N]`
            indicating valid nodes. When provided, only valid nodes are aggregated.
            (default: :obj:`None`)

    Returns:
        ~torch.Tensor: The tensor of pooled node features.
    """
    if mask is not None:
        if reduce_op == "mean":
            x = x * mask.unsqueeze(-1).to(x.dtype)
            if batch is not None:
                count = scatter(
                    mask.to(x.dtype), batch, dim=node_dim, dim_size=size, reduce="sum"
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


def dense_global_reduce(
    x: Tensor,
    reduce_op: ReduceType = "sum",
    node_dim: int = -2,
    mask: Optional[Tensor] = None,
) -> Tensor:
    r"""Global pooling operation for dense methods.

    Args:
        x (~torch.Tensor): The input tensor of shape :math:`[B, N, F]`,
            where :math:`B` is the batch size, :math:`N` is the number
            of nodes in the batch and :math:`F` is the number of node features.
        reduce_op (ReduceType, optional): The aggregation method to use.
            (default: :obj:`"sum"`)
        node_dim (int, optional): The node dimension.
            (default: :obj:`-2`)
        mask (~torch.Tensor, optional): Boolean mask of shape :math:`[B, N]`
            indicating valid nodes. When provided, only valid positions are
            aggregated (padding is ignored). (default: :obj:`None`)

    Returns:
        ~torch.Tensor: The tensor of pooled node features.
    """
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

    # Masked reduction: expand mask to feature dim for broadcasting
    mask_expand = mask.unsqueeze(-1).to(x.dtype)
    x_masked = x * mask_expand

    if reduce_op == "sum":
        return x_masked.sum(dim=node_dim)
    elif reduce_op == "mean":
        n_valid = mask.sum(dim=node_dim, keepdim=True).clamp(min=1).to(x.dtype)
        return x_masked.sum(dim=node_dim) / n_valid
    elif reduce_op == "max":
        x_fill = x.masked_fill(~mask.unsqueeze(-1), float("-inf"))
        out = x_fill.max(dim=node_dim).values
        # If a row has no valid nodes, max gives -inf; replace with 0 for safety
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
