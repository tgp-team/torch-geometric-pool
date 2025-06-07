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

    Returns:
        ~torch.Tensor: The tensor of pooled node features.
    """
    if batch is None:
        return x.sum(dim=node_dim, keepdim=True)
    return scatter(x, batch, dim=node_dim, dim_size=size, reduce=reduce_op)


def dense_global_reduce(
    x: Tensor, reduce_op: ReduceType = "sum", node_dim: int = -2
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

    Returns:
        ~torch.Tensor: The tensor of pooled node features.
    """
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
