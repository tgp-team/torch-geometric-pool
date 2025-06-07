from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tgp.imports import SparseTensor


def rank3_trace(x):
    return torch.einsum("ijj->i", x)


def rank3_diag(x):
    return torch.diag_embed(x)


def connectivity_to_row_col(edge_index: Adj) -> Tuple[Tensor, Tensor]:
    if isinstance(edge_index, Tensor):
        return edge_index[0], edge_index[1]
    elif isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        return row, col
    else:
        raise NotImplementedError()


def connectivity_to_edge_index(
    edge_index: Adj, edge_weight: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        return edge_index, edge_weight
    elif isinstance(edge_index, SparseTensor):
        row, col, edge_weight = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_weight
    else:
        raise NotImplementedError()


def connectivity_to_sparse_tensor(
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> SparseTensor:
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(edge_index, SparseTensor):
        return edge_index
    elif isinstance(edge_index, Tensor):
        adj = SparseTensor.from_edge_index(
            edge_index, edge_weight, (num_nodes, num_nodes)
        )
        return adj
    else:
        raise NotImplementedError()


def pseudo_inverse(edge_index: Adj) -> Tuple[Adj, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):  # Dense pooling
        adj_inv = torch.linalg.pinv(edge_index)
        return adj_inv
    elif isinstance(edge_index, SparseTensor):  # Sparse pooling
        adj = edge_index
    else:
        raise NotImplementedError()
    adj_inv = torch.linalg.pinv(adj.to_dense().float())
    adj_inv = torch.where(torch.abs(adj_inv) < 1e-5, torch.zeros_like(adj_inv), adj_inv)
    adj_inv = SparseTensor.from_dense(adj_inv)
    return adj_inv


def weighted_degree(
    index: Tensor, weights: Optional[Tensor] = None, num_nodes: Optional[int] = None
) -> Tensor:
    r"""Computes the weighted degree of a given one-dimensional index tensor.

    Args:
        index (~torch.Tensor):
            Index tensor.
        weights (~torch.Tensor, optional):
            Edge weights tensor. (default: :obj:`None`)
        num_nodes (int, optional):
            The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`index`. (default: :obj:`None`)
    """
    N = maybe_num_nodes(index, num_nodes)

    if weights is None:
        weights = torch.ones((index.size(0),), device=index.device, dtype=torch.int)
    out = torch.zeros((N,), dtype=weights.dtype, device=weights.device)
    out.scatter_add_(0, index, weights)

    return out


def add_remaining_self_loops(
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    fill_value: float = 1.0,
    num_nodes: Optional[int] = None,
) -> Tuple[Adj, Optional[Tensor]]:
    r"""Adds remaining self loops to the adjacency matrix.

    This method extends the method :obj:`~torch_geometric.utils.add_remaining_self_loops`
    by allowing to pass a :obj:`SparseTensor` as input.

    Args:
        edge_index (~torch.Tensor or SparseTensor): The edge indices.
        edge_weight (~torch.Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): The fill value of the diagonal.
            (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """
    if isinstance(edge_index, SparseTensor):
        if num_nodes is not None and num_nodes != edge_index.size(0):
            edge_index = edge_index.sparse_resize((num_nodes, num_nodes))
        return edge_index.fill_diag(fill_value), None
    from torch_geometric.utils import add_remaining_self_loops as arsl

    return arsl(edge_index, edge_weight, fill_value, num_nodes)
