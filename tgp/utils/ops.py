from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import eye as torch_sparse_eye

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


def check_and_filter_edge_weights(edge_weight: Tensor) -> Optional[Tensor]:
    r"""Check and filter edge weights to ensure they are in the correct shape
     :math:`[E]` or :math:`[E, 1]`.

    Args:
        edge_weight (Tensor): The edge weights tensor.
    """
    if edge_weight is not None:
        if edge_weight.ndim > 1:
            if edge_weight.ndim == 2 and edge_weight.size(-1) == 1:
                edge_weight = edge_weight.flatten()
            else:
                raise RuntimeError(
                    f"Edge weights must be of shape [E] or [E, 1], but got {edge_weight.shape}."
                )
    return edge_weight


def delta_gcn_matrix(
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    delta: float = 2.0,
    num_nodes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    r"""Compute the Delta-GCN propagation matrix for heterophilic message passing.
    
    Constructs the Delta-GCN propagation matrix from `MaxCutPool: differentiable 
    feature-aware Maxcut for pooling in graph neural networks` (Abate & Bianchi, ICLR 2025).
    
    The propagation matrix is computed as: :math:`\mathbf{P} = \mathbf{I} - \delta \cdot \mathbf{L}_{sym}`
    where :math:`\mathbf{L}_{sym}` is the symmetric normalized Laplacian.
    
    As described in the paper, when :math:`\delta > 1`, this operator favors the realization
    of non-smooth (high-frequency) signals on the graph, making it particularly suitable
    for heterophilic graphs and MaxCut optimization where adjacent nodes should have
    different values.
    
    Args:
        edge_index (~torch.Tensor): Graph connectivity in COO format of shape :math:`(2, E)`.
        edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
            (default: :obj:`None`)
        delta (float, optional): Delta parameter for heterophilic message passing. When 
            :math:`\delta > 1`, promotes high-frequency (non-smooth) signals. (default: :obj:`2.0`)
        num_nodes (int, optional): Number of nodes. If :obj:`None`, inferred from
            :obj:`edge_index`. (default: :obj:`None`)
        
    Returns:
        tuple:
            - **edge_index** (*Tensor*): Updated edge indices of shape :math:`(2, E')`.
            - **edge_weight** (*Tensor*): Updated edge weights of shape :math:`(E',)`.
    """
    num_nodes = maybe_num_nodes(edge_index, num_nodes)

    # Get symmetric normalized Laplacian: L_sym = D^(-1/2) (D - A) D^(-1/2)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index, edge_weight, normalization='sym', num_nodes=num_nodes
    )
    
    # Scale by delta and negate: -delta * L_sym
    edge_weight_scaled = -delta * edge_weight_laplacian
    
    # Create identity matrix: I
    eye_index, eye_weight = torch_sparse_eye(
        num_nodes, device=edge_index.device, dtype=edge_weight_scaled.dtype
    )
    
    # Combine to form Delta-GCN propagation matrix: P = I - delta * L_sym
    propagation_matrix = SparseTensor(
        row=torch.cat([edge_index_laplacian[0], eye_index[0]]),
        col=torch.cat([edge_index_laplacian[1], eye_index[1]]),
        value=torch.cat([edge_weight_scaled, eye_weight]),
        sparse_sizes=(num_nodes, num_nodes)
    ).coalesce("sum")  # Sum weights for overlapping edges (diagonal elements)
    
    # Convert back to COO format
    row, col, edge_weight_out = propagation_matrix.coo()
    edge_index_out = torch.stack([row, col], dim=0)

    return edge_index_out, edge_weight_out
