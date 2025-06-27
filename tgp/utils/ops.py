from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import (
    get_laplacian,
    remove_isolated_nodes,
)
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
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    delta: float = 2.0,
    num_nodes: Optional[int] = None,
) -> Tuple[Adj, Optional[Tensor]]:
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
        edge_index (~torch.Tensor or SparseTensor): Graph connectivity in COO format of shape :math:`(2, E)`
            or as SparseTensor.
        edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
            (default: :obj:`None`)
        delta (float, optional): Delta parameter for heterophilic message passing. When
            :math:`\delta > 1`, promotes high-frequency (non-smooth) signals. (default: :obj:`2.0`)
        num_nodes (int, optional): Number of nodes. If :obj:`None`, inferred from
            :obj:`edge_index`. (default: :obj:`None`)

    Returns:
        tuple:
            - **edge_index** (*Tensor or SparseTensor*): Updated connectivity in the same format as input.
            - **edge_weight** (*Tensor or None*): Updated edge weights of shape :math:`(E',)` if input was Tensor,
              or None if input was SparseTensor.
    """
    # Remember the input type to return the same format
    input_is_sparse = isinstance(edge_index, SparseTensor)

    # Convert input to edge_index, edge_weight format for processing
    edge_index_tensor, edge_weight_tensor = connectivity_to_edge_index(
        edge_index, edge_weight
    )

    num_nodes = maybe_num_nodes(edge_index_tensor, num_nodes)

    # Get symmetric normalized Laplacian: L_sym = D^(-1/2) (D - A) D^(-1/2)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index_tensor, edge_weight_tensor, normalization="sym", num_nodes=num_nodes
    )

    # Scale by delta and negate: -delta * L_sym
    edge_weight_scaled = -delta * edge_weight_laplacian

    # Create identity matrix: I
    eye_index, eye_weight = torch_sparse_eye(
        num_nodes, device=edge_index_tensor.device, dtype=edge_weight_scaled.dtype
    )

    # Combine to form Delta-GCN propagation matrix: P = I - delta * L_sym
    propagation_matrix = SparseTensor(
        row=torch.cat([edge_index_laplacian[0], eye_index[0]]),
        col=torch.cat([edge_index_laplacian[1], eye_index[1]]),
        value=torch.cat([edge_weight_scaled, eye_weight]),
        sparse_sizes=(num_nodes, num_nodes),
    ).coalesce("sum")  # Sum weights for overlapping edges (diagonal elements)

    # Return in the same format as input
    if input_is_sparse:
        return propagation_matrix, None
    else:
        # Convert back to COO format
        row, col, edge_weight_out = propagation_matrix.coo()
        edge_index_out = torch.stack([row, col], dim=0)
        return edge_index_out, edge_weight_out


def reset_node_numbers(edge_index, edge_attr=None):
    """Reset node indices after removing isolated nodes.

    Args:
        edge_index (Tensor): Graph connectivity in COO format
        edge_attr (Tensor, optional): Edge attributes. Defaults to None.

    Returns:
        tuple:
            - Tensor: Updated edge indices
            - Tensor: Updated edge attributes
    """
    edge_index, edge_attr, _ = remove_isolated_nodes(edge_index, edge_attr=edge_attr)
    return edge_index, edge_attr


def create_one_hot_tensor(num_nodes, kept_node_tensor, device):
    """Create one-hot encoding tensor for node assignments.

    Args:
        num_nodes (int): Total number of nodes
        kept_node_tensor (Tensor): Indices of nodes to keep
        device (torch.device): Device to create tensor on

    Returns:
        Tensor: One-hot encoding matrix [num_nodes, num_kept_nodes + 1]
    """
    # Ensure kept_node_tensor is at least 1D to avoid issues with len()
    if kept_node_tensor.dim() == 0:
        kept_node_tensor = kept_node_tensor.unsqueeze(0)

    num_kept = kept_node_tensor.size(0)
    tensor = torch.zeros(num_nodes, num_kept + 1, device=device)
    tensor[kept_node_tensor, 1:] = torch.eye(num_kept, device=device)
    return tensor


def get_sparse_map_mask(x, edge_index, kept_node_tensor, mask):
    """Compute sparse assignment mapping using message passing.

    Args:
        x (Tensor): Node features/assignments
        edge_index (Tensor): Graph connectivity
        kept_node_tensor (Tensor): Indices of kept nodes
        mask (Tensor): Boolean mask of already assigned nodes

    Returns:
        tuple:
            - Tensor: Propagated features
            - Tensor: Node assignment mapping
            - Tensor: Updated assignment mask
    """
    sparse_ei = SparseTensor.from_edge_index(
        edge_index, sparse_sizes=(x.size(0), x.size(0))
    )
    y = sparse_ei.matmul(x)  # propagation step
    first_internal_mask = torch.logical_not(
        mask
    )  # get the mask of the nodes that have not been assigned yet
    am = torch.argmax(y, dim=1)  # get the visited nodes
    nonzero = torch.nonzero(am, as_tuple=True)[
        0
    ]  # get the visited nodes that are not zero (since the zero-th node is a fake node)
    second_internal_mask = torch.zeros_like(
        first_internal_mask, dtype=torch.bool
    )  # initialize the second mask
    second_internal_mask[nonzero] = True  # set the mask to True for the visited nodes
    final_mask = torch.logical_and(
        first_internal_mask, second_internal_mask
    )  # newly visited nodes that have not been assigned yet
    indices = torch.arange(x.size(0), device=x.device)  # inizialize the indices
    out = kept_node_tensor[
        am - 1
    ]  # get the supernode indices of the visited nodes (am-1 because the zero-th node is a fake node)

    indices = indices[
        final_mask
    ]  # get the indices of the newly visited nodes that have not been assigned yet

    mappa = torch.stack([indices, out[indices]])  # create the map
    mask[indices] = (
        True  # set the mask to True for the newly visited nodes that have not been assigned yet
    )

    return y, mappa, mask


def get_random_map_mask(kept_nodes, mask, batch=None):
    """Randomly assign remaining unassigned nodes.

    Args:
        kept_nodes (Tensor): Indices of kept nodes
        mask (Tensor): Boolean mask of already assigned nodes
        batch (Tensor, optional): Batch assignments. Defaults to None.

    Returns:
        Tensor: Random assignment mapping for unassigned nodes
    """
    neg_mask = torch.logical_not(mask)
    zero = torch.arange(mask.size(0), device=kept_nodes.device)[neg_mask]
    one = torch.randint(
        0, kept_nodes.size(0), (zero.size(0),), device=kept_nodes.device
    )

    if batch is not None:
        s_batch = batch[kept_nodes]
        s_counts = torch.bincount(s_batch)

        cumsum = torch.zeros(s_counts.size(0), device=batch.device).to(torch.long)
        cumsum[1:] = s_counts.cumsum(dim=0)[:-1]

        count_tensor = s_counts[batch].to(torch.long)
        sum_tensor = cumsum[batch].to(torch.long)

        count_tensor = count_tensor[neg_mask]
        sum_tensor = sum_tensor[neg_mask]

        one = one % count_tensor + sum_tensor

        one = kept_nodes[one]

    mappa = torch.stack([zero, one])
    return mappa


def get_assignments(
    kept_node_indices, edge_index=None, max_iter=5, batch=None, num_nodes=None
):
    r"""Assigns all nodes in a graph to the closest kept nodes (supernodes) using
    a hierarchical assignment strategy with message passing.

    This function implements a graph-aware node assignment algorithm that combines
    iterative message passing with random assignment fallback. It's designed to
    create cluster assignments where each node in the graph is mapped to one of
    the provided kept nodes (supernodes).

    **Algorithm Overview:**

    1. **Initial Assignment**: All kept nodes are assigned to themselves.
    2. **Iterative Propagation**: For `max_iter` iterations, unassigned nodes
       are assigned to supernodes by finding the closest supernode through
       graph message passing.
    3. **Random Fallback**: Any remaining unassigned nodes are randomly assigned
       to supernodes (respecting batch boundaries if provided).

    This approach ensures that all nodes receive assignments while prioritizing
    graph connectivity and topology-aware clustering.

    Args:
        kept_node_indices (~torch.Tensor or list): Indices of nodes to keep as supernodes.
            These nodes will serve as cluster centers. Can be a tensor or list of integers.
        edge_index (~torch.Tensor, optional): Graph connectivity in COO format of shape
            :math:`(2, E)` where :math:`E` is the number of edges. Required when
            :obj:`max_iter > 0` for graph-aware assignment. (default: :obj:`None`)
        max_iter (int, optional): Maximum number of message passing iterations.
            If :obj:`0`, uses only random assignment. Higher values allow more distant
            nodes to be assigned through graph connectivity. (default: :obj:`5`)
        batch (~torch.Tensor, optional): Batch assignment vector of shape :math:`(N,)`
            indicating which graph each node belongs to. When provided, ensures nodes
            are only assigned to supernodes within the same graph. (default: :obj:`None`)
        num_nodes (int, optional): Total number of nodes in the graph(s). If :obj:`None`,
            inferred from :obj:`edge_index` or :obj:`batch`. Must be provided if both
            :obj:`edge_index` and :obj:`batch` are :obj:`None`. (default: :obj:`None`)

    Returns:
        ~torch.Tensor: Assignment mapping tensor of shape :math:`(2, N)` where the first
        row contains the original node indices :math:`[0, 1, ..., N-1]` and the second
        row contains the corresponding cluster (supernode) indices. The cluster indices
        are renumbered to be consecutive starting from :math:`0`.

    Raises:
        ValueError: If :obj:`num_nodes`, :obj:`batch`, and :obj:`edge_index` are all
            :obj:`None` (cannot determine graph size).
        ValueError: If :obj:`max_iter > 0` but :obj:`edge_index` is :obj:`None`
            (cannot perform graph-aware assignment).

    Example:
        >>> # Basic usage with graph connectivity
        >>> kept_nodes = torch.tensor([0, 3])  # Keep nodes 0 and 3 as supernodes
        >>> edge_index = torch.tensor([[0, 1, 2, 3], [1, 2, 3, 0]])  # Cycle graph
        >>> assignments = get_assignments(kept_nodes, edge_index, max_iter=3)
        >>> print(assignments)
        tensor([[0, 1, 2, 3],
                [0, 0, 1, 1]])  # Nodes 0,1 -> supernode 0; nodes 2,3 -> supernode 1

        >>> # Random assignment only (no graph connectivity)
        >>> assignments = get_assignments(kept_nodes, max_iter=0, num_nodes=4)
        >>> print(assignments.shape)
        torch.Size([2, 4])  # All 4 nodes randomly assigned to 2 supernodes
    """
    if isinstance(kept_node_indices, torch.Tensor):
        kept_node_tensor = torch.squeeze(kept_node_indices).to(torch.long)
    else:
        kept_node_tensor = torch.tensor(kept_node_indices, dtype=torch.long)

    # Determine number of nodes and device
    if num_nodes is None:
        if batch is not None:
            num_nodes = batch.size(0)
        elif edge_index is not None:
            num_nodes = edge_index.max().item() + 1
        else:
            raise ValueError(
                "Either num_nodes, batch, or edge_index must be provided to determine the number of nodes"
            )

    # Determine device
    if edge_index is not None:
        device = edge_index.device
    elif batch is not None:
        device = batch.device
    else:
        device = kept_node_tensor.device

    maps_list = []

    # Initialize mask for assigned nodes
    mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    mask[kept_node_indices] = True

    # Create initial mapping for kept nodes
    _map = torch.stack([kept_node_tensor, kept_node_tensor])
    maps_list.append(_map)

    # Only perform iterative assignment if max_iter > 0 and edge_index is provided
    if max_iter > 0:
        if edge_index is None:
            raise ValueError("edge_index must be provided when max_iter > 0")

        # Clone edge_index to avoid modifying the original
        edge_index = edge_index.clone()

        # Initialize one-hot tensor for propagation
        x = create_one_hot_tensor(num_nodes, kept_node_tensor, device)

        # Iterative assignment through message passing
        for _ in range(max_iter):
            if mask.all():  # All nodes assigned
                break
            x, _map, mask = get_sparse_map_mask(x, edge_index, kept_node_tensor, mask)
            maps_list.append(_map)

    # Randomly assign any remaining unassigned nodes
    if not mask.all():
        _map = get_random_map_mask(kept_node_tensor, mask, batch)
        maps_list.append(_map)

    # Combine all mappings and sort by node index
    assignments = torch.cat(maps_list, dim=1)
    assignments = assignments[:, assignments[0].argsort()]

    # Renumber target indices to be consecutive
    _, unique_one = torch.unique(assignments[1], return_inverse=True)
    assignments[1] = unique_one

    return assignments
