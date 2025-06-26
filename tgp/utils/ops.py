from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, coalesce, remove_self_loops
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
    edge_index_tensor, edge_weight_tensor = connectivity_to_edge_index(edge_index, edge_weight)
    
    num_nodes = maybe_num_nodes(edge_index_tensor, num_nodes)

    # Get symmetric normalized Laplacian: L_sym = D^(-1/2) (D - A) D^(-1/2)
    edge_index_laplacian, edge_weight_laplacian = get_laplacian(
        edge_index_tensor, edge_weight_tensor, normalization='sym', num_nodes=num_nodes
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
        sparse_sizes=(num_nodes, num_nodes)
    ).coalesce("sum")  # Sum weights for overlapping edges (diagonal elements)
    
    # Return in the same format as input
    if input_is_sparse:
        return propagation_matrix, None
    else:
        # Convert back to COO format
        row, col, edge_weight_out = propagation_matrix.coo()
        edge_index_out = torch.stack([row, col], dim=0)
        return edge_index_out, edge_weight_out


def create_one_hot_assignment(num_nodes: int, kept_nodes: Tensor, device: torch.device) -> Tensor:
    """Create one-hot encoding tensor for node assignments.
    
    Args:
        num_nodes: Total number of nodes
        kept_nodes: Indices of nodes to keep as supernodes
        device: Device to create tensor on
        
    Returns:
        One-hot encoding matrix [num_nodes, num_kept_nodes + 1]
        Column 0 is for unassigned nodes, columns 1+ for each supernode
    """
    tensor = torch.zeros(num_nodes, len(kept_nodes) + 1, device=device)
    tensor[kept_nodes, 1:] = torch.eye(len(kept_nodes), device=device)
    return tensor


def propagate_assignments(x: Tensor, edge_index: Tensor, kept_nodes: Tensor, mask: Tensor) -> Tuple[Tensor, Tensor, Tensor]:
    """Propagate supernode assignments through message passing.
    
    Args:
        x: Current assignment matrix [num_nodes, num_supernodes + 1]
        edge_index: Graph connectivity in COO format
        kept_nodes: Indices of supernodes
        mask: Boolean mask of already assigned nodes
        
    Returns:
        Tuple of (propagated_assignments, new_mappings, updated_mask)
    """
    # Create sparse adjacency matrix for propagation
    sparse_adj = SparseTensor.from_edge_index(edge_index, sparse_sizes=(x.size(0), x.size(0)))
    
    # Propagate assignments using sparse matrix multiplication
    y = sparse_adj @ x  # Use @ operator instead of .matmul()
    
    # Find nodes that haven't been assigned yet
    first_internal_mask = torch.logical_not(mask)  # get the mask of the nodes that have not been assigned yet
    
    # Find which supernodes they should be assigned to
    am = torch.argmax(y, dim=1)  # get the visited nodes
    
    # Get nodes that actually received non-zero assignments (nonzero because 0-th column is for unassigned)
    nonzero = torch.nonzero(am, as_tuple=True)[0]  # get the visited nodes that are not zero (since the zero-th node is a fake node)
    second_internal_mask = torch.zeros_like(first_internal_mask, dtype=torch.bool)  # initialize the second mask
    second_internal_mask[nonzero] = True  # set the mask to True for the visited nodes
    
    # Combine masks: unassigned AND received assignment
    final_mask = torch.logical_and(first_internal_mask, second_internal_mask)  # newly visited nodes that have not been assigned yet
    
    # Get indices of newly assigned nodes
    indices = torch.arange(x.size(0), device=x.device)  # initialize the indices
    out = kept_nodes[am - 1]  # get the supernode indices of the visited nodes (am-1 because the zero-th node is a fake node)
    
    indices = indices[final_mask]  # get the indices of the newly visited nodes that have not been assigned yet
    
    if len(indices) > 0:
        # Create mapping [original_node, supernode]
        new_mappings = torch.stack([indices, out[indices]])  # create the map
        
        # Update mask
        mask[indices] = True  # set the mask to True for the newly visited nodes that have not been assigned yet
    else:
        # No new assignments
        new_mappings = torch.empty((2, 0), dtype=torch.long, device=x.device)
    
    return y, new_mappings, mask


def assign_remaining_randomly(kept_nodes: Tensor, mask: Tensor, batch: Optional[Tensor] = None) -> Tensor:
    """Randomly assign remaining unassigned nodes to supernodes.
    
    Args:
        kept_nodes: Indices of supernodes
        mask: Boolean mask of already assigned nodes
        batch: Batch assignments for each node (optional)
        
    Returns:
        Random assignment mapping [original_nodes, supernodes]
    """
    unassigned_mask = torch.logical_not(mask)
    unassigned_indices = torch.nonzero(unassigned_mask, as_tuple=True)[0]
    
    if len(unassigned_indices) == 0:
        return torch.empty((2, 0), dtype=torch.long, device=kept_nodes.device)
    
    # Random assignment to supernodes
    random_assignments = torch.randint(0, len(kept_nodes), (len(unassigned_indices),), device=kept_nodes.device)
    
    if batch is not None:
        # Ensure assignments respect batch boundaries
        supernode_batch = batch[kept_nodes]
        supernode_counts = torch.bincount(supernode_batch)
        
        # Calculate cumulative offsets for each batch
        cumsum = torch.zeros(supernode_counts.size(0), device=batch.device, dtype=torch.long)
        cumsum[1:] = supernode_counts.cumsum(dim=0)[:-1]
        
        # Get batch info for unassigned nodes
        unassigned_batch = batch[unassigned_indices]
        batch_counts = supernode_counts[unassigned_batch]
        batch_offsets = cumsum[unassigned_batch]
        
        # Adjust random assignments to respect batch boundaries
        random_assignments = random_assignments % batch_counts + batch_offsets
        random_assignments = kept_nodes[random_assignments]
    else:
        random_assignments = kept_nodes[random_assignments]
    
    return torch.stack([unassigned_indices, random_assignments])


def create_assignment_matrix(mappings_list: list) -> Tensor:
    """Create final assignment matrix from hierarchical mappings.
    
    Args:
        mappings_list: List of assignment mappings at each level
        
    Returns:
        Final assignment matrix [original_nodes, supernodes]
    """
    if len(mappings_list) == 0:
        return torch.empty((2, 0), dtype=torch.long)
    
    # Concatenate all mappings
    all_mappings = torch.cat(mappings_list, dim=1)
    
    # Sort by original node indices
    sorted_indices = all_mappings[0].argsort()
    all_mappings = all_mappings[:, sorted_indices]
    
    # Renumber supernodes to be consecutive
    _, unique_supernodes = torch.unique(all_mappings[1], return_inverse=True)
    all_mappings[1] = unique_supernodes
    
    return all_mappings


def generate_maxcut_assignment_matrix(
    edge_index: Tensor,
    kept_nodes: Tensor,
    max_iter: int = 5,
    batch: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> Tensor:
    """Generate assignment matrix for MaxCut pooling.
    
    Creates a hierarchical assignment matrix that maps all nodes to selected supernodes
    using iterative message passing, following the original MaxCutPool approach.
    
    Args:
        edge_index: Graph connectivity in COO format
        kept_nodes: Indices of nodes selected as supernodes
        max_iter: Maximum iterations for propagation
        batch: Batch assignments for each node (optional)
        num_nodes: Total number of nodes (inferred if None)
        
    Returns:
        Assignment matrix [original_nodes, supernodes] where:
        - Row 0: original node indices
        - Row 1: corresponding supernode indices
    """
    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
    
    device = edge_index.device
    
    # Initialize one-hot assignment matrix
    x = create_one_hot_assignment(num_nodes, kept_nodes, device)
    
    # Track which nodes have been assigned
    mask = torch.zeros(num_nodes, device=device, dtype=torch.bool)
    mask[kept_nodes] = True  # Supernodes are pre-assigned to themselves
    
    # List to store mappings at each iteration
    mappings_list = []
    
    # Initial mapping: supernodes to themselves
    initial_mapping = torch.stack([kept_nodes, kept_nodes])
    mappings_list.append(initial_mapping)
    
    # Iterative propagation
    for _ in range(max_iter):
        if mask.all():  # All nodes assigned
            break
            
        x, new_mapping, mask = propagate_assignments(x, edge_index, kept_nodes, mask)
        
        if new_mapping.size(1) > 0:
            mappings_list.append(new_mapping)
    
    # Assign any remaining unassigned nodes randomly
    if not mask.all():
        random_mapping = assign_remaining_randomly(kept_nodes, mask, batch)
        if random_mapping.size(1) > 0:
            mappings_list.append(random_mapping)
    
    # Create final assignment matrix
    assignment_matrix = create_assignment_matrix(mappings_list)
    
    return assignment_matrix
