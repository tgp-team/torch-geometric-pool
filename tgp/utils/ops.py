from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import add_remaining_self_loops as arsl
from torch_geometric.utils import (
    get_laplacian,
)
from torch_geometric.utils import (
    remove_self_loops as rsl,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_scatter import scatter

from tgp import eps
from tgp.imports import HAS_TORCH_SPARSE, is_sparsetensor

# TODO: add docstrings and put these functions in the API documentation


def rank3_trace(x):
    r"""Compute the trace of each matrix in a rank-3 tensor.

    Args:
        x (~torch.Tensor): Input tensor of shape :math:`(B, N, N)`.

    Returns:
        ~torch.Tensor: Vector of shape :math:`(B,)` containing the trace of each
        matrix.
    """
    return torch.einsum("ijj->i", x)


def rank3_diag(x):
    r"""Create a batch of diagonal matrices from a rank-2 tensor.

    Args:
        x (~torch.Tensor): Input tensor of shape :math:`(B, N)`.

    Returns:
        ~torch.Tensor: Batched diagonal matrices of shape :math:`(B, N, N)`.
    """
    return torch.diag_embed(x)


def dense_to_block_diag(adj_pool: Tensor) -> Tuple[Tensor, Tensor]:
    r"""Convert dense pooled adjacencies to a block-diagonal sparse format.

    Args:
        adj_pool (~torch.Tensor): Dense pooled adjacency of shape
            :math:`(B, K, K)` or :math:`(K, K)`.

    Returns:
        tuple:
            - **edge_index** (*~torch.Tensor*): Edge indices of shape
              :math:`(2, E)` for the block-diagonal adjacency.
            - **edge_weight** (*~torch.Tensor*): Edge weights of shape :math:`(E,)`.
    """
    if adj_pool.dim() == 2:
        adj_pool = adj_pool.unsqueeze(0)
    if adj_pool.dim() != 3:
        raise ValueError("adj_pool must have shape [B, K, K] or [K, K].")

    _, num_clusters, _ = adj_pool.size()
    mask = adj_pool.abs() > eps
    if not mask.any():
        edge_index = torch.empty((2, 0), dtype=torch.long, device=adj_pool.device)
        edge_weight = torch.empty((0,), dtype=adj_pool.dtype, device=adj_pool.device)
        return edge_index, edge_weight

    batch_idx, row_idx, col_idx = mask.nonzero(as_tuple=True)
    offset = batch_idx * num_clusters
    edge_index = torch.stack([row_idx + offset, col_idx + offset], dim=0)
    edge_weight = adj_pool[batch_idx, row_idx, col_idx]
    return edge_index, edge_weight


def get_mask_from_dense_s(
    s: Tensor,
    batch: Optional[Tensor] = None,
) -> Tensor:
    r"""Build a dense boolean mask of shape :math:`[B, K]` indicating which
    supernodes have at least one assigned node.

    Use this when returning a dense :class:`~tgp.src.PoolingOutput`
    (e.g. :obj:`sparse_output=False`) so that downstream layers and global
    pooling can ignore padding. Works for both batched and unbatched paths,
    and for both dense and sparse assignment matrices :obj:`s`.

    Args:
        s (~torch.Tensor): Assignment matrix of shape :math:`[N, K]` or :math:`[B, N, K]`.
        batch: Batch vector of shape :math:`[N]` when multiple graphs are
            present (unbatched path with batch). If :obj:`None`, returns
            mask of shape :math:`[1, K]`.

    Returns:
        Boolean tensor of shape :math:`[B, K]` with :math:`B=1` when
        :obj:`batch` is :obj:`None`.
    """
    K = s.size(-1)
    device = s.device

    assert not s.is_sparse, "s must be a dense tensor"
    # Dense S: [N, K] or [B, N, K]
    if s.dim() == 3:
        mask = s.sum(dim=-2) > 0
        return mask
    # 2D [N, K]: single graph (batch None) or multi-graph (batch provided)
    if batch is None:
        mask = (s.sum(dim=-2) > 0).unsqueeze(0)
        return mask
    batch_size = int(batch.max().item()) + 1
    mask = torch.zeros(batch_size, K, dtype=torch.bool, device=device)
    for b in range(batch_size):
        node_mask = batch == b
        if node_mask.any():
            mask[b] = s[node_mask].sum(dim=0) > 0
    return mask


def is_dense_adj(edge_index: Adj) -> bool:
    r"""Return :obj:`True` if :attr:`edge_index` looks like a dense adjacency matrix.

    Accepts a batched dense tensor of shape :math:`(B, N, N)` or a single dense
    adjacency matrix of shape :math:`(N, N)`.
    """
    if not isinstance(edge_index, Tensor) or edge_index.is_sparse:
        return False
    if edge_index.dim() == 3:
        return True
    if edge_index.dim() == 2 and edge_index.size(0) == edge_index.size(1):
        return edge_index.is_floating_point()
    return False


def postprocess_adj_pool_dense(
    adj_pool: Tensor,
    remove_self_loops: bool = False,
    degree_norm: bool = False,
    adj_transpose: bool = False,
    edge_weight_norm: bool = False,
) -> Tensor:
    r"""Postprocess a batched dense pooled adjacency tensor.

    Args:
        adj_pool (~torch.Tensor): Dense pooled adjacency of shape
            :math:`(B, K, K)`.
        remove_self_loops (bool, optional): If :obj:`True`, zeroes the diagonal.
            (default: :obj:`False`)
        degree_norm (bool, optional): If :obj:`True`, applies
            :math:`\mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}` normalization.
            (default: :obj:`False`)
        adj_transpose (bool, optional): If :obj:`True`, treats the output as
            transposed when computing row sums for normalization. (default: :obj:`False`)
        edge_weight_norm (bool, optional): If :obj:`True`, normalizes by the
            maximum absolute edge weight per graph. (default: :obj:`False`)

    Returns:
        ~torch.Tensor: The postprocessed adjacency tensor of shape :math:`(B, K, K)`.
    """
    if remove_self_loops:
        torch.diagonal(adj_pool, dim1=-2, dim2=-1)[:] = 0

    # Apply degree normalization D^{-1/2} A D^{-1/2}
    if degree_norm:
        if adj_transpose:
            # For the transposed output the "row" sum is along axis -2
            d = adj_pool.sum(-2, keepdim=True)
        else:
            # Compute row sums along the last dimension.
            d = adj_pool.sum(-1, keepdim=True)
        d = torch.sqrt(d.clamp(min=eps))
        adj_pool = (adj_pool / d) / d.transpose(-2, -1)

    # Normalize edge weights by maximum absolute value per graph
    if edge_weight_norm:
        batch_size = adj_pool.size(0)
        # Find max absolute value per graph: [batch_size, 1, 1]
        max_per_graph = adj_pool.view(batch_size, -1).abs().max(dim=1, keepdim=True)[0]
        max_per_graph = max_per_graph.unsqueeze(-1)  # [batch_size, 1, 1]

        # Avoid division by zero
        max_per_graph = torch.where(
            max_per_graph == 0, torch.ones_like(max_per_graph), max_per_graph
        )

        adj_pool = adj_pool / max_per_graph

    return adj_pool


def postprocess_adj_pool_sparse(
    edge_index: Tensor,
    edge_weight: Optional[Tensor],
    num_nodes: int,
    remove_self_loops: bool = False,
    degree_norm: bool = False,
    edge_weight_norm: bool = False,
    batch_pooled: Optional[Tensor] = None,
) -> Tuple[Tensor, Optional[Tensor]]:
    r"""Postprocess a sparse pooled adjacency in :obj:`edge_index` format.

    Args:
        edge_index (~torch.Tensor): Edge indices of shape :math:`(2, E)`.
        edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
            (default: :obj:`None`)
        num_nodes (int): Number of pooled nodes.
        remove_self_loops (bool, optional): If :obj:`True`, removes self-loops.
            (default: :obj:`False`)
        degree_norm (bool, optional): If :obj:`True`, applies symmetric degree
            normalization to edge weights. (default: :obj:`False`)
        edge_weight_norm (bool, optional): If :obj:`True`, normalizes edge
            weights by the maximum absolute value per graph. Requires
            :attr:`batch_pooled`. (default: :obj:`False`)
        batch_pooled (~torch.Tensor, optional): Batch vector for pooled nodes of
            shape :math:`(N,)`, used for per-graph normalization.
            (default: :obj:`None`)

    Returns:
        tuple:
            - **edge_index** (*~torch.Tensor*): Filtered edge indices.
            - **edge_weight** (*~torch.Tensor or None*): Filtered edge weights.
    """
    if remove_self_loops:
        edge_index, edge_weight = rsl(edge_index, edge_weight)

    # Filter out edges with tiny weights.
    if edge_weight is not None:
        edge_weight = edge_weight.view(-1)
        if edge_weight.numel() > 0:
            mask = edge_weight.abs() > eps
            if not torch.all(mask):
                edge_index = edge_index[:, mask]
                edge_weight = edge_weight[mask]

    # Apply degree normalization D^{-1/2} A D^{-1/2}
    if degree_norm:
        if edge_weight is None:
            edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

        # Compute degree
        deg = scatter(
            edge_weight,
            edge_index[0],
            dim=0,
            dim_size=num_nodes,
            reduce="sum",
        )
        deg = deg.clamp(min=eps)  # Avoid tiny degrees that explode gradients
        deg_inv_sqrt = deg.pow(-0.5)

        # Apply symmetric normalization to edge weights
        edge_weight = (
            edge_weight * deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
        )

    # Normalize edge weights by maximum absolute value per graph
    if edge_weight_norm and edge_weight is not None:
        # Per-graph normalization using batch_pooled
        edge_batch = batch_pooled[edge_index[0]]

        # Find maximum absolute edge weight per graph
        max_per_graph = scatter(edge_weight.abs(), edge_batch, dim=0, reduce="max")

        # Avoid division by zero
        max_per_graph = torch.where(
            max_per_graph == 0, torch.ones_like(max_per_graph), max_per_graph
        )

        # Normalize edge weights by their respective graph's maximum
        edge_weight = edge_weight / max_per_graph[edge_batch]

    return edge_index, edge_weight


####### EXTERNAL FUNCTIONS - INPUT CAN BE EDGE INDEX, TORCH COO SPARSE TENSOR OR SPARSE TENSOR ###########


def connectivity_to_edge_index(
    edge_index: Adj, edge_weight: Optional[Tensor] = None
) -> Tuple[Tensor, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        if edge_index.is_sparse:
            # Handle torch COO sparse tensor
            # Clone to avoid returning views that share memory with the sparse tensor
            indices = edge_index.indices().clone()
            values = edge_index.values().clone()
            return indices, values
        else:
            # Handle regular tensor [2, E]
            edge_weight = check_and_filter_edge_weights(edge_weight)
            return edge_index, edge_weight
    elif is_sparsetensor(edge_index):
        row, col, edge_weight = edge_index.coo()
        edge_index = torch.stack([row, col], dim=0)
        return edge_index, edge_weight
    else:
        raise NotImplementedError()


def connectivity_to_torch_coo(
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
) -> torch.Tensor:
    # Validate input type first
    if not isinstance(edge_index, Tensor) and not is_sparsetensor(edge_index):
        raise ValueError(
            f"Edge index must be of type Tensor or SparseTensor, got {type(edge_index)}"
        )

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_weight = check_and_filter_edge_weights(edge_weight)

    if isinstance(edge_index, Tensor):
        if edge_index.is_sparse:
            # Already a torch COO sparse tensor
            return edge_index
        else:
            # Handle regular tensor [2, E]
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)
            return torch.sparse_coo_tensor(
                edge_index, edge_weight, (num_nodes, num_nodes)
            ).coalesce()
    elif is_sparsetensor(edge_index):
        row, col, value = edge_index.coo()
        indices = torch.stack([row, col], dim=0)
        if value is None:
            value = torch.ones(row.size(0), device=row.device)
        return torch.sparse_coo_tensor(
            indices, value, (num_nodes, num_nodes)
        ).coalesce()
    else:
        raise ValueError("Edge index must be a Tensor or SparseTensor.")


def connectivity_to_sparsetensor(
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
):
    if not HAS_TORCH_SPARSE:
        raise ImportError(
            "Cannot convert connectivity to sparse tensor: torch_sparse is not installed."
        )
    else:
        from torch_sparse import SparseTensor
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if isinstance(edge_index, SparseTensor):
        return edge_index
    elif isinstance(edge_index, Tensor):
        if edge_index.is_sparse:
            # Handle torch COO sparse tensor
            sparse_tensor = edge_index
            edge_index = sparse_tensor.indices().clone()
            edge_weight = sparse_tensor.values().clone()

        edge_weight = check_and_filter_edge_weights(edge_weight)
        adj = SparseTensor.from_edge_index(
            edge_index, edge_weight, (num_nodes, num_nodes)
        )
        return adj
    else:
        raise NotImplementedError()


########### INTERNAL FUNCTIONS - ONLY INPUT ARE TENSORS (EDGE INDEX OR TORCH COO SPARSE TENSOR) ###########


def pseudo_inverse(edge_index: Tensor) -> Tuple[Adj, Optional[Tensor]]:
    if isinstance(edge_index, Tensor):
        to_torch_coo = False
        if edge_index.is_sparse:  # Sparse pooling with torch COO
            to_torch_coo = True
            edge_index = edge_index.to_dense()
        # Convert to float for pinv computation
        adj_inv = torch.linalg.pinv(edge_index.float())
        if to_torch_coo:
            adj_inv = torch.where(
                torch.abs(adj_inv) < 1e-5, torch.zeros_like(adj_inv), adj_inv
            )
            adj_inv = adj_inv.to_sparse_coo()
        return adj_inv
    else:
        raise NotImplementedError()


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
    by allowing to pass a :obj:`SparseTensor` or torch COO sparse tensor as input.

    Args:
        edge_index (~torch.Tensor or SparseTensor): The edge indices.
        edge_weight (~torch.Tensor, optional): One-dimensional edge weights.
            (default: :obj:`None`)
        fill_value (float, optional): The fill value of the diagonal.
            (default: :obj:`1.`)
        num_nodes (int, optional): The number of nodes, *i.e.*
            :obj:`max_val + 1` of :attr:`edge_index`. (default: :obj:`None`)
    """
    if is_sparsetensor(edge_index):
        if num_nodes is not None and num_nodes != edge_index.size(0):
            edge_index = edge_index.sparse_resize((num_nodes, num_nodes))
        return edge_index.fill_diag(fill_value), None

    if isinstance(edge_index, Tensor) and edge_index.is_sparse:
        # Handle torch sparse COO adjacency matrices.
        indices = edge_index.indices()
        values = edge_index.values()
        num_nodes = maybe_num_nodes(indices, num_nodes)
        loop_index, loop_weight = arsl(indices, values, fill_value, num_nodes)
        # Rebuild a sparse COO tensor to return the same input type.
        adj = torch.sparse_coo_tensor(
            loop_index,
            loop_weight,
            size=(num_nodes, num_nodes),
            device=edge_index.device,
        ).coalesce()
        return adj, None

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
    r"""Compute the :math:`\delta`-GCN propagation matrix for heterophilic message passing.

    Constructs the :math:`\delta`-GCN propagation matrix from `MaxCutPool: differentiable
    feature-aware Maxcut for pooling in graph neural networks` (Abate & Bianchi, ICLR 2025).

    The propagation matrix is computed as: :math:`\mathbf{P} = \mathbf{I} - \delta \cdot \mathbf{L}_{sym}`
    where :math:`\mathbf{L}_{sym}` is the symmetric normalized Laplacian.

    As described in the paper, when :math:`\delta > 1`, this operator favors the realization
    of non-smooth (high-frequency) signals on the graph, making it particularly suitable
    for heterophilic graphs and MaxCut optimization where adjacent nodes should have
    different values.

    Args:
        edge_index (~torch.Tensor or SparseTensor): Graph connectivity in COO format of shape :math:`(2, E)`,
            as torch COO sparse tensor, or as SparseTensor.
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
              or None if input was SparseTensor or torch COO sparse tensor.
    """
    # Remember the input type to return the same format
    input_is_sparsetensor = is_sparsetensor(edge_index)
    input_is_torch_coo = isinstance(edge_index, Tensor) and edge_index.is_sparse

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
    diag_indices = torch.arange(num_nodes, device=edge_index_tensor.device)
    eye_index = torch.stack([diag_indices, diag_indices], dim=0)
    eye_weight = torch.ones(
        num_nodes, device=edge_index_tensor.device, dtype=edge_weight_scaled.dtype
    )

    # Combine to form Delta-GCN propagation matrix: P = I - delta * L_sym
    # Concatenate indices and values from Laplacian and identity
    combined_indices = torch.cat([edge_index_laplacian, eye_index], dim=1)
    combined_values = torch.cat([edge_weight_scaled, eye_weight], dim=0)

    # Create torch sparse COO tensor and coalesce to sum overlapping edges (diagonal elements)
    propagation_matrix = torch.sparse_coo_tensor(
        combined_indices,
        combined_values,
        size=(num_nodes, num_nodes),
        device=edge_index_tensor.device,
    ).coalesce()

    # Return in the same format as input
    if input_is_sparsetensor:
        # Convert torch COO to SparseTensor
        propagation_matrix_spt = connectivity_to_sparsetensor(
            propagation_matrix, None, num_nodes
        )
        return propagation_matrix_spt, None
    elif input_is_torch_coo:
        return propagation_matrix, None
    else:
        # Convert back to edge_index, edge_weight format
        edge_index_out, edge_weight_out = connectivity_to_edge_index(
            propagation_matrix, None
        )
        return edge_index_out, edge_weight_out


def create_one_hot_tensor(num_nodes, kept_node_tensor, device, dtype=None):
    r"""Create a one-hot assignment matrix for kept nodes.

    Args:
        num_nodes (int): Total number of nodes :math:`N`.
        kept_node_tensor (~torch.Tensor): Indices of kept nodes of shape :math:`(K,)`.
        device (~torch.device): Device to create the tensor on.
        dtype (~torch.dtype, optional): Desired dtype. (default: :obj:`torch.float32`)

    Returns:
        ~torch.Tensor: One-hot matrix of shape :math:`(N, K + 1)`, where column
        :math:`0` denotes "unassigned" and columns :math:`1..K` correspond to kept nodes.
    """
    # Ensure kept_node_tensor is at least 1D to avoid issues with len()
    if kept_node_tensor.dim() == 0:
        kept_node_tensor = kept_node_tensor.unsqueeze(0)

    num_kept = kept_node_tensor.size(0)
    if dtype is None:
        dtype = torch.float32
    tensor = torch.zeros(num_nodes, num_kept + 1, device=device, dtype=dtype)
    tensor[kept_node_tensor, 1:] = torch.eye(num_kept, device=device, dtype=dtype)
    return tensor


def get_random_map_mask(kept_nodes, mask, batch=None):
    r"""Randomly assign remaining unassigned nodes to kept nodes.

    Args:
        kept_nodes (~torch.Tensor): Indices of kept nodes (supernodes).
        mask (~torch.Tensor): Boolean mask of already assigned nodes of shape
            :math:`(N,)`.
        batch (~torch.Tensor, optional): Batch vector of shape :math:`(N,)`. If
            provided, nodes are assigned only to kept nodes within the same
            graph. This assumes kept nodes are grouped by batch (e.g., sorted by
            node index) and each graph has at least one kept node.
            (default: :obj:`None`)

    Returns:
        ~torch.Tensor: Mapping tensor of shape :math:`(2, N_u)` where the first
        row contains unassigned node indices and the second row contains the
        chosen kept node indices.
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


def propagate_assignments_sparse(
    assignments, edge_index, kept_node_tensor, mask, num_clusters
):
    r"""Propagate assignments through edges using sparse operations.

    This function avoids allocating a dense [num_nodes, num_clusters] tensor and
    works directly on the COO edge list. For each unassigned destination node,
    it counts how many incoming edges come from each assigned cluster, then picks
    the cluster with the highest count (ties are resolved by the smallest cluster
    index). Propagation follows the direction of `edge_index`; for undirected
    behavior, ensure the edge list is symmetric.

    Complexity: O(E log E) time due to sorting/unique and O(E) memory.

    Args:
        assignments (~torch.Tensor): Assignment vector of shape :math:`(N,)`,
            with :obj:`0` for unassigned nodes and :obj:`1..K` for cluster indices.
        edge_index (~torch.Tensor): Edge indices of shape :math:`(2, E)`.
        kept_node_tensor (~torch.Tensor): Indices of kept nodes (supernodes).
        mask (~torch.Tensor): Boolean mask of assigned nodes of shape :math:`(N,)`.
        num_clusters (int): Number of clusters :math:`K`.

    Returns:
        tuple:
            - **assignments** (*~torch.Tensor*): Updated assignment vector.
            - **mapping** (*~torch.Tensor*): Mapping tensor of shape :math:`(2, N_a)`
              from newly assigned node indices to kept node indices.
            - **mask** (*~torch.Tensor*): Updated boolean mask.
    """
    src, dst = edge_index[0], edge_index[1]

    src_assignments = assignments[src]

    valid_edges = (src_assignments > 0) & (~mask[dst])

    if valid_edges.sum() == 0:
        return (
            assignments,
            torch.empty((2, 0), device=assignments.device, dtype=torch.long),
            mask,
        )

    valid_dst = dst[valid_edges]
    valid_src_assignments = src_assignments[valid_edges]

    combined = valid_dst * (num_clusters + 1) + valid_src_assignments
    unique_combined, counts = torch.unique(combined, return_counts=True)

    unique_dst_per_pair = unique_combined // (num_clusters + 1)
    unique_assignment_per_pair = unique_combined % (num_clusters + 1)

    max_count = counts.max().item() + 1
    sort_key = (
        unique_dst_per_pair * (max_count * (num_clusters + 1))
        - counts * (num_clusters + 1)
        + unique_assignment_per_pair
    )

    sorted_indices = torch.argsort(sort_key)
    sorted_dst = unique_dst_per_pair[sorted_indices]
    sorted_assignments = unique_assignment_per_pair[sorted_indices]

    dst_changes = torch.cat(
        [
            torch.tensor([True], device=sorted_dst.device),
            sorted_dst[1:] != sorted_dst[:-1],
        ]
    )

    best_dst = sorted_dst[dst_changes]
    best_assignments = sorted_assignments[dst_changes]

    valid_mask = best_assignments > 0
    if valid_mask.sum() == 0:
        return (
            assignments,
            torch.empty((2, 0), device=assignments.device, dtype=torch.long),
            mask,
        )

    newly_assigned_dst = best_dst[valid_mask]
    newly_assigned_clusters = best_assignments[valid_mask]

    supernode_indices = kept_node_tensor[newly_assigned_clusters - 1]

    assignments = assignments.clone()
    assignments[newly_assigned_dst] = newly_assigned_clusters
    mask = mask.clone()
    mask[newly_assigned_dst] = True

    mappa = torch.stack([newly_assigned_dst, supernode_indices])

    return assignments, mappa, mask


def get_assignments(
    kept_node_indices, edge_index=None, max_iter=5, batch=None, num_nodes=None
):
    r"""Assigns all nodes in a graph to the closest kept nodes (supernodes) using
    a hierarchical assignment strategy with message passing (torch COO version).

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

        # Get edge_index in [2, E] format
        if isinstance(edge_index, Tensor) and edge_index.is_sparse:
            edge_index_2d = edge_index.indices()
        else:
            edge_index_2d = edge_index

        # Initialize assignment vector: 0 = unassigned, 1-K = cluster index
        num_clusters = kept_node_tensor.size(0)
        assignments = torch.zeros(num_nodes, device=device, dtype=torch.long)
        assignments[kept_node_tensor] = torch.arange(1, num_clusters + 1, device=device)

        # Iterative assignment through message passing (fully sparse)
        for _ in range(max_iter):
            if mask.all():  # All nodes assigned
                break
            assignments, _map, mask = propagate_assignments_sparse(
                assignments, edge_index_2d, kept_node_tensor, mask, num_clusters
            )
            if _map.size(1) > 0:
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
