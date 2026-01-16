import warnings
from typing import Optional, Tuple, Union

import torch
from torch import Tensor
from torch_geometric.utils import (
    cumsum,
    degree,
    index_sort,
    to_undirected,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes

_MIN_PROB_EDGES = 0.5  # 50%
_MAX_PROB_EDGES = 2 / 3  # 66%


def negative_edge_sampling(
    edge_index: Tensor,
    num_nodes: Optional[Union[int, Tuple[int, int]]] = None,
    num_neg_samples: Optional[int] = None,
    method: str = "auto",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of a graph given by :attr:`edge_index`.

    Args:
        edge_index (LongTensor): The edge indices.
        num_nodes (int or Tuple[int, int], optional): The number of nodes,
            *i.e.* :obj:`max_val + 1` of :attr:`edge_index`.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph with shape :obj:`(num_src_nodes, num_dst_nodes)`.
            (default: :obj:`None`)
        num_neg_samples (int, optional): The (approximate) number of negative
            samples to return.
            If set to :obj:`None`, will try to return a negative edge for every
            positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"`, :obj:`"dense"`, or :obj:`"auto"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, but it could
            retrieve a different number of negative samples
            :obj:`"dense"` will work only on small graphs since it enumerates
            all possible edges
            :obj:`"auto"` will automatically choose the best method
            (default: :obj:`"auto"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> negative_sampling(edge_index)
        tensor([[3, 0, 0, 3],
                [2, 3, 2, 1]])

        >>> # For bipartite graph
        >>> negative_sampling(edge_index, num_nodes=(3, 4))
        tensor([[0, 2, 2, 1],
                [2, 2, 1, 3]])
    """
    assert method in ["sparse", "dense", "auto"]

    if num_nodes is None:
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

    if isinstance(num_nodes, int):
        size = (num_nodes, num_nodes)
        bipartite = False
    else:
        size = num_nodes
        bipartite = True
        force_undirected = False

    num_edges = edge_index.size(1)
    num_tot_edges = size[0] * size[1]

    if num_neg_samples is None:
        num_neg_samples = min(num_edges, num_tot_edges - num_edges)

    if force_undirected:
        num_neg_samples = num_neg_samples // 2

    # transform a pair (u,v) in an edge id
    edge_id = edge_index_to_vector_id(edge_index, size)
    edge_id, _ = index_sort(edge_id, max_value=num_tot_edges)

    # probability to randomly pick a negative edge
    prob_neg_edges = 1 - (num_edges / num_tot_edges)
    method = get_method(method, prob_neg_edges)

    k = None
    if method == "sparse":
        if prob_neg_edges >= _MIN_PROB_EDGES:
            # the probability of sampling non-existing edge is high,
            # so the sparse method should be ok
            if prob_neg_edges <= _MAX_PROB_EDGES:
                k = int(1.5 * num_neg_samples)
                warnings.warn(
                    "The probability of sampling a negative edge is low! "
                    "It could be that the number of sampled edges is smaller!"
                )
            else:
                k = int(num_neg_samples / prob_neg_edges)
        else:
            # the probability is too low, but sparse has been requsted!
            warnings.warn(
                f"The probability of sampling a negative edge is too low (less than {100 * _MIN_PROB_EDGES:0.0f}%)!"
                "Consider using dense sampling since O(E) is near to O(N^2), "
                "and there is little/no memory advantage in using a sparse method!"
            )
            k = int(2 * num_neg_samples)

    guess_edge_index, guess_edge_id = sample_almost_k_edges(
        size,
        k,
        force_undirected=force_undirected,
        remove_self_loops=not bipartite,
        method=method,
        device=edge_index.device,
    )

    neg_edge_mask = get_neg_edge_mask(edge_id, guess_edge_id)

    # we fiter the guessed id to maintain only the negative ones
    neg_edge_index = guess_edge_index[:, neg_edge_mask]

    if neg_edge_index.shape[-1] > num_neg_samples:
        neg_edge_index = neg_edge_index[:, :num_neg_samples]

    assert neg_edge_index is not None

    if force_undirected:
        neg_edge_index = to_undirected(neg_edge_index)

    return neg_edge_index


def batched_negative_edge_sampling(
    edge_index: Tensor,
    batch: Union[Tensor, Tuple[Tensor, Tensor]],
    num_neg_samples: Optional[int] = None,
    method: str = "auto",
    force_undirected: bool = False,
) -> Tensor:
    r"""Samples random negative edges of multiple graphs given by
    :attr:`edge_index` and :attr:`batch`.

    Args:
        edge_index (LongTensor): The edge indices.
        batch (LongTensor or Tuple[LongTensor, LongTensor]): Batch vector
            :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which assigns each
            node to a specific example.
            If given as a tuple, then :obj:`edge_index` is interpreted as a
            bipartite graph connecting two different node types.
        num_neg_samples (int, optional): The number of negative samples to
            return for each graph in the batch. If set to :obj:`None`,
            will try to return a negative edge
            for every positive edge. (default: :obj:`None`)
        method (str, optional): The method to use for negative sampling,
            *i.e.* :obj:`"sparse"`, :obj:`"dense"`, or :obj:`"auto"`.
            This is a memory/runtime trade-off.
            :obj:`"sparse"` will work on any graph of any size, but it could
            retrieve a different number of negative samples
            :obj:`"dense"` will work only on small graphs since it enumerates
            all possible edges
            :obj:`"auto"` will automatically choose the best method
            (default: :obj:`"auto"`)
        force_undirected (bool, optional): If set to :obj:`True`, sampled
            negative edges will be undirected. (default: :obj:`False`)

    :rtype: LongTensor

    Examples:
        >>> # Standard usage
        >>> edge_index = torch.as_tensor([[0, 0, 1, 2], [0, 1, 2, 3]])
        >>> edge_index = torch.cat([edge_index, edge_index + 4], dim=1)
        >>> edge_index
        tensor([[0, 0, 1, 2, 4, 4, 5, 6],
                [0, 1, 2, 3, 4, 5, 6, 7]])
        >>> batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1])
        >>> batched_negative_sampling(edge_index, batch)
        tensor([[3, 1, 3, 2, 7, 7, 6, 5],
                [2, 0, 1, 1, 5, 6, 4, 4]])

        >>> # For bipartite graph
        >>> edge_index1 = torch.as_tensor([[0, 0, 1, 1], [0, 1, 2, 3]])
        >>> edge_index2 = edge_index1 + torch.tensor([[2], [4]])
        >>> edge_index3 = edge_index2 + torch.tensor([[2], [4]])
        >>> edge_index = torch.cat([edge_index1, edge_index2, edge_index3], dim=1)
        >>> edge_index
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 0,  1,  2,  3,  4,  5,  6,  7,  8,  9, 10, 11]])
        >>> src_batch = torch.tensor([0, 0, 1, 1, 2, 2])
        >>> dst_batch = torch.tensor([0, 0, 0, 0, 1, 1, 1, 1, 2, 2, 2, 2])
        >>> batched_negative_sampling(edge_index, (src_batch, dst_batch))
        tensor([[ 0,  0,  1,  1,  2,  2,  3,  3,  4,  4,  5,  5],
                [ 2,  3,  0,  1,  6,  7,  4,  5, 10, 11,  8,  9]])
    """
    if isinstance(batch, Tensor):
        src_batch, dst_batch = batch, batch
    else:
        src_batch, dst_batch = batch[0], batch[1]

    split = degree(src_batch[edge_index[0]], dtype=torch.long).tolist()
    edge_indices = torch.split(edge_index, split, dim=1)

    num_src = degree(src_batch, dtype=torch.long)
    cum_src = cumsum(num_src)[:-1]

    if isinstance(batch, Tensor):
        num_nodes = num_src.tolist()
        ptr = cum_src
    else:
        num_dst = degree(dst_batch, dtype=torch.long)
        cum_dst = cumsum(num_dst)[:-1]

        num_nodes = torch.stack([num_src, num_dst], dim=1).tolist()
        ptr = torch.stack([cum_src, cum_dst], dim=1).unsqueeze(-1)

    neg_edge_indices = []
    for i, edge_index in enumerate(edge_indices):
        edge_index = edge_index - ptr[i]
        neg_edge_index = negative_edge_sampling(
            edge_index, num_nodes[i], num_neg_samples, method, force_undirected
        )
        neg_edge_index += ptr[i]
        neg_edge_indices.append(neg_edge_index)

    return torch.cat(neg_edge_indices, dim=1)


###############################################################################


def get_method(method: str, prob_neg_edges: float) -> str:
    # prefer the dense method if the graph is small
    auto_method = "dense" if prob_neg_edges < _MIN_PROB_EDGES else "sparse"
    method = auto_method if method == "auto" else method

    return method


def sample_almost_k_edges(
    size: Tuple[int, int],
    k: Optional[int],
    force_undirected: bool,
    remove_self_loops: bool,
    method: str,
    device: Optional[Union[torch.device, str]] = None,
) -> Tuple[Tensor, Tensor]:
    assert method in ["sparse", "dense"]
    N1, N2 = size
    tot_edges = N1 * N2
    if method == "sparse":
        assert k is not None
        k = 2 * k if force_undirected else k
        if k > tot_edges:
            k = tot_edges

        new_edge_id = torch.randint(tot_edges, (k,), device=device)
        # remove duplicates
        new_edge_id = torch.unique(new_edge_id)
    else:
        new_edge_id = torch.randperm(tot_edges, device=device)

    new_edge_index = torch.stack(vector_id_to_edge_index(new_edge_id, size), dim=0)

    if remove_self_loops:
        not_in_diagonal = new_edge_index[0] != new_edge_index[1]
        new_edge_index = new_edge_index[:, not_in_diagonal]
        new_edge_id = new_edge_id[not_in_diagonal]

    if force_undirected:
        # we consider only the upper part, i.e. col_idx > row_idx
        in_upper_part = new_edge_index[1] > new_edge_index[0]
        new_edge_index = new_edge_index[:, in_upper_part]
        new_edge_id = new_edge_id[in_upper_part]

    return new_edge_index, new_edge_id


def get_neg_edge_mask(edge_id: Tensor, guess_edge_id: Tensor) -> Tensor:
    num_edges = edge_id.size(0)
    pos = torch.searchsorted(edge_id, guess_edge_id)
    # pos contains the position where to insert the guessed id
    # to maintain the edge_id sort. There are two cases for new_id:
    # 1) if pos == num_edges (it means that we should add it at the end)
    # 2) if pos != num_edges but the id in position pos != from the guessed one
    # negative edge from case 1)
    neg_edge_mask = torch.eq(pos, num_edges)
    not_neg_edge_mask = torch.logical_not(neg_edge_mask)
    # negative edge from case 2)
    neg_edge_mask[not_neg_edge_mask] = (
        edge_id[pos[not_neg_edge_mask]] != guess_edge_id[not_neg_edge_mask]
    )
    return neg_edge_mask


def edge_index_to_vector_id(
    edge_index: Union[Tensor, Tuple[Tensor, Tensor]],
    size: Tuple[int, int],
) -> Tensor:
    row, col = edge_index
    return (row * size[1]).add_(col)


def vector_id_to_edge_index(
    vector_id: Tensor,
    size: Tuple[int, int],
) -> Tuple[Tensor, Tensor]:
    row, col = vector_id // size[1], vector_id % size[1]
    return row, col
