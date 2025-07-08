from typing import Callable, Optional, Tuple

import torch
import torch.nn.functional as F
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor, SparseTensor
from torch_geometric.utils import scatter, softmax

from tgp.select import Select, SelectOutput
from tgp.utils.typing import SinvType


def maximal_matching(
    edge_index: Adj, num_nodes: Optional[int] = None, perm: OptTensor = None
) -> Tensor:
    r"""Returns a Maximal Matching of a graph, i.e., a set of edges (as a
    :class:`ByteTensor`) such that none of them are incident to a common
    vertex, and any edge in the graph is incident to an edge in the returned
    set.

    The algorithm greedily selects the edges in their canonical order. If a
    permutation :obj:`perm` is provided, the edges are extracted following
    that permutation instead.

    This method implements `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        num_nodes (int, optional): The number of nodes in the graph.
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(m,)` (defaults to :obj:`None`).

    :rtype: :class:`ByteTensor`
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n, m = edge_index.size(0), edge_index.nnz()
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n, m = num_nodes, row.size(0)

        if n is None:
            n = edge_index.max().item() + 1

    if perm is None:
        rank = torch.arange(m, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(m, dtype=torch.long, device=device)

    match = torch.zeros(m, dtype=torch.bool, device=device)
    mask = torch.ones(m, dtype=torch.bool, device=device)

    # TODO: Use scatter's `out` and `include_self` arguments,
    #       when available, instead of adding self-loops
    max_rank = torch.full((n,), fill_value=n * n, dtype=torch.long, device=device)
    max_idx = torch.arange(n, dtype=torch.long, device=device)

    while mask.any():
        src = torch.cat([rank[mask], rank[mask], max_rank])
        idx = torch.cat([row[mask], col[mask], max_idx])
        node_rank = scatter(src, idx, reduce="min")
        edge_rank = torch.minimum(node_rank[row], node_rank[col])

        match = match | torch.eq(rank, edge_rank)

        unmatched = torch.ones(n, dtype=torch.bool, device=device)
        idx = torch.cat([row[match], col[match]], dim=0)
        unmatched[idx] = False
        mask = mask & unmatched[row] & unmatched[col]

    return match


def maximal_matching_cluster(
    edge_index: Adj, num_nodes: Optional[int] = None, perm: OptTensor = None
) -> Tuple[Tensor, Tensor]:
    r"""Computes the Maximal Matching clustering of a graph, where the
    matched edges form 2-element clusters while unmatched vertices are treated
    as singletons.

    The algorithm greedily selects the edges in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.

    This method returns both the matching and the clustering.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        num_nodes (int, optional): The number of nodes in the graph.
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(m,)` (defaults to :obj:`None`).

    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    if isinstance(edge_index, SparseTensor):
        row, col, _ = edge_index.coo()
        device = edge_index.device()
        n = edge_index.size(0)
    else:
        row, col = edge_index[0], edge_index[1]
        device = row.device
        n = num_nodes

        if n is None:
            n = edge_index.max().item() + 1

    match = maximal_matching(edge_index, num_nodes, perm)
    cluster = torch.arange(n, dtype=torch.long, device=device)
    cluster[col[match]] = row[match]

    _, cluster = torch.unique(cluster, return_inverse=True)
    return match, cluster


class EdgeContractionSelect(Select):
    r"""The :math:`\texttt{select}` operator from the papers `"Towards Graph Pooling by Edge
    Contraction" <https://graphreason.github.io/papers/17.pdf>`_ (Diehl et al. 2019) and
    `"Edge Contraction Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/1905.10990>`_ (Diehl, 2019).
    This implementation is based on the paper `"Revisiting Edge Pooling in Graph Neural Networks"
    <https://www.esann.org/sites/default/files/proceedings/2022/ES2022-92.pdf>`_ (Landolfi, 2022).

    In short, a score is computed for each edge.
    Edges are contracted iteratively according to that score unless one of
    their nodes has already been part of a contracted edge.

    Args:
        in_channels (int):
            Size of each input sample.
        edge_score_method (callable, optional):
            The function to apply to compute the edge score from raw edge scores. By default,
            this is the softmax over all incoming edges for each node.
            This function takes in a :obj:`raw_edge_score` tensor of shape
            :obj:`[num_nodes]`, an :obj:`edge_index` tensor and the number of
            nodes :obj:`num_nodes`, and produces a new tensor of the same size
            as :obj:`raw_edge_score` describing normalized edge scores.
            Included functions are
            :func:`~tgp.select.EdgeContractionSelect.compute_edge_score_softmax`,
            :func:`~tgp.select.EdgeContractionSelect.compute_edge_score_tanh`, and
            :func:`~tgp.select.EdgeContractionSelect.compute_edge_score_sigmoid`.
            (default: :func:`~tgp.select.EdgeContractionSelect.compute_edge_score_softmax`)
        dropout (float, optional):
            The probability with which to drop edge scores during training.
            (default: :obj:`0.0`)
        add_to_edge_score (float, optional):
            A value to be added to each computed edge score.
            Adding this greatly helps with unpooling stability.
            (default: :obj:`0.5`)
    """

    def __init__(
        self,
        in_channels: int,
        edge_score_method: Optional[Callable] = None,
        dropout: Optional[float] = 0.0,
        add_to_edge_score: float = 0.5,
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__()
        self.in_channels = in_channels
        self.s_inv_op = s_inv_op
        if edge_score_method is None:
            edge_score_method = self.compute_edge_score_softmax
        self.compute_edge_score = edge_score_method
        self.add_to_edge_score = add_to_edge_score
        self.dropout = dropout

        self.lin = torch.nn.Linear(2 * in_channels, 1)

        self.reset_parameters()

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()

    @staticmethod
    def compute_edge_score_softmax(
        raw_edge_score: Tensor,
        edge_index: Tensor,
        num_nodes: int,
    ) -> Tensor:
        r"""Normalizes edge scores via softmax application."""
        return softmax(raw_edge_score, edge_index[1], num_nodes=num_nodes)

    @staticmethod
    def compute_edge_score_tanh(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via hyperbolic tangent application."""
        return torch.tanh(raw_edge_score)

    @staticmethod
    def compute_edge_score_sigmoid(
        raw_edge_score: Tensor,
        edge_index: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        r"""Normalizes edge scores via sigmoid application."""
        return torch.sigmoid(raw_edge_score)

    def forward(self, x: Tensor, edge_index: Tensor, **kwargs) -> SelectOutput:
        r"""Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            edge_index (~torch.Tensor):
                The edge indices. Is a tensor of of shape  :math:`[2, E]`,
                where :math:`E` is the number of edges in the batch.
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        e = torch.cat([x[edge_index[0]], x[edge_index[1]]], dim=-1)
        e = self.lin(e).view(-1)
        e = F.dropout(e, p=self.dropout, training=self.training)
        e = self.compute_edge_score(e, edge_index, x.size(0))
        e = e + self.add_to_edge_score

        perm = torch.argsort(e, descending=True)
        match, cluster = maximal_matching_cluster(
            edge_index, num_nodes=x.size(0), perm=perm
        )
        c = cluster.max() + 1
        new_edge_score = torch.ones(c, dtype=x.dtype, device=x.device)
        new_edge_score[cluster[edge_index[0, match]]] = e[match]

        so = SelectOutput(
            node_index=torch.arange(x.size(0), device=x.device),
            num_nodes=x.size(0),
            cluster_index=cluster,
            num_supernodes=c,
            weight=new_edge_score[cluster],
            s_inv_op=self.s_inv_op,
        )
        return so

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"edge_score_method={self.compute_edge_score.__name__}, "
            f"dropout={self.dropout}, "
            f"add_to_edge_score={self.add_to_edge_score}, "
            f"s_inv_op={self.s_inv_op})"
        )
