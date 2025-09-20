from typing import Optional

import torch
from torch_geometric.nn.dense import Linear
from torch_geometric.typing import Adj, OptTensor, PairTensor, Tensor
from torch_geometric.utils import scatter, to_undirected
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.imports import HAS_TORCH_SCATTER

if HAS_TORCH_SCATTER:
    from torch_scatter import scatter_add, scatter_max, scatter_min
from tgp.select import Select, SelectOutput
from tgp.utils import (
    check_and_filter_edge_weights,
    connectivity_to_edge_index,
    connectivity_to_row_col,
    connectivity_to_sparse_tensor,
    weighted_degree,
)
from tgp.utils.typing import SinvType


def degree_scorer(
    edge_index: Adj,
    edge_weight: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    dim: int = 1,
):
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)

    # Check if edge_weight is a 1D tensor
    if edge_weight is not None and edge_weight.dim() > 1 and edge_weight.size(1) > 1:
        raise ValueError(
            "`edge_weight` must be a 1D tensor, but got "
            f"`edge_weight.size(1)={edge_weight.size(1)}`"
        )

    neigh = edge_index[dim]
    deg = weighted_degree(neigh, edge_weight, num_nodes)
    return deg.float()


def maximal_independent_set(
    edge_index: Adj, order_k: int = 1, perm: OptTensor = None
) -> Tensor:
    r"""Returns a Maximal :math:`k`-Independent Set of a graph, i.e., a set of
    nodes (as a :class:`ByteTensor`) such that none of them are :math:`k`-hop
    neighbors, and any node in the graph has a :math:`k`-hop neighbor in the
    returned set.

    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method follows `Blelloch's Alogirithm
    <https://arxiv.org/abs/1202.3205>`_ for :math:`k = 1`, and its
    generalization by `Bacciu et al. <https://arxiv.org/abs/2208.03523>`_ for
    higher values of :math:`k`.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        order_k (int): The :math:`k`-th order (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: :class:`ByteTensor`
    """
    n = maybe_num_nodes(edge_index)
    row, col = connectivity_to_row_col(edge_index)
    device = row.device

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    mis = torch.zeros(n, dtype=torch.bool, device=device)
    mask = mis.clone()
    min_rank = rank.clone()

    while not mask.all():
        for _ in range(order_k):
            if HAS_TORCH_SCATTER:
                min_neigh = torch.full_like(min_rank, fill_value=n)
                scatter_min(min_rank[row], col, out=min_neigh)
                torch.minimum(min_neigh, min_rank, out=min_rank)  # self-loops
            else:
                min_scatter = scatter(
                    src=min_rank[row], index=col, dim=0, dim_size=n, reduce="min"
                )
                # Compute a count for each node to detect which indices received no update:
                counts = scatter(
                    src=torch.ones_like(min_rank[row]),
                    index=col,
                    dim=0,
                    dim_size=n,
                    reduce="sum",
                )
                # For indices with no incoming message, assign the identity value (n):
                min_scatter[counts == 0] = n
                min_rank = torch.minimum(min_scatter, min_rank)  # self-loops

        mis = mis | torch.eq(rank, min_rank)
        mask = mis.clone().byte()

        for _ in range(order_k):
            if HAS_TORCH_SCATTER:
                max_neigh = torch.full_like(mask, fill_value=0)
                scatter_max(mask[row], col, out=max_neigh)
                torch.maximum(max_neigh, mask, out=mask)  # self-loops
            else:
                mask_int = mask.long()
                max_scatter = scatter(
                    src=mask_int[row], index=col, dim=0, dim_size=n, reduce="max"
                )
                mask_int = torch.maximum(mask_int, max_scatter)  # self-loops
                mask = mask_int.bool()

        mask = mask.to(dtype=torch.bool)
        min_rank = rank.clone()
        min_rank[mask] = n

    return mis


def maximal_independent_set_cluster(
    edge_index: Adj, order_k: int = 1, perm: OptTensor = None
) -> PairTensor:
    r"""Computes the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    clustering of a graph, as defined in `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_.

    The algorithm greedily selects the nodes in their canonical order. If a
    permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.
    This method returns both the :math:`k`-MIS and the clustering, where the
    :math:`c`-th cluster refers to the :math:`c`-th element of the
    :math:`k`-MIS.

    Args:
        edge_index (Tensor or SparseTensor): The graph connectivity.
        order_k (int): The :math:`k`-th order (defaults to 1).
        perm (LongTensor, optional): Permutation vector. Must be of size
            :obj:`(n,)` (defaults to :obj:`None`).
    :rtype: (:class:`ByteTensor`, :class:`LongTensor`)
    """
    mis = maximal_independent_set(edge_index=edge_index, order_k=order_k, perm=perm)
    n, device = mis.size(0), mis.device

    row, col = connectivity_to_row_col(edge_index)

    if perm is None:
        rank = torch.arange(n, dtype=torch.long, device=device)
    else:
        rank = torch.zeros_like(perm)
        rank[perm] = torch.arange(n, dtype=torch.long, device=device)

    min_rank = torch.full((n,), fill_value=n, dtype=torch.long, device=device)
    rank_mis = rank[mis]
    min_rank[mis] = rank_mis

    for _ in range(order_k):
        min_neigh = torch.full_like(min_rank, fill_value=n)
        scatter_min(min_rank[row], col, out=min_neigh)
        torch.minimum(min_neigh, min_rank, out=min_rank)

    _, clusters = torch.unique(min_rank, return_inverse=True)
    perm = torch.argsort(rank_mis)
    return mis, perm[clusters]


class KMISSelect(Select):
    r"""Computes the node assignments following the Maximal :math:`k`-Independent Set (:math:`k`-MIS)
    algorithm, as defined in  the paper `"Generalizing Downsampling from
    Regular Data to Graphs" <https://arxiv.org/abs/2208.03523>`_ (Bacciu et al., AAAI 2023).

    To compute the :math:`k`-MIS, the algorithm greedily selects the nodes
    in their canonical order. If a permutation :obj:`perm` is provided, the nodes are extracted following
    that permutation instead.

    Args:
        in_channels (int, optional):
            Size of each input sample. Ignored if :obj:`scorer` is not
            :obj:`"linear"`. (default: :obj:`None`)
        order_k (int):
            The :math:`k`-th order for the independent set. (default: :obj:`1`)
        scorer (str):
            A function that computes a score for each node. Nodes with higher score
            have a higher chance of being selected for pooling. It can be one of:

            - :obj:`"linear"` (default): Uses a sigmoid-activated linear layer to
              compute the scores. :obj:`in_channels`
              must be set when using this option.
            - :obj:`"random"`: Assigns a random score in :math:`[0, 1]` to each
              node.
            - :obj:`"constant"`: Assigns a constant score of :math:`1` to each node.
            - :obj:`"canonical"`: Assigns the score :math:`-i` to the :math:`i`-th
              node.
            - :obj:`"degree"`: Uses the degree of each node as the score.
        score_heuristic (str, optional):
            Heuristic to increase the total score of selected nodes. Given an initial
            score vector :math:`\mathbf{s} \in \mathbb{R}^n`, options include:

            - :obj:`None`: No heuristic applied.
            - :obj:`"greedy"` (default): Computes the updated score
              :math:`\mathbf{s}'` as

              .. math::
                  \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} + \mathbf{I})^k
                  \mathbf{1}

              where :math:`\oslash` is element-wise division.
            - :obj:`"w-greedy"`: Computes the updated score :math:`\mathbf{s}'` as

              .. math::
                  \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} + \mathbf{I})^k
                  \mathbf{s}
        force_undirected (bool, optional):
            Whether to force the input graph to be undirected. (default: :obj:`False`)
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
        node_dim (int, optional):
            The node dimension in the input feature matrix. (default: :obj:`-2`)
    """

    _heuristics = {None, "greedy", "w-greedy"}
    _scorers = {"linear", "degree", "random", "constant", "canonical"}

    def __init__(
        self,
        in_channels: Optional[int] = None,
        order_k: int = 1,
        scorer: str = "linear",
        score_heuristic: Optional[str] = "greedy",
        force_undirected: bool = False,
        s_inv_op: SinvType = "transpose",
        node_dim: int = -2,
    ):
        super(KMISSelect, self).__init__()

        assert score_heuristic in self._heuristics, (
            f"Unrecognized `score_heuristic` value: {score_heuristic}"
        )
        assert scorer in self._scorers, f"Unrecognized `scorer` value: {scorer}"

        self.order_k = order_k
        self.scorer = scorer
        self.score_heuristic = score_heuristic
        self.node_dim = node_dim
        self.force_undirected = force_undirected
        self.s_inv_op = s_inv_op

        if scorer == "linear":
            if isinstance(in_channels, list):
                in_channels = in_channels[0]
            self.lin = Linear(
                in_channels=in_channels, out_channels=1, weight_initializer="uniform"
            )

    def _apply_heuristic(self, x: Tensor, adj: SparseTensor) -> Tensor:
        if self.score_heuristic is None:
            return x

        row, col = connectivity_to_row_col(adj)
        x = x.view(-1)

        k_sums = torch.ones_like(x) if self.score_heuristic == "greedy" else x.clone()

        if HAS_TORCH_SCATTER:
            for _ in range(self.order_k):
                scatter_add(k_sums[row], col, out=k_sums)
        else:
            for _ in range(self.order_k):
                k_sums += scatter(
                    k_sums[row], col, dim=0, dim_size=k_sums.size(0), reduce="add"
                )

        return x / k_sums

    def _scorer(
        self,
        adj: SparseTensor,
        x: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
    ) -> Tensor:
        device = adj.device()

        if self.scorer == "linear":
            assert x is not None, "x must be provided when scorer is 'linear'"
            return self.lin(x).sigmoid()

        if self.scorer == "random":
            return torch.rand((num_nodes, 1), device=device)

        if self.scorer == "constant":
            return torch.ones((num_nodes, 1), device=device)

        if self.scorer == "canonical":
            return -torch.arange(num_nodes, device=device).view(-1, 1)

        if self.scorer == "degree":
            return degree_scorer(edge_index=adj, num_nodes=num_nodes)
        raise ValueError(f"Unrecognized `scorer` value: {self.scorer}")

    def forward(
        self,
        *,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        x: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            edge_index (~torch_geometric.typing.Adj):
                The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or  :math:`[E, 1]` containing the weights of the edges.
                (default: :obj:`None`)
            x (~torch.Tensor, optional):
                The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)
            num_nodes (int, optional):
                The total number of nodes of the graphs in the batch.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of the :math:`\texttt{select}` operator.
        """
        size_x = x.size(0) if x is not None else None
        num_nodes = (
            num_nodes if num_nodes is not None else maybe_num_nodes(edge_index, size_x)
        )

        if self.force_undirected:
            if isinstance(edge_index, SparseTensor):
                edge_index, edge_weight = connectivity_to_edge_index(edge_index)
            edge_index, edge_weight = to_undirected(
                edge_index, edge_weight, num_nodes, reduce="max"
            )
        edge_weight = check_and_filter_edge_weights(edge_weight)
        adj = connectivity_to_sparse_tensor(
            edge_index, edge_weight, num_nodes=num_nodes
        )
        score = self._scorer(adj, x, num_nodes=num_nodes)
        updated_score = self._apply_heuristic(score, adj)
        perm = torch.argsort(updated_score.view(-1), 0, descending=True)

        mis, cluster = maximal_independent_set_cluster(adj, self.order_k, perm)
        mis = mis.nonzero().view(-1)

        so = SelectOutput(
            cluster_index=cluster,
            num_nodes=num_nodes,
            num_supernodes=mis.size(0),
            weight=score.view(-1),
            s_inv_op=self.s_inv_op,
            mis=mis,
        )

        return so

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"order_k={self.order_k}, "
            f"scorer={self.scorer}, "
            f"score_heuristic={self.score_heuristic}, "
            f"force_undirected={self.force_undirected}, "
            f"s_inv_op={self.s_inv_op})"
        )
