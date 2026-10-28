from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import coalesce
from torch_geometric.utils.num_nodes import maybe_num_nodes

from tgp.imports import is_sparsetensor
from tgp.select import SelectOutput
from tgp.utils import (
    connectivity_to_edge_index,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
    postprocess_adj_pool_sparse,
)
from tgp.utils.typing import ConnectionType


class Connect(torch.nn.Module):
    r"""An abstract base class implementing the :math:`\texttt{connect}` operator.

    Specifically, :math:`\texttt{connect}` determines for each pair of supernodes the
    presence or absence of an edge based on the existing edges between the
    nodes in the two supernodes.
    """

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(
        self,
        edge_index: Adj,
        so: SelectOutput,
        *,
        edge_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Adj, Optional[Tensor]]:
        r"""Forward pass.

        Args:
            edge_index (torch.Tensor):
                The original edge indices.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
            edge_weight (torch.Tensor, optional):
                The original edge weights.
                (default: :obj:`None`)
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


def sparse_connect(
    edge_index: Adj,
    edge_weight: OptTensor = None,
    node_index: Optional[Tensor] = None,
    cluster_index: Optional[Tensor] = None,
    num_nodes: int = None,
    num_supernodes: int = None,
    remove_self_loops: bool = True,
    reduce_op: ConnectionType = "sum",
    edge_weight_norm: bool = False,
    batch_pooled: Optional[Tensor] = None,
    degree_norm: bool = False,
) -> Tuple[Adj, OptTensor]:
    r"""Connects original edges using explicit original-node-to-pooled-node pairs.

    ``node_index[i]`` maps to ``cluster_index[i]``; their order need not match
    original node order. Original nodes absent from ``node_index`` are dropped,
    together with their incident edges. Each original node may occur at most
    once in ``node_index``.

    If ``cluster_index`` is omitted, consecutive pooled IDs are assigned in
    ``node_index`` order. If ``node_index`` is omitted, ``cluster_index`` must
    contain one assignment for every original node, in original node order.

    Selecting a proper subset preserves surviving edge order and parallel
    edges. Assigning every original node coalesces parallel edges with
    ``reduce_op``, including when all nodes are retained by TopK. The existing
    self-loop and normalization options are applied after this step. Output
    connectivity has the same format as the input; sparse formats embed the
    weights and may coalesce parallel entries during conversion.

    Provide ``num_nodes`` explicitly when an edge list does not include every
    original node. ``num_supernodes`` defaults to the largest assigned pooled
    ID plus one, or zero for an empty assignment.

    Raises:
        RuntimeError: If neither index vector is supplied.
        ValueError: If the index vectors are not one-dimensional or their
            lengths do not describe a valid sparse assignment.
    """
    to_sparsetensor = False
    to_torch_coo = False
    if is_sparsetensor(edge_index):
        to_sparsetensor = True
    elif isinstance(edge_index, Tensor) and edge_index.is_sparse:
        to_torch_coo = True
    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)

    if node_index is None and cluster_index is None:
        raise RuntimeError(
            "At least one of node_index and cluster_index must be provided."
        )
    if node_index is None:
        if cluster_index.numel() != num_nodes:
            raise ValueError(
                "cluster_index must contain num_nodes entries when node_index is omitted."
            )
        node_index = torch.arange(num_nodes, device=cluster_index.device)
    if cluster_index is None:
        cluster_index = torch.arange(node_index.numel(), device=node_index.device)
    if node_index.dim() != 1 or cluster_index.dim() != 1:
        raise ValueError("node_index and cluster_index must be one-dimensional.")
    if node_index.numel() != cluster_index.numel():
        raise ValueError("node_index and cluster_index must have the same length.")
    if node_index.numel() > num_nodes:
        raise ValueError("node_index cannot contain more than num_nodes entries.")
    if num_supernodes is None:
        num_supernodes = int(cluster_index.max()) + 1 if cluster_index.numel() else 0

    node_to_cluster = edge_index.new_full((num_nodes,), -1)
    node_to_cluster[node_index] = cluster_index
    pooled_edges = node_to_cluster[edge_index]
    edge_mask = (pooled_edges >= 0).all(dim=0)
    edge_index = pooled_edges[:, edge_mask]
    if edge_weight is not None:
        edge_weight = edge_weight[edge_mask]

    # Preserve the historical aggregation policy independently of node ordering.
    if node_index.numel() == num_nodes:
        edge_index, edge_weight = coalesce(
            edge_index, edge_weight, num_nodes=num_supernodes, reduce=reduce_op
        )

    edge_index, edge_weight = postprocess_adj_pool_sparse(
        edge_index,
        edge_weight,
        num_nodes=num_supernodes,
        remove_self_loops=remove_self_loops,
        degree_norm=degree_norm,
        edge_weight_norm=edge_weight_norm,
        batch_pooled=batch_pooled,
    )

    if to_sparsetensor:
        edge_index = connectivity_to_sparsetensor(
            edge_index, edge_weight, num_supernodes
        )
        edge_weight = None
    elif to_torch_coo:
        edge_index = connectivity_to_torch_coo(edge_index, edge_weight, num_supernodes)
        edge_weight = None

    return edge_index, edge_weight


class SparseConnect(Connect):
    r"""The :math:`\texttt{connect}` operator for sparse methods
    where each node is assigned at most one supernode.
    This is, for example, the case of one-over-:math:`K` methods
    such as :class:`~tgp.select.GraclusSelect`, :class:`~tgp.select.NDPSelect`, and
    :class:`~tgp.select.KMISSelect`.

    It also works for scoring-based methods such as
    :class:`~tgp.select.TopkSelect` that compute the pooled adjacency as

    .. math::
            \mathbf{A}_{\text{pool}} = \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where :math:`\mathbf{i}` denotes the set of supernodes.

    The pairing of ``so.node_index`` and ``so.cluster_index`` determines pooled
    node IDs, independently of storage order. Proper subsets preserve parallel
    edges in edge-list output; full assignments coalesce them with ``reduce_op``.

    Args:
        reduce_op (~tgp.utils.typing.ConnectionType, optional):
            The aggregation function for edges sharing pooled endpoints when
            every original node is assigned. Can be
            any string admitted by :obj:`~torch_geometric.utils.scatter` (e.g., ``'sum'``, ``'mean'``,
            ``'max'``) or any :class:`~tgp.utils.typing.ConnectionType`.
            (default: :obj:`sum`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value
            per graph.
            (default: :obj:`False`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized using
            :math:`D^{-1/2} A D^{-1/2}` where :math:`D` is the degree matrix.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        reduce_op: ConnectionType = "sum",
        remove_self_loops: bool = True,
        edge_weight_norm: bool = False,
        degree_norm: bool = False,
    ):
        super().__init__()
        self.reduce_op = reduce_op
        self.remove_self_loops = remove_self_loops
        self.edge_weight_norm = edge_weight_norm
        self.degree_norm = degree_norm

    def forward(
        self,
        edge_index: Adj,
        so: SelectOutput,
        *,
        edge_weight: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Adj, Optional[Tensor]]:
        r"""Forward pass.

        Args:
            edge_index (~torch_geometric.typing.Adj):
                The connectivity matrix.
                It can either be a ``torch_sparse.SparseTensor`` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
            edge_weight (~torch.Tensor, optional): A vector of shape
                :math:`[E]` containing the weights of the edges.
                (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                Batch vector which assigns each supernode to a specific graph.
                Required when edge_weight_norm=True for per-graph normalization.
                (default: :obj:`None`)

        Returns:
            (~torch_geometric.typing.Adj, ~torch.Tensor or None):
            The pooled adjacency matrix and the edge weights.
            If the pooled adjacency is a ``torch_sparse.SparseTensor``,
            returns :obj:`None` as the edge weights.
        """
        if self.edge_weight_norm and batch_pooled is None:
            raise AssertionError(
                "edge_weight_norm=True but batch_pooled=None. "
                "batch_pooled parameter is required for per-graph normalization in SparseConnect."
            )

        out = sparse_connect(
            edge_index,
            edge_weight,
            node_index=so.node_index,
            cluster_index=so.cluster_index,
            num_nodes=so.num_nodes,
            num_supernodes=so.num_supernodes,
            remove_self_loops=self.remove_self_loops,
            reduce_op=self.reduce_op,
            edge_weight_norm=self.edge_weight_norm,
            batch_pooled=batch_pooled,
            degree_norm=self.degree_norm,
        )

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"reduce_op={self.reduce_op}, "
            f"remove_self_loops={self.remove_self_loops}, "
            f"edge_weight_norm={self.edge_weight_norm}, "
            f"degree_norm={self.degree_norm})"
        )
