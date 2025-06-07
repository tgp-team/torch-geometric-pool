from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import coalesce, subgraph
from torch_geometric.utils import remove_self_loops as rsl
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.select import SelectOutput
from tgp.utils import connectivity_to_edge_index, connectivity_to_sparse_tensor
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
    node_index: Tensor = None,
    cluster_index: Optional[Tensor] = None,
    num_nodes: int = None,
    num_clusters: int = None,
    remove_self_loops: bool = True,
    reduce_op: ConnectionType = "sum",
) -> Tuple[Adj, OptTensor]:
    r"""Connects the nodes in the coarsened graph."""
    to_sparse = False

    if isinstance(edge_index, SparseTensor):
        edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)
        to_sparse = True

    num_nodes = maybe_num_nodes(edge_index, num_nodes)
    if node_index is not None and len(node_index) < num_nodes:
        edge_index, edge_weight = subgraph(
            node_index, edge_index, edge_weight, relabel_nodes=True, num_nodes=num_nodes
        )
    elif cluster_index is not None and len(cluster_index) == num_nodes:
        edge_index = cluster_index[edge_index]
        edge_index, edge_weight = coalesce(
            edge_index, edge_weight, num_nodes=num_clusters, reduce=reduce_op
        )
    else:
        raise RuntimeError

    if remove_self_loops:
        edge_index, edge_weight = rsl(edge_index, edge_weight)

    if to_sparse:
        edge_index = connectivity_to_sparse_tensor(
            edge_index, edge_weight, num_clusters
        )
        edge_weight = None

    return edge_index, edge_weight


class SparseConnect(Connect):
    r"""The :math:`\texttt{connect}` operator for sparse methods
    where each node is assigned at most one supernode.
    This is, for example, the case of one-over-:math:`K` methods
    such as :class:`~tgp.select.GraclusSelect`, :class:`~tgp.select.NDPSelect`, and
    :class:`~tgp.select.KMISSelect`.

    It also works of scoring-based methods such as
    :class:`~tgp.select.TopkSelect` that compute the pooled adjacency as

    .. math::
            \mathbf{A}_{\text{pool}} = \mathbf{A}_{\mathbf{i},\mathbf{i}},

    where :math:`\mathbf{i}` denotes the set of supernodes.

    Args:
        reduce_op (~tgp.utils.typing.ReduceType, optional):
            The aggregation function to be applied to nodes in the same cluster. Can be
            any string admitted by :obj:`~torch_geometric.utils.scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`~tgp.utils.typing.ReduceType`.
            (default: :obj:`sum`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
    """

    def __init__(
        self, reduce_op: ConnectionType = "sum", remove_self_loops: bool = True
    ):
        super().__init__()
        self.reduce_op = reduce_op
        self.remove_self_loops = remove_self_loops

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
            edge_index (~torch_geometric.typing.Adj):
                The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
            edge_weight (~torch.Tensor, optional): A vector of shape
                :math:`[E]` containing the weights of the edges.
                (default: :obj:`None`)

        Returns:
            (~torch_geometric.typing.Adj, ~torch.Tensor or None):
            The pooled adjacency matrix and the edge weights.
            If the pooled adjacency is a :obj:`~torch_sparse.SparseTensor`,
            returns :obj:`None` as the edge weights.
        """
        out = sparse_connect(
            edge_index,
            edge_weight,
            node_index=so.node_index,
            cluster_index=so.cluster_index,
            num_nodes=so.num_nodes,
            num_clusters=so.num_clusters,
            remove_self_loops=self.remove_self_loops,
            reduce_op=self.reduce_op,
        )

        return out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"reduce_op={self.reduce_op}, "
            f"remove_self_loops={self.remove_self_loops})"
        )
