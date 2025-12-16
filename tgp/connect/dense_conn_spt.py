from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops as rsl
from torch_scatter import scatter

from tgp.connect import Connect
from tgp.imports import is_torch_sparse_tensor
from tgp.select import SelectOutput
from tgp.utils.ops import (
    connectivity_to_edge_index,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
)


class DenseConnectSPT(Connect):
    r"""A :math:`\texttt{connect}` operator to be used when the assignment matrix
    :math:`\mathbf{S}` in the :class:`~tgp.select.SelectOutput` is a :class:`SparseTensor`
    with a block diagonal structure, and each block is dense.

    The pooled adjacency matrix is computed as:

    .. math::
        \mathbf{A}_{\mathrm{pool}} =
        \mathbf{S}^{\top}\mathbf{A}\mathbf{S}_{\mathrm{inv}}^{\top}

    Args:
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = False,
        edge_weight_norm: bool = False,
    ):
        super().__init__()

        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm
        self.edge_weight_norm = edge_weight_norm

    def forward(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        so: SelectOutput = None,
        batch_pooled: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Adj, Optional[Tensor]]:
        r"""Forward pass.

        Args:
            edge_index (~torch_geometric.typing.Adj):
                The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch, representing
                the list of edges.
            edge_weight (~torch.Tensor, optional): A vector of shape  :math:`[E]` or :math:`[E, 1]`
                containing the weights of the edges.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
            batch_pooled (~torch.Tensor, optional):
                Batch vector which assigns each supernode to a specific graph.
                Required when edge_weight_norm=True for per-graph normalization.
                (default: :obj:`None`)

        Returns:
            (~torch_geometric.typing.Adj, ~torch.Tensor or None):
            The pooled adjacency matrix and the edge weights.
            If the pooled adjacency is a :obj:`~torch_sparse.SparseTensor`,
            returns :obj:`None` as the edge weights.
        """
        if self.edge_weight_norm and batch_pooled is None:
            raise AssertionError(
                "edge_weight_norm=True but batch_pooled=None. "
                "batch_pooled parameter is required for per-graph normalization in DenseConnectSPT."
            )

        num_supernodes = so.s.size(1)
        to_sparsetensor = False
        to_edge_index = False
        if is_torch_sparse_tensor(edge_index):
            to_sparsetensor = True
        elif isinstance(edge_index, Tensor) and not edge_index.is_sparse:
            to_edge_index = True

        edge_index_coo = connectivity_to_torch_coo(edge_index, edge_weight)

        # Handle edge case: empty adjacency matrix
        if edge_index_coo._nnz() == 0:
            # No edges: create empty sparse tensor with correct size
            adj_pooled = torch.sparse_coo_tensor(
                torch.empty((2, 0), dtype=torch.long, device=edge_index_coo.device),
                torch.empty(
                    0, dtype=edge_index_coo.dtype, device=edge_index_coo.device
                ),
                size=(num_supernodes, num_supernodes),
            ).coalesce()
        else:
            # Compute: s^T @ edge_index @ s using torch.sparse.mm
            temp = torch.sparse.mm(so.s.transpose(-2, -1), edge_index_coo)
            adj_pooled = torch.sparse.mm(temp, so.s)

        edge_index, edge_weight = connectivity_to_edge_index(adj_pooled)

        if self.remove_self_loops:
            edge_index, edge_weight = rsl(edge_index, edge_weight)

        if self.degree_norm:
            if edge_weight is None:
                edge_weight = torch.ones(edge_index.size(1), device=edge_index.device)

            # Compute degree normalization D^{-1/2} A D^{-1/2}
            deg = scatter(
                edge_weight, edge_index[0], dim=0, dim_size=num_supernodes, reduce="sum"
            )
            deg[deg == 0] = 1.0  # Avoid division by zero
            deg_inv_sqrt = 1.0 / deg.sqrt()

            # Apply symmetric normalization to edge weights
            edge_weight = (
                edge_weight * deg_inv_sqrt[edge_index[0]] * deg_inv_sqrt[edge_index[1]]
            )

        if self.edge_weight_norm and edge_weight is not None:
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

        if to_sparsetensor:
            edge_index = connectivity_to_sparsetensor(
                edge_index, edge_weight, num_supernodes
            )
            edge_weight = None
        elif to_edge_index:
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )

        return edge_index, edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm}, "
            f"edge_weight_norm={self.edge_weight_norm})"
        )
