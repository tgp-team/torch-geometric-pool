from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from tgp.connect import Connect
from tgp.select import SelectOutput

EPS = 1e-15


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
    """

    def __init__(self, remove_self_loops: bool = True, degree_norm: bool = False):
        super().__init__()

        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm

    def forward(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        so: SelectOutput = None,
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
        if isinstance(edge_index, SparseTensor):
            adj_pooled = so.s.t() @ edge_index @ so.s
        elif isinstance(edge_index, Tensor):
            adj = SparseTensor.from_edge_index(
                edge_index, edge_weight, sparse_sizes=(so.s.size(0), so.s.size(0))
            )
            adj_pooled = so.s.t() @ adj @ so.s
        else:
            raise ValueError(
                f"Edge index must be of type {Adj}, got {type(edge_index)}"
            )

        if self.remove_self_loops:
            row, col, val = adj_pooled.coo()
            mask = row != col
            row, col, val = row[mask], col[mask], val[mask]

            # Rebuild adjacency without diagonal
            adj_pooled = SparseTensor(
                row=row, col=col, value=val, sparse_sizes=adj_pooled.sparse_sizes()
            )

        if self.degree_norm:
            deg = adj_pooled.sum(dim=1)

            deg_inv_sqrt = 1.0 / (deg + EPS).sqrt()

            # Recompute values to apply D^{-1/2} * A * D^{-1/2}
            row, col, val = adj_pooled.coo()
            val = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]

            adj_pooled = SparseTensor(
                row=row, col=col, value=val, sparse_sizes=adj_pooled.sparse_sizes()
            ).coalesce()

        if isinstance(edge_index, Tensor):
            row, col, edge_weight = adj_pooled.coo()
            adj_pooled = torch.stack([row, col], dim=0)

        return adj_pooled, edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm})"
        )
