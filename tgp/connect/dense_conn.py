import torch
from torch import Tensor

from tgp import eps
from tgp.connect import Connect
from tgp.select import SelectOutput


class DenseConnect(Connect):
    r"""The :math:`\texttt{connect}` operator for *dense* pooling methods.

    It computes the pooled adjacency matrix as:

    .. math::
        \mathbf{A}_{\mathrm{pool}} =
        \mathbf{S}^{\top}\mathbf{A}\mathbf{S}_{\mathrm{inv}}^{\top}

    where :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}` and
    :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}` are dense tensors.

    Args:
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        adj_transpose (bool, optional):
            If :obj:`True`, it returns a transposed pooled
            adjacency matrix, so that it can be passed "as is" to the dense
            message passing layers.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        edge_weight_norm: bool = False,
    ):
        super().__init__()
        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm
        self.adj_transpose = adj_transpose
        self.edge_weight_norm = edge_weight_norm

    @staticmethod
    def dense_connect(
        s: Tensor,
        adj: Tensor,
    ) -> Tensor:
        r"""Connects the nodes in the coarsened graph for dense pooling methods."""
        sta = torch.matmul(s.transpose(-2, -1), adj)
        adj_pool = torch.matmul(sta, s)
        return adj_pool

    @staticmethod
    def postprocess_adj_pool(
        adj_pool: Tensor,
        remove_self_loops: bool = False,
        degree_norm: bool = False,
        adj_transpose: bool = False,
        edge_weight_norm: bool = False,
    ) -> Tensor:
        r"""Postprocess the adjacency matrix of the pooled graph."""
        if remove_self_loops:
            torch.diagonal(adj_pool, dim1=-2, dim2=-1)[:] = 0

        if degree_norm:
            if adj_transpose:
                # For the transposed output the "row" sum is along axis -2
                d = adj_pool.sum(-2, keepdim=True)
            else:
                # Compute row sums along the last dimension.
                d = adj_pool.sum(-1, keepdim=True)
            d = torch.sqrt(d.clamp(min=eps))
            adj_pool = (adj_pool / d) / d.transpose(-2, -1)

        if edge_weight_norm:
            # Per-graph normalization for dense adjacency matrices
            # adj_pool has shape [batch_size, num_supernodes, num_supernodes]
            batch_size = adj_pool.size(0)
            # Find max absolute value per graph: [batch_size, 1, 1]
            max_per_graph = (
                adj_pool.view(batch_size, -1).abs().max(dim=1, keepdim=True)[0]
            )
            max_per_graph = max_per_graph.unsqueeze(-1)  # [batch_size, 1, 1]

            # Avoid division by zero
            max_per_graph = torch.where(
                max_per_graph == 0, torch.ones_like(max_per_graph), max_per_graph
            )

            adj_pool = adj_pool / max_per_graph

        return adj_pool

    def forward(self, edge_index: Tensor, so: SelectOutput, **kwargs) -> Tensor:
        r"""Forward pass.

        Args:
            edge_index (~torch.Tensor):
                A tensor containing the dense adjacency matrices of the graphs
                in the batch. It has shape
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`,
                where :math:`B` is the batch size, :math:`N` is the maximum number of nodes
                for each graph in the bacth, and :math:`F` is the dimension of the node features.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.

        Returns:
            ~torch.Tensor: The pooled adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{B \times K \times K}`,
            where :math:`K` is the number of supernodes in the pooled graph.
            It also returns :obj:`None` for compatibility with the interface of other
            connect operations returning pooled edge weights as the second argument.
        """
        assert isinstance(so.s, Tensor), "SelectOutput.s must be a tensor"

        adj_pool = self.dense_connect(so.s, edge_index)

        adj_pool = self.postprocess_adj_pool(
            adj_pool,
            remove_self_loops=self.remove_self_loops,
            degree_norm=self.degree_norm,
            adj_transpose=self.adj_transpose,
            edge_weight_norm=self.edge_weight_norm,
        )

        return adj_pool, None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm}, "
            f"adj_transpose={self.adj_transpose}, "
            f"edge_weight_norm={self.edge_weight_norm})"
        )
