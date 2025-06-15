import torch
from torch import Tensor

from tgp.connect import Connect
from tgp.select import SelectOutput


def dense_connect(
    s: Tensor,
    adj: Tensor,
    remove_self_loops: bool = False,
    degree_norm: bool = False,
    adj_transpose: bool = False,
) -> Tensor:
    r"""Connects the nodes in the coarsened graph for dense pooling methods."""
    sta = torch.matmul(s.transpose(-2, -1), adj)
    adj_pool = torch.matmul(sta, s)
    adj_pool = postprocess_adj_pool(
        adj_pool,
        remove_self_loops=remove_self_loops,
        degree_norm=degree_norm,
        adj_transpose=adj_transpose,
    )

    return adj_pool


def postprocess_adj_pool(
    adj_pool: Tensor,
    remove_self_loops: bool = False,
    degree_norm: bool = False,
    adj_transpose: bool = False,
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
        d[d == 0] = 1  # avoid division by zero
        d = torch.sqrt(d)
        adj_pool = (adj_pool / d) / d.transpose(-2, -1)

    return adj_pool


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
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
    ):
        super().__init__()
        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm
        self.adj_transpose = adj_transpose

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
        """
        assert isinstance(so.s, Tensor), "SelectOutput.s must be a tensor"

        adj_pool = dense_connect(
            so.s,
            edge_index,
            remove_self_loops=self.remove_self_loops,
            degree_norm=self.degree_norm,
            adj_transpose=self.adj_transpose,
        )

        return adj_pool, None

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm}, "
            f"adj_transpose={self.adj_transpose})"
        )
