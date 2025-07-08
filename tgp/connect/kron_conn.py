import warnings
from typing import Optional, Tuple

import numpy as np
import scipy.sparse as sp
import torch
from torch import Tensor
from torch_geometric.typing import Adj, OptTensor
from torch_geometric.utils import (
    from_scipy_sparse_matrix,
    get_laplacian,
    to_scipy_sparse_matrix,
)
from torch_sparse import SparseTensor

from tgp.connect import Connect
from tgp.select import SelectOutput
from tgp.utils.ops import connectivity_to_edge_index


class KronConnect(Connect):
    r"""The :math:`\texttt{connect}` operator based on Kron reduction proposed in the paper
    `"Hierarchical Representation Learning in Graph Neural Networks with
    Node Decimation Pooling" <https://arxiv.org/abs/1910.11436>`_ (Bianchi et al., TNNLS 2020).

    Given two disjoint sets of nodes, :math:`\mathcal{V}^+` and :math:`\mathcal{V}^-`,
    a pair of nodes :math:`i` and :math:`j` are connected if :math:`i,j \in \mathcal{V}^+`,
    and there is a path in the original graph that connects :math:`i` and :math:`j`, for each
    other node :math:`k` on the path :math:`k \notin \mathcal{V}^-`

    Args:
        sparse_threshold (float, optional):
            Deletes edges whose weight is inferior to the given value.
            (default: :obj:`1e-2`)
    """

    def __init__(self, sparse_threshold: float = 1e-2):
        super().__init__()

        self.sparse_threshold = sparse_threshold

    def forward(
        self,
        edge_index: Adj,
        so: SelectOutput,
        edge_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Adj, OptTensor]:
        r"""The forward pass.

        Args:
            edge_index (~torch.Tensor):
                A tensor of shape :math:`[2, E]`, where :math:`E`
                is the number of edges in the batch.
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape
                :math:`[E]` containing the weights of the edges.
                (default: :obj:`None`)

        Returns:
            (~torch_geometric.typing.Adj, ~torch.Tensor or None): The pooled adjacency matrix and the
            edge weights. If the pooled adjacency is a :obj:`~torch_sparse.SparseTensor`,
            returns :obj:`None` as the edge weights.
        """
        # Remember the original input type to preserve output format
        edge_index_is_sparse = isinstance(edge_index, SparseTensor)
        edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)

        # Compute the Laplacian (if not given)
        if hasattr(so, "L"):
            L = so.L
        else:
            warnings.warn(
                "Laplacian not provided. The SelectOutput is not computed with NDPSelect."
            )
            assert len(so.node_index) == so.num_supernodes, (
                "Inconsistent number of clusters and node indices."
            )

            eiL, ewL = get_laplacian(edge_index, edge_weight, normalization=None)
            L = to_scipy_sparse_matrix(eiL, ewL).tocsr()

        idx_pos = so.node_index.cpu()
        all_nodes = torch.arange(so.num_nodes)
        idx_neg = all_nodes[~torch.isin(all_nodes, idx_pos)]

        # Link reconstruction with Kron reduction
        if len(idx_pos) <= 1:
            # No need to compute Kron reduction with 0 or 1 node
            Lnew = sp.csc_matrix(-np.ones((1, 1)))  # L = -1
        else:
            # Kron reduction
            L_red = L[np.ix_(idx_pos, idx_pos)]
            L_in_out = L[np.ix_(idx_pos, idx_neg)]
            L_out_in = L[np.ix_(idx_neg, idx_pos)].tocsc()
            L_comp = L[np.ix_(idx_neg, idx_neg)].tocsc()

            try:
                Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(L_comp, L_out_in))
            except RuntimeError:
                # If L_comp is exactly singular, damp the inversion with
                # Marquardt-Levenberg coefficient ml_c
                ml_c = sp.csc_matrix(sp.eye(L_comp.shape[0]) * 1e-6)
                Lnew = L_red - L_in_out.dot(sp.linalg.spsolve(ml_c + L_comp, L_out_in))

            # Make the laplacian symmetric if it is almost symmetric
            if np.abs(Lnew - Lnew.T).sum() < np.spacing(1) * np.abs(Lnew).sum():
                Lnew = (Lnew + Lnew.T) / 2.0

        # Get the pooled adjacency matrix
        A_pool = -Lnew
        if self.sparse_threshold > 0:
            A_pool = A_pool.multiply(np.abs(A_pool) > self.sparse_threshold)
        A_pool.setdiag(0)
        A_pool.eliminate_zeros()
        A_pool = A_pool.astype(np.float32)

        if edge_index_is_sparse:
            A_pool = SparseTensor.from_scipy(A_pool).to(edge_index.device)
            out = (A_pool, None)
        else:
            device = edge_index.device
            edge_index, edge_weight = from_scipy_sparse_matrix(A_pool)
            out = (edge_index.to(device), edge_weight.to(device))

        return out

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(sparse_threshold={self.sparse_threshold})"
