from typing import Optional, Tuple, Union

import numpy as np
import torch
from scipy.sparse import block_diag
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import (
    get_laplacian,
    is_undirected,
    to_scipy_sparse_matrix,
    to_undirected,
)
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.select import Select, SelectOutput
from tgp.utils import check_and_filter_edge_weights, connectivity_to_edge_index
from tgp.utils.typing import SinvType


class NDPSelect(Select):
    r"""The select operator for Node Decimation Pooling (:class:`~tgp.pooler.NDPPooling`),
    as presented in the paper `"Hierarchical Representation Learning in Graph Neural Networks with
    Node Decimation Pooling" <https://arxiv.org/abs/1910.11436>`_ (Bianchi et al., TNNLS 2020).

    It partitions the nodes based on the sign of the largest eigenvector of the Laplacian.
    One side of the partition becomes the set of supernodes in the pooled graph,
    while the other side is dropped.

    Args:
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    def __init__(self, s_inv_op: SinvType = "transpose"):
        super().__init__()

        self.s_inv_op = s_inv_op

    def forward(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            edge_index (~torch_geometric.typing.Adj, optional):
                The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or :math:`[E, 1]` containing the weights of the edges.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)
            num_nodes (int, optional):
                The total number of nodes of the graphs in the batch.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        if num_nodes is None:
            num_nodes = maybe_num_nodes(edge_index)

        if isinstance(edge_index, SparseTensor):
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )
        edge_weight = check_and_filter_edge_weights(edge_weight)
        device = edge_index.device

        # If no batch is provided, treat everything as one subgraph (batch=0).
        if batch is None:
            batch = torch.zeros(num_nodes, dtype=torch.long, device=device)
        batch_size = batch.max().item() + 1

        global_idx_pos = []
        global_L = []

        for b in range(batch_size):
            # Nodes of subgraph b
            sub_nodes = (batch == b).nonzero(as_tuple=False).view(-1)
            if sub_nodes.numel() == 0:
                continue  # skip empty subgraphs

            # Edges of subgraph b
            sub_mask = (batch[edge_index[0]] == b) & (batch[edge_index[1]] == b)
            sub_edge_index = edge_index[:, sub_mask]
            sub_edge_weight = edge_weight[sub_mask] if edge_weight is not None else None

            # Reindex nodes from global -> local, e.g. if sub_nodes=[12,13,14], mapping_dict={12:0, 13:1, 14:2}
            mapping_dict = {int(gn.item()): i for i, gn in enumerate(sub_nodes)}
            sub_edge_index_reindexed = torch.empty_like(sub_edge_index)
            for i in range(sub_edge_index.size(1)):
                g0 = int(sub_edge_index[0, i].item())
                g1 = int(sub_edge_index[1, i].item())
                sub_edge_index_reindexed[0, i] = mapping_dict[g0]
                sub_edge_index_reindexed[1, i] = mapping_dict[g1]

            # NDP select on subgraph b
            idx_pos_local, _, L = self._spectral_partition(
                sub_edge_index_reindexed, sub_edge_weight, sub_nodes.size(0), device
            )

            # Map local back to global
            idx_pos_global = sub_nodes[idx_pos_local]

            global_idx_pos.append(idx_pos_global)
            global_L.append(L)

        # Merge indices and L from all subgraphs
        global_idx_pos = torch.cat(global_idx_pos, dim=0)
        L = block_diag(global_L).tocsr()

        S = SparseTensor.eye(num_nodes, device=device)
        S = S[:, global_idx_pos]
        so = SelectOutput(
            s=S,
            s_inv_op=self.s_inv_op,
            L=L,
        )

        return so

    @staticmethod
    def eval_cut(total_volume, L, z):
        r"""Computes the normalized size of a cut.

        Args:
            L (~scipy.sparse.csr.csr_matrix):
                The (unweighted) Laplacian.
            z (~numpy.ndarray):
                Partition vector of shape :math:`[N, 1]` with entries in :math:`\{-1, 1\}`.

        Returns:
            ~numpy.ndarray: A value in :math:`[0,1]` representing the normalized size of the cut.
        """
        cut = z.T.matmul(L.matmul(z))  # z.T @ L @ z
        cut /= 2 * total_volume
        return cut

    @staticmethod
    def sign_partition(vec_or_size: Union[Tensor, int]) -> Tuple[Tensor, Tensor]:
        if isinstance(vec_or_size, int):
            n = vec_or_size  # it is always >= 2
            vec = torch.empty(n, dtype=torch.long)
            vec[0] = 1
            vec[1] = -1
            if n > 2:
                vec[2:] = torch.randint(0, 2, (n - 2,), dtype=torch.long) * 2 - 1
        else:  # assume it's a vector
            vec = vec_or_size
        return torch.where(vec >= 0)[0], torch.where(vec < 0)[0]

    def _spectral_partition(
        self,
        sub_edge_index: Tensor,
        sub_edge_weight: Optional[Tensor],
        num_sub_nodes: int,
        device: torch.device,
    ):
        """Build Laplacian, compute largest-eigvec partition,
        fallback to random if needed.

        Returns local idx_pos, idx_neg.
        """
        # Ensure undirected
        if not is_undirected(sub_edge_index, num_nodes=num_sub_nodes):
            sub_edge_index, sub_edge_weight = to_undirected(
                sub_edge_index, sub_edge_weight, num_nodes=num_sub_nodes, reduce="max"
            )

        # Build Laplacians
        eiL, ewL = get_laplacian(
            sub_edge_index, sub_edge_weight, normalization=None, num_nodes=num_sub_nodes
        )
        L = torch.sparse_coo_tensor(
            eiL, ewL, (num_sub_nodes, num_sub_nodes), dtype=torch.float32, device=device
        ).coalesce()

        eiLs, ewLs = get_laplacian(
            sub_edge_index,
            sub_edge_weight,
            normalization="sym",
            num_nodes=num_sub_nodes,
        )
        Ls = torch.sparse_coo_tensor(
            eiLs,
            ewLs,
            (num_sub_nodes, num_sub_nodes),
            dtype=torch.float32,
            device=device,
        ).coalesce()  # make indices unique

        if num_sub_nodes <= 1:  # Trivial case
            idx_pos_local = np.array(list(range(num_sub_nodes)), dtype=int)
            idx_neg_local = np.array([], dtype=int)
        else:
            # Try largest eigenvalue
            try:
                eigvals, eigvecs = torch.lobpcg(Ls, largest=True)
                idx_pos_local, idx_neg_local = self.sign_partition(eigvecs[:, 0])
            except Exception:  # Which exception?
                # fallback: random +/- 1
                idx_pos_local, idx_neg_local = self.sign_partition(num_sub_nodes)

            # Evaluate the size of the cut
            z = torch.ones((num_sub_nodes, 1)).to(device)
            z[idx_neg_local] = -1

            # total_volume = #edges or sum of edge_weight
            if sub_edge_weight is None:
                total_volume = sub_edge_index.size(1)
            else:
                total_volume = torch.sum(sub_edge_weight).item()

            cut_size = self.eval_cut(total_volume, L, z)

            # If the cut is too small, do random
            if cut_size < 0.5:
                idx_pos_local, idx_neg_local = self.sign_partition(num_sub_nodes)

        L = to_scipy_sparse_matrix(L.indices(), L.values(), num_nodes=num_sub_nodes)
        return idx_pos_local, idx_neg_local, L

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s_inv_op={self.s_inv_op})"
