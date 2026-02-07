from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj, unbatch, unbatch_edge_index

from tgp.connect import DenseConnect
from tgp.select import SelectOutput
from tgp.utils.ops import (
    connectivity_to_edge_index,
    dense_to_block_diag,
    is_dense_adj,
    postprocess_adj_pool_dense,
    postprocess_adj_pool_sparse,
)


class EigenPoolConnect(DenseConnect):
    r"""The :math:`\texttt{connect}` operator for EigenPooling.

    EigenPooling replaces the standard :math:`\mathbf{S}^{\top}\mathbf{A}\mathbf{S}`
    connection with:

    .. math::
        \mathbf{A}_{\text{coar}} = \boldsymbol{\Omega}^{\top}\mathbf{A}_{\text{ext}}\boldsymbol{\Omega},

    where:

    .. math::
        \mathbf{A}_{\text{ext}} = \mathbf{A} - \mathbf{A}_{\text{int}}, \qquad
        (\mathbf{A}_{\text{int}})_{ij} =
        \begin{cases}
        \mathbf{A}_{ij} & \text{if } c_i = c_j \\
        0 & \text{otherwise}
        \end{cases}

    and :math:`\boldsymbol{\Omega}` is the hard cluster membership matrix returned
    by :class:`~tgp.select.EigenPoolSelect` (i.e., :obj:`so.s`), with
    :math:`c_i = \arg\max_k \Omega_{ik}`.

    Input representations:
        - **Batched dense inputs**: adjacency :math:`[B, N, N]`, assignment :math:`[B, N, K]`.
        - **Unbatched sparse inputs**: sparse adjacency and dense assignment :math:`[N, K]`
          (or :math:`[1, N, K]`).

    Output representations:
        - Batched dense inputs always return a dense adjacency :math:`[B, K, K]`
          (edge weights are :obj:`None`).
        - Unbatched sparse inputs return either a dense adjacency :math:`[B, K, K]`
          or a block-diagonal sparse adjacency :math:`[B*K, B*K]` depending on
          :attr:`sparse_output`.

    Args:
        remove_self_loops (bool, optional):
            Whether to remove self-loops after coarsening. (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, symmetrically normalize the pooled adjacency. (default: :obj:`True`)
        adj_transpose (bool, optional):
            If :obj:`True`, transpose the dense pooled adjacency for message passing.
            Only applies to batched dense inputs. (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize edge weights by their max absolute value per graph.
            (default: :obj:`False`)
        sparse_output (bool, optional):
            Controls the output format **only for unbatched inputs**. If :obj:`True`,
            return a block-diagonal sparse adjacency of shape :math:`[B*K, B*K]`.
            If :obj:`False`, return a dense adjacency of shape :math:`[B, K, K]`.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        edge_weight_norm: bool = False,
        sparse_output: bool = False,
    ):
        super().__init__(
            remove_self_loops=remove_self_loops,
            degree_norm=degree_norm,
            adj_transpose=adj_transpose,
            edge_weight_norm=edge_weight_norm,
            sparse_output=sparse_output,
        )

    @staticmethod
    def _compute_a_ext(
        adj: Tensor,
        cluster_index: Tensor,
    ) -> Tensor:
        """Compute the external adjacency matrix (inter-cluster edges only).

        A_ext = A - A_int where A_int contains only intra-cluster edges.

        Args:
            adj: Dense adjacency matrix [N, N].
            cluster_index: Node-to-cluster assignment [N].

        Returns:
            A_ext: Adjacency with only inter-cluster edges [N, N].
        """
        # Create mask for intra-cluster edges
        # Two nodes are in the same cluster if their cluster_index values are equal
        cluster_i = cluster_index.unsqueeze(1)  # [N, 1]
        cluster_j = cluster_index.unsqueeze(0)  # [1, N]
        same_cluster_mask = (cluster_i == cluster_j).float()  # [N, N]

        # A_int = A * same_cluster_mask (element-wise)
        # A_ext = A - A_int = A * (1 - same_cluster_mask)
        a_ext = adj * (1.0 - same_cluster_mask)

        return a_ext

    @staticmethod
    def _coarsen_dense_adj(adj: Tensor, omega: Tensor) -> Tensor:
        """Compute A_coar from a single dense adjacency and assignment.

        Args:
            adj: Dense adjacency [N, N].
            omega: One-hot assignment [N, K].

        Returns:
            A_coar: Coarsened adjacency [K, K].
        """
        # EigenPooling assumes hard cluster assignments (one-hot).
        cluster_index = omega.argmax(dim=-1)
        a_ext = EigenPoolConnect._compute_a_ext(adj, cluster_index)
        return omega.t() @ a_ext @ omega

    def forward(
        self,
        edge_index: Adj,
        so: SelectOutput,
        *,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        **kwargs,
    ) -> Tuple[Adj, Optional[Tensor]]:
        """Forward pass.

        Computes A_coar = Omega^T A_ext Omega using so.s as Omega.

        Args:
            edge_index: Graph connectivity (dense [B, N, N] or sparse).
            so: SelectOutput with standard [N, K] one-hot assignment matrix.
            edge_weight: Optional edge weights for sparse inputs.
            batch: Batch vector for unbatched inputs.
            batch_pooled: Pooled batch vector.

        Returns:
            Tuple of (coarsened adjacency, edge weights or None).
        """
        # Validate assignment matrix (must be dense).
        omega = self._validate_select_output(so)

        if is_dense_adj(edge_index):
            # Batched dense inputs: adj [B, N, N], omega [B, N, K]
            omega, adj = self._prepare_batched_dense_inputs(omega, edge_index)

            # Compute A_coar for each graph in the batch.
            adj_pool = torch.stack(
                [self._coarsen_dense_adj(adj[b], omega[b]) for b in range(adj.size(0))],
                dim=0,
            )
            adj_pool = postprocess_adj_pool_dense(
                adj_pool,
                remove_self_loops=self.remove_self_loops,
                degree_norm=self.degree_norm,
                adj_transpose=self.adj_transpose,
                edge_weight_norm=self.edge_weight_norm,
            )
            return adj_pool, None

        edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
            edge_index, edge_weight
        )
        # Unbatched sparse inputs: omega must be [N, K] (or [1, N, K]).
        if omega.dim() == 3:
            if omega.size(0) != 1:
                raise ValueError(
                    "[EigenPoolConnect - unbatched]: SelectOutput.s must have shape "
                    f"[N, K] or [1, N, K], but got {omega.size()}."
                )
            omega = omega.squeeze(0)
        elif omega.dim() != 2:
            raise ValueError(
                "[EigenPoolConnect - unbatched]: SelectOutput.s must have shape "
                f"[N, K] or [1, N, K], but got {omega.size()}."
            )

        num_nodes, num_clusters = omega.size()

        # Ensure a batch vector exists for unbatching utilities.
        if batch is None:
            batch = omega.new_zeros(num_nodes, dtype=torch.long)
        batch_size = int(batch.max().item()) + 1 if batch.numel() > 0 else 1

        # Unbatch assignments and edges per graph.
        unbatched_s = unbatch(omega, batch=batch)
        unbatched_edges = unbatch_edge_index(edge_index_conv, batch=batch)
        if edge_weight_conv is None:
            unbatched_weights = [None] * batch_size
        else:
            edge_batch = batch[edge_index_conv[0]]
            unbatched_weights = unbatch(edge_weight_conv.view(-1), batch=edge_batch)

        # Convert each graph to dense, compute A_ext, then coarsen.
        adj_pool_list = []
        for s_b, edge_index_b, edge_weight_b in zip(
            unbatched_s, unbatched_edges, unbatched_weights
        ):
            n_nodes = s_b.size(0)
            adj_b = to_dense_adj(
                edge_index_b, edge_attr=edge_weight_b, max_num_nodes=n_nodes
            ).squeeze(0)
            adj_pool_list.append(self._coarsen_dense_adj(adj_b, s_b))

        adj_pool = torch.stack(adj_pool_list, dim=0)

        # Return dense pooled adjacency for sparse_output=False.
        if not self.sparse_output:
            adj_pool = postprocess_adj_pool_dense(
                adj_pool,
                remove_self_loops=self.remove_self_loops,
                degree_norm=self.degree_norm,
                adj_transpose=False,
                edge_weight_norm=self.edge_weight_norm,
            )
            return adj_pool, None

        # Sparse output: convert [B, K, K] to block-diagonal edge_index.
        edge_index_out, edge_weight_out = dense_to_block_diag(adj_pool)
        num_supernodes = batch_size * num_clusters
        edge_index_out, edge_weight_out = postprocess_adj_pool_sparse(
            edge_index_out,
            edge_weight_out,
            num_nodes=num_supernodes,
            remove_self_loops=self.remove_self_loops,
            degree_norm=self.degree_norm,
            edge_weight_norm=self.edge_weight_norm,
            batch_pooled=batch_pooled,
        )
        return edge_index_out, edge_weight_out

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm}, "
            f"adj_transpose={self.adj_transpose}, "
            f"edge_weight_norm={self.edge_weight_norm}, "
            f"sparse_output={self.sparse_output})"
        )
