from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import unbatch, unbatch_edge_index

from tgp.connect import Connect
from tgp.imports import is_sparsetensor
from tgp.select import SelectOutput
from tgp.utils.ops import (
    connectivity_to_edge_index,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
    dense_to_block_diag,
    is_dense_adj,
    postprocess_adj_pool_dense,
    postprocess_adj_pool_sparse,
)


class DenseConnect(Connect):
    r"""The :math:`\texttt{connect}` operator for dense pooling methods.

    Input representations:
        - Batched dense inputs: adjacency :math:`[B, N, N]`, assignment
          :math:`[B, N, K]`.
        - Unbatched sparse inputs: sparse adjacency and dense assignment
          :math:`[N, K]` (or :math:`[1, N, K]`).

    Output representations:
        - Batched dense inputs always return a dense adjacency
          :math:`[B, K, K]` (edge weights are :obj:`None`).
        - Unbatched sparse inputs return either a dense adjacency
          :math:`[B, K, K]` or a block-diagonal sparse adjacency
          :math:`[B*K, B*K]` depending on :attr:`sparse_output`.

    It computes the pooled adjacency matrix as:

    .. math::
        \mathbf{A}_{\mathrm{pool}} =
        \mathbf{S}^{\top}\mathbf{A}\mathbf{S}

    Args:
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        adj_transpose (bool, optional):
            If :obj:`True`, it returns a transposed pooled adjacency matrix for
            batched dense outputs, so that it can be passed "as is" to the dense
            message passing layers. This only applies to batched dense inputs.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum
            absolute value per graph.
            (default: :obj:`False`)
        sparse_output (bool, optional):
            Controls the output format **only for unbatched inputs**.
            If :obj:`True`, return a block-diagonal sparse adjacency of shape
            :math:`[B*K, B*K]`. If :obj:`False`, return a dense adjacency of
            shape :math:`[B, K, K]`. Batched dense inputs always return a dense
            adjacency. (default: :obj:`False`)
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        edge_weight_norm: bool = False,
        sparse_output: bool = False,
    ):
        super().__init__()
        if not isinstance(sparse_output, bool):
            raise TypeError("sparse_output must be a bool.")
        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm
        self.adj_transpose = adj_transpose
        self.edge_weight_norm = edge_weight_norm
        self.sparse_output = sparse_output

    @staticmethod  # TODO: move to ops?
    def _prepare_batched_dense_inputs(s: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if s.dim() != 3 or adj.dim() != 3:
            raise ValueError("Expected batched dense inputs with 3 dimensions.")
        if s.size(0) != adj.size(0):
            raise ValueError(
                "Assignment and adjacency batch sizes do not match: "
                f"got s.size(0)={s.size(0)} and adj.size(0)={adj.size(0)}."
            )
        return s, adj

    @staticmethod  # TODO: move to ops?
    def _validate_select_output(so: SelectOutput) -> Tensor:
        if so is None:
            raise ValueError("SelectOutput is required for DenseConnect.")
        s = so.s
        if not isinstance(s, Tensor):
            raise TypeError("SelectOutput.s must be a torch.Tensor.")
        if s.is_sparse:
            raise ValueError("DenseConnect expects a dense assignment matrix.")
        return s

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
        r"""Postprocess the (batched) pooled adjacency matrix."""
        return postprocess_adj_pool_dense(
            adj_pool,
            remove_self_loops=remove_self_loops,
            degree_norm=degree_norm,
            adj_transpose=adj_transpose,
            edge_weight_norm=edge_weight_norm,
        )

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
        r"""Forward pass.

        Args:
            edge_index (~torch_geometric.typing.Adj):
                For batched dense inputs, a tensor of shape :math:`[B, N, N]`.
                For unbatched sparse inputs, a sparse connectivity matrix in one
                of the formats supported by :class:`~torch_geometric.typing.Adj`.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator. The assignment
                matrix :attr:`so.s` must be a **dense** tensor.
            edge_weight (~torch.Tensor, optional):
                A vector of shape :math:`[E]` or :math:`[E, 1]` containing the
                weights of the edges for unbatched inputs. (default: :obj:`None`)
            batch (~torch.Tensor, optional):
                The batch vector for unbatched inputs. (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                The pooled batch vector, required for edge weight normalization
                with unbatched sparse outputs. (default: :obj:`None`)

        Returns:
            (~torch_geometric.typing.Adj, ~torch.Tensor or None):
            The pooled adjacency matrix and the edge weights. If the pooled
            adjacency is dense, returns :obj:`None` for the edge weights.
        """
        s = self._validate_select_output(so)
        if is_dense_adj(edge_index):
            # Batched dense inputs always return a dense adjacency.
            return self._forward_batched_inputs(edge_index, s)

        return self._forward_unbatched_inputs(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            s=s,
            batch_pooled=batch_pooled,
        )

    def _forward_batched_inputs(self, adj: Tensor, s: Tensor) -> Tuple[Tensor, None]:
        """Handle batched dense inputs; always returns dense adjacency."""
        s, adj = self._prepare_batched_dense_inputs(s, adj)

        adj_pool = self.dense_connect(s, adj)

        adj_pool = self.postprocess_adj_pool(
            adj_pool,
            remove_self_loops=self.remove_self_loops,
            degree_norm=self.degree_norm,
            adj_transpose=self.adj_transpose,
            edge_weight_norm=self.edge_weight_norm,
        )

        return adj_pool, None

    def _compute_dense_pooled_adj_from_unbatched_inputs(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        s: Tensor,
        num_nodes: int,
        num_clusters: int,
        batch_size: int,
    ) -> Tensor:
        """Compute a dense pooled adjacency [B, K, K] from unbatched inputs."""
        # Single graph case
        if batch_size == 1:
            edge_index_coo = connectivity_to_torch_coo(
                edge_index, edge_weight, num_nodes=num_nodes
            )

            if edge_index_coo._nnz() == 0:  # No edges
                adj_pooled_dense = s.new_zeros((num_clusters, num_clusters))
            else:
                temp = torch.sparse.mm(edge_index_coo, s)
                adj_pooled_dense = s.transpose(-2, -1).matmul(temp)

            return adj_pooled_dense.unsqueeze(0)

        # Multi-graph case
        edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
            edge_index, edge_weight
        )
        E = edge_index_conv.size(1)
        dev = edge_index_conv.device

        # Make sure edge weights are 1D
        if edge_weight_conv is None:
            edge_weight_conv = torch.ones(E, device=dev)
        else:
            edge_weight_conv = edge_weight_conv.view(-1)

        unbatched_s = unbatch(s, batch=batch)  # list of B elements, each Ni x K

        if E == 0:  # No edges in the entire batch
            out_list = [
                unb_s.new_zeros((num_clusters, num_clusters)) for unb_s in unbatched_s
            ]
        else:
            out_list = []
            unbatched_adj = unbatch_edge_index(edge_index_conv, batch=batch)
            edge_batch = batch[edge_index_conv[0]]
            unbatched_edge_weight = unbatch(edge_weight_conv, batch=edge_batch)

            for unb_adj_i, unb_s, unb_w in zip(
                unbatched_adj, unbatched_s, unbatched_edge_weight
            ):
                N_i = unb_s.size(0)
                # Build sparse adjacency for this graph
                sp_unb_adj = torch.sparse_coo_tensor(
                    unb_adj_i, unb_w, size=(N_i, N_i)
                ).coalesce()
                # Compute S^T @ A @ S
                temp = torch.sparse.mm(sp_unb_adj, unb_s)
                out = unb_s.t().matmul(temp)
                out_list.append(out)

        return torch.stack(out_list, dim=0)  # has shape [B, K, K]

    def _forward_unbatched_inputs(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        s: Tensor,
        batch_pooled: Optional[Tensor],
    ) -> Tuple[Adj, Optional[Tensor]]:
        """Handle unbatched sparse inputs with dense assignments."""
        # Determine batch size
        batch_size = 1 if batch is None else int(batch.max().item()) + 1

        if s.dim() == 3:
            if s.size(0) != 1:
                raise ValueError(
                    "[DenseConnect - unbatched]: SelectOutput.s must have shape "
                    f"[N, K] or [1, N, K], but got {s.size()}."
                )
            s = s.squeeze(0)
        elif s.dim() != 2:
            raise ValueError(
                "[DenseConnect - unbatched]: SelectOutput.s must have shape "
                f"[N, K] or [1, N, K], but got {s.size()}."
            )

        num_nodes, num_clusters = s.size()

        # Compute pooled adjacency in dense format [B, K, K]
        adj_pool_dense = self._compute_dense_pooled_adj_from_unbatched_inputs(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            s=s,
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            batch_size=batch_size,
        )

        # Return dense adjacency [B, K, K] when sparse_output=False.
        if not self.sparse_output:
            adj_pool = self.postprocess_adj_pool(
                adj_pool_dense,
                remove_self_loops=self.remove_self_loops,
                degree_norm=self.degree_norm,
                adj_transpose=False,
                edge_weight_norm=self.edge_weight_norm,
            )
            return adj_pool, None

        if self.edge_weight_norm and batch_pooled is None:
            raise AssertionError(
                "edge_weight_norm=True but batch_pooled=None. "
                "batch_pooled parameter is required for per-graph normalization "
                "in DenseConnect."
            )

        edge_index_out, edge_weight_out = dense_to_block_diag(adj_pool_dense)
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

        # Convert back to the original sparse format when needed.
        if is_sparsetensor(edge_index):
            edge_index_out = connectivity_to_sparsetensor(
                edge_index_out, edge_weight_out, num_supernodes
            )
            edge_weight_out = None
        elif isinstance(edge_index, Tensor) and edge_index.is_sparse:
            edge_index_out = connectivity_to_torch_coo(
                edge_index_out, edge_weight_out, num_supernodes
            )
            edge_weight_out = None

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
