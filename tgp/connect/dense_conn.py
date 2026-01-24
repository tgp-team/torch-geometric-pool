from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import remove_self_loops as rsl
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_scatter import scatter

from tgp import eps
from tgp.connect import Connect
from tgp.imports import is_sparsetensor
from tgp.select import SelectOutput
from tgp.utils.ops import (
    connectivity_to_edge_index,
    connectivity_to_sparsetensor,
    connectivity_to_torch_coo,
    dense_to_block_diag,
    is_dense_adj,
)


class DenseConnect(Connect):
    r"""The :math:`\texttt{connect}` operator for dense pooling methods.

    It supports both batched dense representations and unbatched sparse ones.
    For dense inputs, it expects adjacency matrices of shape
    :math:`[B, N, N]` and assignment matrices of shape :math:`[B, N, K]`.
    For unbatched inputs, it expects a connectivity matrix in sparse form and
    a dense assignment matrix of shape :math:`[N, K]` (or :math:`[1, N, K]`).

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
            message passing layers.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum
            absolute value per graph.
            (default: :obj:`False`)
        unbatched_output (str, optional):
            Output format for unbatched inputs. Use :obj:`"block"` to return a
            block-diagonal sparse adjacency of shape :math:`[B*K, B*K]`, or
            :obj:`"batch"` to return a dense adjacency of shape :math:`[B, K, K]`.
            (default: :obj:`"block"`)
    """

    def __init__(
        self,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        edge_weight_norm: bool = False,
        unbatched_output: str = "block",
    ):
        super().__init__()
        if unbatched_output not in {"block", "batch"}:
            raise ValueError(
                "unbatched_output must be one of {'block', 'batch'}, "
                f"got {unbatched_output}."
            )
        self.remove_self_loops = remove_self_loops
        self.degree_norm = degree_norm
        self.adj_transpose = adj_transpose
        self.edge_weight_norm = edge_weight_norm
        self.unbatched_output = unbatched_output

    @staticmethod
    def _prepare_dense_inputs(s: Tensor, adj: Tensor) -> Tuple[Tensor, Tensor]:
        if s.dim() == 2:
            s = s.unsqueeze(0)
        if adj.dim() == 2:
            adj = adj.unsqueeze(0)
        if s.dim() != 3 or adj.dim() != 3:
            raise ValueError("Expected dense inputs with 3 dimensions.")
        if s.size(0) != adj.size(0):
            raise ValueError(
                "Assignment and adjacency batch sizes do not match: "
                f"got s.size(0)={s.size(0)} and adj.size(0)={adj.size(0)}."
            )
        return s, adj

    @staticmethod
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
        r"""Postprocess the adjacency matrix of the pooled graph."""
        if remove_self_loops:
            torch.diagonal(adj_pool, dim1=-2, dim2=-1)[:] = 0

        # Apply degree normalization D^{-1/2} A D^{-1/2}
        if degree_norm:
            if adj_transpose:
                # For the transposed output the "row" sum is along axis -2
                d = adj_pool.sum(-2, keepdim=True)
            else:
                # Compute row sums along the last dimension.
                d = adj_pool.sum(-1, keepdim=True)
            d = torch.sqrt(d.clamp(min=eps))
            adj_pool = (adj_pool / d) / d.transpose(-2, -1)

        # Normalize edge weights by maximum absolute value per graph
        if edge_weight_norm:
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
                For batched dense pooling, a tensor of shape :math:`[B, N, N]`.
                For unbatched pooling, a sparse connectivity matrix in one of the
                formats supported by :class:`~torch_geometric.typing.Adj`.
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
            return self._forward_dense(edge_index, s)

        return self._forward_unbatched(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            s=s,
            batch_pooled=batch_pooled,
        )

    def _forward_dense(self, adj: Tensor, s: Tensor) -> Tuple[Tensor, None]:
        s, adj = self._prepare_dense_inputs(s, adj)

        adj_pool = self.dense_connect(s, adj)

        adj_pool = self.postprocess_adj_pool(
            adj_pool,
            remove_self_loops=self.remove_self_loops,
            degree_norm=self.degree_norm,
            adj_transpose=self.adj_transpose,
            edge_weight_norm=self.edge_weight_norm,
        )

        return adj_pool, None

    def _dense_unbatched_output(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        s: Tensor,
        num_nodes: int,
        num_clusters: int,
        batch_size: int,
    ) -> Tensor:
        """Compute a dense pooled adjacency tensor of shape [B, K, K]."""
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
        elif edge_weight_conv.dim() > 1:
            edge_weight_conv = edge_weight_conv.view(-1)

        unbatched_s = unbatch(s, batch=batch)  # list of B elements, each Ni x K

        out_list = []
        if E == 0:  # No edges in the entire batch
            for unb_s in unbatched_s:
                out_list.append(unb_s.new_zeros((num_clusters, num_clusters)))
        else:
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

    def _forward_unbatched(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        s: Tensor,
        batch_pooled: Optional[Tensor],
    ) -> Tuple[Adj, Optional[Tensor]]:
        """Connect unbatched inputs with dense assignments."""
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

        num_nodes = s.size(0)
        num_clusters = s.size(1)

        # Compute pooled adjacency in dense format [B, K, K]
        adj_pool_dense = self._dense_unbatched_output(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            s=s,
            num_nodes=num_nodes,
            num_clusters=num_clusters,
            batch_size=batch_size,
        )

        # Return dense adjacency [B, K, K]
        if self.unbatched_output == "batch":
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

        # Handle input format conversion
        to_sparsetensor = False
        to_torch_coo = False
        if is_sparsetensor(edge_index):
            to_sparsetensor = True
        elif isinstance(edge_index, Tensor) and edge_index.is_sparse:
            to_torch_coo = True

        edge_index_out, edge_weight_out = dense_to_block_diag(adj_pool_dense)
        num_supernodes = batch_size * num_clusters

        if self.remove_self_loops:
            edge_index_out, edge_weight_out = rsl(edge_index_out, edge_weight_out)

        # Drop near-zero edges to keep graph sparse
        if edge_weight_out is not None and edge_weight_out.numel() > 0:
            mask = edge_weight_out.abs() > eps
            if not torch.all(mask):
                edge_index_out = edge_index_out[:, mask]
                edge_weight_out = edge_weight_out[mask]

        # Apply degree normalization D^{-1/2} A D^{-1/2}
        if self.degree_norm:
            if edge_weight_out is None or edge_weight_out.numel() == 0:
                edge_weight_out = torch.ones(
                    edge_index_out.size(1), device=edge_index_out.device
                )

            # Compute degree
            deg = scatter(
                edge_weight_out,
                edge_index_out[0],
                dim=0,
                dim_size=num_supernodes,
                reduce="sum",
            )
            deg[deg == 0] = eps  # Avoid division by zero
            deg_inv_sqrt = deg.pow(-0.5)

            # Apply symmetric normalization to edge weights
            edge_weight_out = (
                edge_weight_out
                * deg_inv_sqrt[edge_index_out[0]]
                * deg_inv_sqrt[edge_index_out[1]]
            )

        # Normalize edge weights by maximum absolute value per graph
        if self.edge_weight_norm and edge_weight_out is not None:
            # Per-graph normalization using batch_pooled
            edge_batch = batch_pooled[edge_index_out[0]]

            # Find maximum absolute edge weight per graph
            max_per_graph = scatter(
                edge_weight_out.abs(), edge_batch, dim=0, reduce="max"
            )

            # Avoid division by zero
            max_per_graph = torch.where(
                max_per_graph == 0, torch.ones_like(max_per_graph), max_per_graph
            )

            # Normalize edge weights by their respective graph's maximum
            edge_weight_out = edge_weight_out / max_per_graph[edge_batch]

        # Convert back to the original format
        if to_sparsetensor:
            edge_index_out = connectivity_to_sparsetensor(
                edge_index_out, edge_weight_out, num_supernodes
            )
            edge_weight_out = None
        elif to_torch_coo:
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
            f'unbatched_output="{self.unbatched_output}")'
        )
