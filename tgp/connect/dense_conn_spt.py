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
)


class DenseConnectUnbatched(Connect):
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
        batch: Optional[Tensor] = None,
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
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
                Expects :attr:`so.s` to be a dense tensor of shape :math:`[N, K]`
                (or :math:`[1, N, K]` for a single graph).
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
                "batch_pooled parameter is required for per-graph normalization in DenseConnectUnbatched."
            )

        # Determine batch size
        BS = 1 if batch is None else batch.max().item() + 1

        # Handle input format conversion
        to_sparsetensor = False
        to_torch_coo = False
        if is_sparsetensor(edge_index):
            to_sparsetensor = True
        elif isinstance(edge_index, Tensor) and edge_index.is_sparse:
            to_torch_coo = True

        # Get assignment matrix s
        if not isinstance(so.s, Tensor):
            raise TypeError(
                "DenseConnectUnbatched expects a dense Tensor assignment matrix."
            )
        s = so.s
        if s.dim() == 3:
            if s.size(0) != 1:
                raise ValueError(
                    "DenseConnectUnbatched expects a 2D assignment matrix for multi-graph batches."
                )
            s = s.squeeze(0)

        num_nodes = s.size(0)
        K = s.size(1)

        if BS == 1:  # Single graph - use efficient torch.sparse.mm
            edge_index_coo = connectivity_to_torch_coo(
                edge_index, edge_weight, num_nodes=num_nodes
            )

            # Handle edge case: empty adjacency matrix
            if edge_index_coo._nnz() == 0:
                num_supernodes = K
                edge_index_out = torch.empty((2, 0), dtype=torch.long, device=s.device)
                edge_weight_out = torch.empty(0, dtype=s.dtype, device=s.device)
            elif s.is_sparse:
                # Sparse assignment matrix - use original sparse matmul approach
                s_t = s.transpose(-2, -1)
                temp = torch.sparse.mm(s_t, edge_index_coo)
                adj_pooled = torch.sparse.mm(temp, s)
                edge_index_out = adj_pooled._indices()
                edge_weight_out = adj_pooled._values()
            else:
                # Dense assignment matrix - compute result as dense, then sparsify
                # torch.sparse.mm(sparse, dense) -> dense
                temp = torch.sparse.mm(edge_index_coo, s)  # [N, N] @ [N, K] -> [N, K]
                adj_pooled_dense = s.transpose(-2, -1).matmul(
                    temp
                )  # [K, N] @ [N, K] -> [K, K]

                # Convert dense [K, K] to sparse representation
                # Filter out near-zero values to keep it sparse
                mask = adj_pooled_dense.abs() > eps
                if mask.any():
                    indices = mask.nonzero(as_tuple=False).t()  # [2, nnz]
                    values = adj_pooled_dense[mask]
                    adj_pooled = torch.sparse_coo_tensor(
                        indices, values, size=(K, K)
                    ).coalesce()
                    edge_index_out = adj_pooled._indices()
                    edge_weight_out = adj_pooled._values()
                else:
                    edge_index_out = torch.empty(
                        (2, 0), dtype=torch.long, device=s.device
                    )
                    edge_weight_out = torch.empty(0, dtype=s.dtype, device=s.device)

            num_supernodes = K
        else:  # Multiple graphs - iterate over batch
            # Multi-graph batch requires dense assignment matrix
            if s.is_sparse:
                raise ValueError(
                    "DenseConnectUnbatched requires a dense assignment matrix for "
                    "multi-graph batches. Use batch=None for sparse assignment matrices."
                )

            # Convert to edge_index representation
            edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
                edge_index, edge_weight
            )
            E = edge_index_conv.size(1)
            dev = edge_index_conv.device

            if edge_weight_conv is None:
                edge_weight_conv = torch.ones(E, device=dev)
            elif edge_weight_conv.dim() > 1:
                edge_weight_conv = edge_weight_conv.view(-1)

            unbatched_s = unbatch(s, batch=batch)  # list of B elements, each Ni x K

            out_list = []
            if E == 0:
                for unb_s in unbatched_s:
                    out_list.append(unb_s.new_zeros((K, K)))
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

            # Build a sparse B*K x B*K block diagonal matrix from list of dense K x K
            new_vals = torch.stack(out_list, dim=0)
            zero_mask = torch.logical_not(
                torch.isclose(new_vals, new_vals.new_zeros(()))
            ).view(-1)  # Filter small values
            new_vals = new_vals.view(-1)[zero_mask]
            aux = (torch.arange(BS, device=dev) * K).view(-1, 1) + torch.arange(
                K, device=dev
            )
            new_rows = aux.view(-1, K).repeat_interleave(K, dim=-1).view(-1)[zero_mask]
            new_cols = aux.view(-1, K).repeat(1, K).view(-1)[zero_mask]

            num_supernodes = BS * K
            adj_pooled = torch.sparse_coo_tensor(
                torch.stack([new_rows, new_cols], dim=0),
                new_vals,
                size=(num_supernodes, num_supernodes),
            ).coalesce()

            edge_index_out = adj_pooled._indices()
            edge_weight_out = adj_pooled._values()

        if self.remove_self_loops:
            edge_index_out, edge_weight_out = rsl(edge_index_out, edge_weight_out)

        # Drop near-zero edges to keep graph sparse
        if edge_weight_out is not None and edge_weight_out.numel() > 0:
            mask = edge_weight_out.abs() > eps
            if not torch.all(mask):
                edge_index_out = edge_index_out[:, mask]
                edge_weight_out = edge_weight_out[mask]

        if self.degree_norm:
            if edge_weight_out is None or edge_weight_out.numel() == 0:
                edge_weight_out = torch.ones(
                    edge_index_out.size(1), device=edge_index_out.device
                )

            # Compute degree normalization D^{-1/2} A D^{-1/2}
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
            f"edge_weight_norm={self.edge_weight_norm})"
        )
