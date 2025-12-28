from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor

from tgp.connect import Connect
from tgp.select import SelectOutput
from tgp.utils import connectivity_to_edge_index, connectivity_to_sparse_tensor


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

        convert_back = False

        if isinstance(edge_index, (SparseTensor, Tensor)):
            if isinstance(edge_index, Tensor):
                convert_back = True

            BS = 1 if batch is None else batch.max().item() + 1

            if BS == 1:
                # we canno use the for computation, we compute it as before
                adj = connectivity_to_sparse_tensor(
                    edge_index, edge_weight
                ).to_torch_sparse_coo_tensor()
                sos = so.s.to_torch_sparse_coo_tensor()
                temp = torch.sparse.mm(sos.transpose(-1, -2), adj)
                adj_pooled = torch.sparse.mm(temp, sos)
                adj_pooled = SparseTensor.from_torch_sparse_coo_tensor(adj_pooled)
            else:
                # we iterate over the block diagonal structure

                # transform in edge_index representation
                edge_index, edge_weight = connectivity_to_edge_index(
                    edge_index, edge_weight
                )
                E = edge_index.size(1)
                dev = edge_index.device

                if edge_weight is None:
                    edge_weight = torch.ones(E, device=dev)

                sos = (
                    so.s.to_torch_sparse_coo_tensor()
                )  # this is sparse and has shape N x (BS*K)
                K = sos.size(1) // BS
                vals = sos._values()
                idx = sos._indices()
                idx[1] = idx[1] % K
                s = torch.sparse_coo_tensor(indices=idx, values=vals).to_dense()
                # assert s.is_coalesced()
                # s=s.to_dense()

                unbatched_s = unbatch(s, batch=batch)
                unbatched_adj = unbatch_edge_index(edge_index, batch=batch)

                out_list = []
                cont = 0
                for i in range(len(unbatched_s)):
                    unb_adj_i = unbatched_adj[i]
                    unb_s = unbatched_s[i]
                    E_i = unb_adj_i.shape[-1]
                    N_i = unb_s.shape[0]

                    vals = edge_weight[cont : cont + E_i]
                    cont += E_i

                    sp_unb_adj = torch.sparse_coo_tensor(
                        indices=unb_adj_i,
                        values=vals,
                        size=(N_i, N_i),
                    )

                    temp = sp_unb_adj @ unb_s
                    out = unb_s.t() @ temp
                    out_list.append(out)

                new_vals = torch.stack(out_list, dim=0)
                zero_mask = torch.logical_not(
                    torch.isclose(new_vals, torch.tensor(0.0, device=dev))
                ).view(-1)
                new_vals = new_vals.view(-1)[zero_mask]
                aux = (torch.arange(BS, device=dev) * K).view(-1, 1) + torch.arange(
                    K, device=dev
                )
                new_rows = (
                    aux.view(-1, K).repeat_interleave(K, dim=-1).view(-1)[zero_mask]
                )
                new_cols = aux.view(-1, K).repeat(1, K).view(-1)[zero_mask]

                adj_pooled = torch.sparse_coo_tensor(
                    values=new_vals,
                    indices=torch.stack([new_rows, new_cols], dim=0),
                ).coalesce()

                adj_pooled = SparseTensor.from_torch_sparse_coo_tensor(adj_pooled)
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
            deg[deg == 0] = 1.0  # Avoid division by zero
            deg_inv_sqrt = 1.0 / deg.sqrt()

            # Recompute values to apply D^{-1/2} * A * D^{-1/2}
            row, col, val = adj_pooled.coo()
            val = val * deg_inv_sqrt[row] * deg_inv_sqrt[col]

            adj_pooled = SparseTensor(
                row=row, col=col, value=val, sparse_sizes=adj_pooled.sparse_sizes()
            ).coalesce()

        if self.edge_weight_norm:
            row, col, val = adj_pooled.coo()
            # Use batch_pooled to map edges to graphs
            edge_batch = batch_pooled[row]

            # Find maximum absolute value per graph
            max_per_graph = scatter(val.abs(), edge_batch, dim=0, reduce="max")

            # Avoid division by zero
            max_per_graph = torch.where(
                max_per_graph == 0, torch.ones_like(max_per_graph), max_per_graph
            )

            # Normalize values by their respective graph's maximum
            val = val / max_per_graph[edge_batch]

            adj_pooled = SparseTensor(
                row=row, col=col, value=val, sparse_sizes=adj_pooled.sparse_sizes()
            ).coalesce()

        # Convert back to edge index if needed
        if convert_back:
            adj_pooled, edge_weight = connectivity_to_edge_index(
                adj_pooled, edge_weight
            )

        return adj_pooled, edge_weight

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"remove_self_loops={self.remove_self_loops}, "
            f"degree_norm={self.degree_norm}, "
            f"edge_weight_norm={self.edge_weight_norm})"
        )
