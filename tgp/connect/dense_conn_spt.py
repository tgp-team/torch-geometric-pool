from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import unbatch, unbatch_edge_index
from torch_scatter import scatter
from torch_sparse import SparseTensor

from tgp import eps
from tgp.connect import Connect
from tgp.select import SelectOutput
from tgp.utils import connectivity_to_edge_index, connectivity_to_sparse_tensor


class DenseConnectUnbatched(Connect):
    r"""A :math:`\texttt{connect}` operator for *unbatched* dense assignment matrices.

    This operator is intended for selection matrices :math:`\mathbf{S}` stored as a
    dense tensor of shape :math:`[N, K]`, where :math:`N=\sum_i N_i` is the total
    number of nodes across a mini-batch of graphs and :math:`K` is the number of
    clusters per graph. The `batch` vector is used to split :math:`\mathbf{S}` and
    the input adjacency into per-graph blocks, and the pooled adjacency is returned
    as a block-diagonal matrix of shape :math:`[B \cdot K, B \cdot K]`.

    The pooled adjacency matrix is computed as:

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

        convert_back = False

        if isinstance(edge_index, (SparseTensor, Tensor)):
            if isinstance(edge_index, Tensor):
                convert_back = True
            if edge_weight is not None and edge_weight.dim() > 1:
                edge_weight = edge_weight.view(-1)

            BS = 1 if batch is None else batch.max().item() + 1

            if BS == 1:  # There is only one graph, no need to iterate
                if not isinstance(so.s, Tensor):
                    raise TypeError(
                        "DenseConnectUnbatched expects a dense Tensor assignment matrix."
                    )
                s = so.s
                if s.dim() == 3:
                    if s.size(0) != 1:
                        raise ValueError(
                            "DenseConnectUnbatched expects a 2D assignment matrix for a single graph."
                        )
                    s = s.squeeze(0)
                num_nodes = s.size(0)
                adj = connectivity_to_sparse_tensor(
                    edge_index, edge_weight, num_nodes=num_nodes
                )
                if adj.sparse_sizes()[0] != num_nodes:
                    adj = adj.sparse_resize((num_nodes, num_nodes))
                temp = adj.matmul(s)
                adj_pooled_dense = s.transpose(-1, -2).matmul(temp)
                adj_pooled = SparseTensor.from_dense(adj_pooled_dense)
            else:  # we iterate over the block diagonal structure
                # transform in edge_index representation
                edge_index, edge_weight = connectivity_to_edge_index(
                    edge_index, edge_weight
                )
                E = edge_index.size(1)
                dev = edge_index.device

                if edge_weight is None:
                    edge_weight = torch.ones(E, device=dev)

                if not isinstance(so.s, Tensor):
                    raise TypeError(
                        "DenseConnectUnbatched expects a dense Tensor assignment matrix."
                    )
                s = so.s
                if s.dim() != 2:
                    raise ValueError(
                        "DenseConnectUnbatched expects a 2D assignment matrix for batched graphs."
                    )
                K = s.size(1)

                unbatched_s = unbatch(
                    s, batch=batch
                )  # list of B elements, each of shape Ni x K

                out_list = []
                if E == 0:
                    for unb_s in unbatched_s:
                        out_list.append(unb_s.new_zeros((K, K)))
                else:
                    unbatched_adj = unbatch_edge_index(
                        edge_index, batch=batch
                    )  # list of B elements, each of shape 2 x Ei
                    edge_batch = batch[edge_index[0]]
                    unbatched_edge_weight = unbatch(edge_weight, batch=edge_batch)

                    for unb_adj_i, unb_s, unb_w in zip(
                        unbatched_adj, unbatched_s, unbatched_edge_weight
                    ):
                        N_i = unb_s.size(0)

                        sp_unb_adj = SparseTensor.from_edge_index(
                            unb_adj_i, unb_w, sparse_sizes=(N_i, N_i)
                        )
                        temp = sp_unb_adj.matmul(unb_s)
                        out = unb_s.t().matmul(temp)
                        out_list.append(out)

                # Build a sparse B*K x B*K block diagonal matrix from a list of B dense K x K matrices
                new_vals = torch.stack(out_list, dim=0)
                zero_mask = torch.logical_not(
                    torch.isclose(new_vals, new_vals.new_zeros(()))
                ).view(
                    -1
                )  # Filter small values to avoid storing zeros in sparse tensor
                new_vals = new_vals.view(-1)[zero_mask]
                aux = (torch.arange(BS, device=dev) * K).view(-1, 1) + torch.arange(
                    K, device=dev
                )
                new_rows = (
                    aux.view(-1, K).repeat_interleave(K, dim=-1).view(-1)[zero_mask]
                )
                new_cols = aux.view(-1, K).repeat(1, K).view(-1)[zero_mask]

                adj_pooled = SparseTensor(
                    row=new_rows,
                    col=new_cols,
                    value=new_vals,
                    sparse_sizes=(BS * K, BS * K),
                ).coalesce()
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

        # Drop near-zero edges to avoid tiny degrees and keep the graph sparse.
        row, col, val = adj_pooled.coo()
        mask = val.abs() > eps
        if not torch.all(mask):
            adj_pooled = SparseTensor(
                row=row[mask],
                col=col[mask],
                value=val[mask],
                sparse_sizes=adj_pooled.sparse_sizes(),
            ).coalesce()

        if self.degree_norm:
            deg = adj_pooled.sum(dim=1)
            deg[deg == 0] = eps  # Avoid division by zero

            # Recompute values to apply D^{-1/2} * A * D^{-1/2}
            row, col, val = adj_pooled.coo()
            deg_inv_sqrt = deg.pow(-0.5)
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
