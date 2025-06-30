from typing import Optional

import scipy.sparse.csgraph as csgraph
import torch
from torch import Tensor
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_scatter import scatter_add, scatter_mul
from torch_sparse import SparseTensor, spmm

from tgp.select import Select, SelectOutput
from tgp.utils import check_and_filter_edge_weights, connectivity_to_edge_index
from tgp.utils.typing import SinvType


def sparse_cosine_similarity(x, n_nodes, mask, batch):
    """Compute a block-diagonal sparse cosine similarity matrix.
    Each entry (n, k) is the cosine similarity between node n and leader k.
    """
    device = x.device
    mask_int = mask.int()

    if batch is None:
        batch = torch.zeros(n_nodes, dtype=torch.int64, device=device)

    # Calculate nodes per graph and leader counts per graph
    ones = torch.ones_like(batch, dtype=torch.int64, device=device)
    ns = scatter_add(ones, batch, dim=0)  # number of nodes per graph
    ks = scatter_add(mask_int, batch, dim=0)  # number of leaders per graph

    # Compute starting index for each graph's nodes
    starts = torch.cumsum(ns, dim=0) - ns

    # Repeat start indices and node counts for each leader in each graph
    starts_rep = torch.repeat_interleave(starts, ks)
    ns_rep = torch.repeat_interleave(ns, ks)

    # Create indices for nodes in each leader block using a vectorized ragged range
    max_ns = ns_rep.max()
    r = torch.arange(max_ns, device=device).unsqueeze(0).expand(len(ns_rep), max_ns)
    valid = r < ns_rep.unsqueeze(1)
    index_n = (starts_rep.unsqueeze(1) + r)[valid]

    # Prepare leader block indices for the sparse tensor
    total_leaders = int(ks.sum().item())
    leader_ids = torch.arange(total_leaders, device=device)
    repeats = torch.repeat_interleave(ns, ks)
    index_k_for_s = torch.repeat_interleave(leader_ids, repeats)

    # Map each leader block to its global leader index
    global_leader_idx = torch.nonzero(mask, as_tuple=True)[0]
    index_k = torch.repeat_interleave(global_leader_idx, repeats)

    # Compute cosine similarities
    x_n = x[index_n]
    x_k = x[index_k]
    eps = 1e-8
    cos_vals = (x_n * x_k).sum(dim=-1) / (x_n.norm(dim=-1) * x_k.norm(dim=-1) + eps)

    # Build and return the sparse cosine similarity matrix
    indices = torch.stack([index_n, index_k_for_s], dim=0)
    sim = torch.sparse_coo_tensor(
        indices, cos_vals, size=(n_nodes, total_leaders), device=device
    )
    return sim.coalesce()


class LaPoolSelect(Select):
    r"""The select operator for the LaPool operator (:class:`~tgp.pooler.LaPoolPooling`)
    as proposed in the paper `Towards Interpretable Sparse Graph Representation Learning
    with Laplacian Pooling <https://arxiv.org/abs/1905.11577>`_. (Emmanuel Noutahi et al., 2019).

    This operator computes a soft assignment matrix :math:`\mathbf{S}` by first identifying a set of
    leaders, and then assigning every remaining node to the cluster of the closest
    leader:

    .. math::
        \begin{align*}
            \mathbf{v} &= \| \mathbf{L} \mathbf{X} \|_d \\
            \mathbf{i} &= \{ i \mid \mathbf{v}_i > \mathbf{v}_j, \forall j \in \mathcal{N}(i) \} \\
            \mathbf{S}^\top &= \texttt{SparseSoftmax} \left( \beta \frac{\mathbf{X}\mathbf{X}_{\mathbf{i}}^\top}{\|\mathbf{X}\|\|\mathbf{X}_{\mathbf{i}}\|} \right)
        \end{align*}

    where:

    + :math:`\mathbf{L}` is the Laplacian matrix of the graph,
    + :math:`\mathbf{X}` is the input node feature matrix,
    + :math:`\beta` is a regularization vector that is applied element-wise to the selection matrix.

    Args:
        shortest_path_reg (bool, optional): If :obj:`True`, :math:`\beta` it is equal to
            the inverse of the shortest path between each node and its corresponding leader
            (this can be expensive since it runs on CPU). Otherwise :math:`\beta=1`.
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    def __init__(
        self, shortest_path_reg: bool = False, s_inv_op: SinvType = "transpose"
    ):
        super().__init__()

        self.s_inv_op = s_inv_op
        self.shortest_path_reg = shortest_path_reg

    def forward(
        self,
        x: Tensor,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor):
                The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            edge_index (~torch_geometric.typing.Adj, optional): The connectivity matrix.
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
        if isinstance(edge_index, SparseTensor):
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )
        edge_weight = check_and_filter_edge_weights(edge_weight)

        # Use dummy x if not provided (e.g., in precoarsening)
        if x is None:
            x = torch.ones((num_nodes, 1), device=edge_index.device)

        # Compute Laplacian and its associated node feature norm
        lap_edge_index, lap_edge_weights = get_laplacian(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes
        )
        v = spmm(
            index=lap_edge_index,
            value=lap_edge_weights,
            m=num_nodes,
            n=num_nodes,
            matrix=x,
        )  # v = Lx
        v = v.norm(dim=-1, keepdim=True)

        # Determine leader nodes: a node is a leader if its norm is >= that of its neighbors for all incident edges
        row, col = lap_edge_index[0], lap_edge_index[1]
        leader_check = (v[row] >= v[col]).int().squeeze()
        leader_mask = scatter_mul(leader_check, row, dim=0, dim_size=num_nodes).bool()

        # Compute sparse cosine similarity
        cosine_similarity = sparse_cosine_similarity(x, num_nodes, leader_mask, batch)

        # Shortest path regularization
        if self.shortest_path_reg:
            # Compute shortest path distances and corresponding beta regularization matrix
            sp_matrix = to_scipy_sparse_matrix(edge_index).tocsr()
            shortest_path = torch.tensor(
                csgraph.shortest_path(sp_matrix, directed=False), dtype=torch.float32
            )
            beta = torch.zeros_like(shortest_path, dtype=torch.float32)
            nonzero = shortest_path != 0
            beta[nonzero] = 1 / shortest_path[nonzero]

            # Select beta columns corresponding to leaders and match shape with cosine_similarity
            beta = (
                beta[:, leader_mask]
                .to(dtype=cosine_similarity.dtype)
                .view_as(cosine_similarity)
            )

        else:
            beta = 1.0

        s = torch.sparse.softmax(cosine_similarity, dim=-1)
        s = beta * s

        # Filter out entries corresponding to leader rows for the non-leader component
        s_coalesced = s.coalesce()
        s_indices = s_coalesced.indices()
        s_values = s_coalesced.values()
        non_leader_mask = ~leader_mask[s_indices[0]]
        filtered_indices = s_indices[:, non_leader_mask]
        filtered_values = s_values[non_leader_mask]
        s_non_leader = SparseTensor(
            row=filtered_indices[0],
            col=filtered_indices[1],
            value=filtered_values,
            sparse_sizes=s.shape,
        )

        # Construct a sparse identity (Kronecker delta) for leader nodes
        leader_idx = torch.nonzero(leader_mask).squeeze()
        leader_cols = torch.arange(leader_idx.size(0), device=leader_idx.device)
        kd_values = torch.ones(leader_cols.size(0), dtype=s.dtype, device=s.device)
        kronecker_delta_sparse = SparseTensor(
            row=leader_idx, col=leader_cols, value=kd_values, sparse_sizes=s.shape
        )

        # Combine the non-leader similarities with the leader identity
        s = (s_non_leader + kronecker_delta_sparse).coalesce()

        so = SelectOutput(s=s)

        return so

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s_inv_op={self.s_inv_op})"
