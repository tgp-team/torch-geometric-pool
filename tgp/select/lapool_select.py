from typing import Optional

import scipy.sparse.csgraph as csgraph
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import get_laplacian, to_scipy_sparse_matrix
from torch_scatter import scatter_mul
from torch_sparse import SparseTensor, spmm

from tgp.select import Select, SelectOutput
from tgp.utils import check_and_filter_edge_weights, connectivity_to_edge_index
from tgp.utils.typing import SinvType


def dense_cosine_similarity(x, n_nodes, mask, batch):
    r"""Computes a dense block-diagonal cosine similarity matrix.

    This method calculates the cosine similarity between each node and the identified leaders.
    Only similarities between nodes belonging to the same graph are considered valid.

    Args:
        x (Tensor): The node feature matrix of shape :math:`[N, F]`, where :math:`N` is the number of nodes
                    and :math:`F` is the number of features.
        n_nodes (int): The total number of nodes in the batch.
        mask (Tensor): A boolean mask of shape :math:`[N]` indicating which nodes are leaders.
        batch (Tensor, optional): A vector of shape :math:`[N]` indicating the graph each node belongs to.

    Returns:
        Tensor: A dense cosine similarity matrix of shape :math:`[N, K]`, where :math:`K` is the number of leaders.
    """
    device = x.device
    eps = 1e-8

    if batch is None:
        batch = torch.zeros(n_nodes, dtype=torch.int64, device=device)

    # Get all leader indices
    global_leader_idx = torch.nonzero(mask, as_tuple=True)[0]

    # Get leader features
    x_leaders = x[global_leader_idx]  # [K, F]

    # Compute cosine similarity between all nodes and all leaders
    # Use broadcasting: x is [N, F], x_leaders is [K, F]
    # We want to compute similarity for each (n, k) pair
    dot_product = x @ x_leaders.t()  # [N, K]
    node_norms = x.norm(dim=-1, keepdim=True)  # [N, 1]
    leader_norms = x_leaders.norm(dim=-1, keepdim=True).t()  # [1, K]

    # Compute cosine similarity
    cosine_sim = dot_product / (node_norms * leader_norms + eps)  # [N, K]

    # Zero out similarities between nodes and leaders from different graphs
    # Create a mask where cosine_sim[i, j] is valid only if batch[i] == batch[global_leader_idx[j]]
    batch_nodes = batch.unsqueeze(1)  # [N, 1]
    batch_leaders = batch[global_leader_idx].unsqueeze(0)  # [1, K]
    same_graph_mask = batch_nodes == batch_leaders  # [N, K]

    # Ensure nodes are only assigned to leaders within the same graph.
    # Using -inf guarantees that softmax produces exactly zero probability mass
    # for leaders belonging to different graphs.
    cosine_sim = cosine_sim.masked_fill(~same_graph_mask, float("-inf"))

    return cosine_sim


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
        edge_index: Adj,
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

        # Check if we have meaningful edges (non-zero weights)
        if row.size(0) == 0 or (lap_edge_weights == 0).all():
            # No meaningful edges: all nodes are leaders since they have no neighbors to compare against
            leader_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        else:
            leader_check = (v[row] >= v[col]).int().squeeze()
            leader_mask = scatter_mul(
                leader_check, row, dim=0, dim_size=num_nodes
            ).bool()

        # Compute dense cosine similarity directly
        cosine_similarity_dense = dense_cosine_similarity(
            x, num_nodes, leader_mask, batch
        )

        # Shortest path regularization
        if self.shortest_path_reg:
            # Compute shortest path distances and corresponding beta regularization matrix
            sp_matrix = to_scipy_sparse_matrix(edge_index).tocsr()
            shortest_path = torch.tensor(
                csgraph.shortest_path(sp_matrix, directed=False),
                dtype=torch.float32,
                device=x.device,
            )
            beta = torch.zeros_like(shortest_path, dtype=torch.float32)
            nonzero = shortest_path != 0
            beta[nonzero] = 1 / shortest_path[nonzero]

            # Select beta columns corresponding to leaders
            beta = beta[:, leader_mask].to(dtype=cosine_similarity_dense.dtype)
        else:
            beta = 1.0

        # Apply softmax and beta regularization
        # Note: softmax is applied to all nodes (including leaders) before filtering
        s = torch.softmax(cosine_similarity_dense, dim=-1)
        s = beta * s

        # Filter out entries corresponding to leader rows for the non-leader component
        s_non_leader = s.clone()
        s_non_leader[leader_mask] = 0.0  # Zero out leader rows

        # Construct a dense identity (Kronecker delta) for leader nodes
        leader_idx = torch.nonzero(leader_mask).squeeze()
        if leader_idx.dim() == 0:  # edge case where there is only one leader
            leader_idx = leader_idx.unsqueeze(0)
        leader_cols = torch.arange(leader_idx.size(0), device=leader_idx.device)

        # Create dense identity matrix for leaders
        kronecker_delta = torch.zeros_like(s)
        kronecker_delta[leader_idx, leader_cols] = 1.0

        # Combine the non-leader similarities with the leader identity
        s = s_non_leader + kronecker_delta

        # Return dense [N, K] tensor for efficiency
        so = SelectOutput(s=s, s_inv_op=self.s_inv_op, batch=batch)

        return so

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s_inv_op={self.s_inv_op})"
