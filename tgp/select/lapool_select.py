from typing import Optional

import scipy.sparse.csgraph as csgraph
import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import (
    get_laplacian,
    scatter,
    to_scipy_sparse_matrix,
    unbatch,
    unbatch_edge_index,
)
from torch_scatter import scatter_mul

from tgp.select import Select, SelectOutput
from tgp.utils import (
    check_and_filter_edge_weights,
    connectivity_to_edge_index,
    is_dense_adj,
)
from tgp.utils.typing import SinvType


# TODO: should this be a static method?
def dense_cosine_similarity(x, leader_mask, batch):
    r"""Computes a dense block-diagonal cosine similarity matrix.

    This method calculates the cosine similarity between each node and the identified leaders.
    Only similarities between nodes belonging to the same graph are considered valid.

    Args:
        x (Tensor): The node feature matrix of shape :math:`[N, F]`, where :math:`N` is the number of nodes
                    and :math:`F` is the number of features.
        leader_mask (Tensor): A boolean mask of shape :math:`[N]` indicating which nodes are leaders.
        batch (Tensor, optional): A vector of shape :math:`[N]` indicating the graph each node belongs to.

    Returns:
        Tensor: A dense cosine similarity matrix of shape :math:`[N, K]`, where :math:`K` is the number of leaders.
    """
    device = x.device
    eps = 1e-8

    if batch is None:
        batch = torch.zeros(x.size(0), dtype=torch.int64, device=device)

    # Get all leader indices
    global_leader_idx = torch.nonzero(leader_mask, as_tuple=True)[0]

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
    is_dense: bool = True
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
        batched_representation (bool, optional):
            If :obj:`True`, expects batched input :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
            and a dense adjacency tensor :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            If :obj:`False`, expects unbatched input :math:`\mathbf{X} \in \mathbb{R}^{N \times F}` and a
            sparse adjacency in one of the formats supported by
            :class:`~torch_geometric.typing.Adj` (or a dense :math:`[N, N]` tensor).
            (default: :obj:`True`)
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
        self,
        shortest_path_reg: bool = False,
        batched_representation: bool = True,
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__()

        self.s_inv_op = s_inv_op
        self.shortest_path_reg = shortest_path_reg
        self.batched_representation = batched_representation

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
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
                For batched dense inputs, it also accepts dense adjacency tensors of shape
                :math:`[B, N, N]` (or :math:`[N, N]` for unbatched dense inputs).
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or :math:`[E, 1]` containing the weights of the edges.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional):
                Mask matrix :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph (batched mode only).
                (default: :obj:`None`)
            num_nodes (int, optional):
                The total number of nodes of the graphs in the batch.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        if self.batched_representation:
            if x.dim() == 2:
                x = x.unsqueeze(0)
            elif x.dim() != 3:
                raise ValueError("x must have shape [B, N, F].")

            if not is_dense_adj(edge_index):
                raise ValueError(
                    "Batched LaPoolSelect expects a dense adjacency tensor."
                )
            if edge_index.dim() == 2:
                edge_index = edge_index.unsqueeze(0)
            elif edge_index.dim() != 3:
                raise ValueError(
                    "Batched LaPoolSelect expects a dense adjacency tensor of shape "
                    "[B, N, N]."
                )

            s = self._forward_batched(x, edge_index, mask)
            return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask)

        if x.dim() != 2:
            raise ValueError("x must have shape [N, F].")
        if mask is not None:
            raise ValueError("mask is only supported for batched representations.")

        if is_dense_adj(edge_index):
            raise ValueError(
                "Unbatched LaPoolSelect expects a sparse adjacency tensor."
            )

        s = self._forward_unbatched(
            x=x,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes or x.size(0),
        )
        return SelectOutput(s=s, s_inv_op=self.s_inv_op, batch=batch)

    def _forward_batched(
        self, x: Tensor, adj: Tensor, mask: Optional[Tensor]
    ) -> Tensor:
        batch_size, num_nodes, _ = x.shape
        if mask is None:
            mask = x.new_ones(batch_size, num_nodes, dtype=torch.bool)
        else:
            if mask.dim() == 1:
                mask = mask.unsqueeze(0)
            mask = mask.to(torch.bool)

        adj = adj * mask.unsqueeze(-1) * mask.unsqueeze(-2)

        deg = adj.sum(dim=-1)
        v = deg.unsqueeze(-1) * x - torch.bmm(adj, x)
        v_norm = v.norm(dim=-1)

        neighbor_mask = adj != 0
        neighbor_vals = v_norm.unsqueeze(1).expand(-1, num_nodes, -1)
        neighbor_vals = neighbor_vals.masked_fill(~neighbor_mask, float("-inf"))
        neighbor_max = neighbor_vals.max(dim=-1).values
        leader_mask = (v_norm >= neighbor_max) & mask

        need_leader = (~leader_mask.any(dim=1)) & mask.any(dim=1)
        leader_mask = leader_mask | (need_leader.unsqueeze(1) & mask)

        x_flat = x.reshape(batch_size * num_nodes, -1)
        leader_flat = leader_mask.reshape(-1)
        batch_flat = torch.arange(batch_size, device=x.device).repeat_interleave(
            num_nodes
        )

        cosine_similarity_dense = dense_cosine_similarity(
            x_flat, leader_flat, batch_flat
        )

        beta = 1.0
        if self.shortest_path_reg:
            edge_index = adj.nonzero(as_tuple=False)
            if edge_index.numel() > 0:
                row = edge_index[:, 0] * num_nodes + edge_index[:, 1]
                col = edge_index[:, 0] * num_nodes + edge_index[:, 2]
                edge_index = torch.stack([row, col], dim=0)
                sp_matrix = to_scipy_sparse_matrix(
                    edge_index, num_nodes=batch_size * num_nodes
                ).tocsr()
                shortest_path = torch.tensor(
                    csgraph.shortest_path(sp_matrix, directed=False),
                    dtype=torch.float32,
                    device=x.device,
                )
                beta = torch.zeros_like(shortest_path, dtype=torch.float32)
                nonzero = shortest_path != 0
                beta[nonzero] = 1 / shortest_path[nonzero]
                beta = beta[:, leader_flat].to(dtype=cosine_similarity_dense.dtype)

        s = torch.softmax(cosine_similarity_dense, dim=-1)
        s = beta * s

        s_non_leader = s.clone()
        s_non_leader[leader_flat] = 0.0

        leader_idx = torch.nonzero(leader_flat).squeeze()
        if leader_idx.dim() == 0:
            leader_idx = leader_idx.unsqueeze(0)
        leader_cols = torch.arange(leader_idx.size(0), device=leader_idx.device)

        kronecker_delta = torch.zeros_like(s)
        kronecker_delta[leader_idx, leader_cols] = 1.0
        s = s_non_leader + kronecker_delta

        leaders_per_graph = scatter(
            leader_flat.float(), batch_flat, dim=0, dim_size=batch_size, reduce="sum"
        ).long()
        K_max = int(leaders_per_graph.max().item())
        cum_leaders = torch.cat(
            [
                leaders_per_graph.new_zeros(1),
                torch.cumsum(leaders_per_graph, dim=0),
            ]
        ).long()
        s_new = torch.zeros(
            batch_size * num_nodes, K_max, device=x.device, dtype=s.dtype
        )
        for b in range(batch_size):
            start = cum_leaders[b].item()
            end = cum_leaders[b + 1].item()
            k_b = end - start
            if k_b > 0:
                s_new[b * num_nodes : (b + 1) * num_nodes, :k_b] = s[
                    b * num_nodes : (b + 1) * num_nodes, start:end
                ]
        s = s_new * mask.reshape(-1, 1).to(s.dtype)
        return s.reshape(batch_size, num_nodes, K_max)

    def _forward_unbatched(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        num_nodes: int,
    ) -> Tensor:
        # Multi-graph: run single-graph path per graph, pad to K_max, concatenate
        if (
            batch is not None
            and batch.numel() > 0
            and int(batch.min().item()) != int(batch.max().item())
        ):
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )
            batch_size = int(batch.max().item()) + 1
            unbatched_x = unbatch(x, batch)
            unbatched_ei = unbatch_edge_index(edge_index, batch)
            unbatched_ew = []
            for b in range(batch_size):
                mask = batch[edge_index[0]] == b
                ew = edge_weight[mask] if edge_weight is not None else None
                unbatched_ew.append(
                    ew.squeeze(-1) if ew is not None and ew.dim() > 1 else ew
                )
            s_list = [
                self._forward_unbatched(
                    x=x_i,
                    edge_index=ei_i,
                    edge_weight=ew_i,
                    batch=None,
                    num_nodes=x_i.size(0),
                )
                for x_i, ei_i, ew_i in zip(unbatched_x, unbatched_ei, unbatched_ew)
            ]
            K_max = max(s_i.size(-1) for s_i in s_list)
            return torch.cat(
                [
                    s_i
                    if s_i.size(-1) == K_max
                    else torch.cat(
                        [
                            s_i,
                            s_i.new_zeros(s_i.size(0), K_max - s_i.size(-1)),
                        ],
                        dim=-1,
                    )
                    for s_i in s_list
                ],
                dim=0,
            )

        edge_index, edge_weight = connectivity_to_edge_index(edge_index, edge_weight)
        edge_weight = check_and_filter_edge_weights(edge_weight)

        lap_edge_index, lap_edge_weights = get_laplacian(
            edge_index, edge_weight=edge_weight, num_nodes=num_nodes
        )
        lap_sparse = torch.sparse_coo_tensor(
            lap_edge_index,
            lap_edge_weights,
            size=(num_nodes, num_nodes),
            device=x.device,
        )
        v = torch.sparse.mm(lap_sparse, x)
        v = v.norm(dim=-1, keepdim=True)

        row, col = lap_edge_index[0], lap_edge_index[1]
        if row.size(0) == 0 or (lap_edge_weights == 0).all():
            leader_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)
        else:
            leader_check = (v[row] >= v[col]).int().squeeze()
            leader_mask = scatter_mul(
                leader_check, row, dim=0, dim_size=num_nodes
            ).bool()

        if not leader_mask.any():
            leader_mask = torch.ones(num_nodes, dtype=torch.bool, device=x.device)

        cosine_similarity_dense = dense_cosine_similarity(x, leader_mask, batch)

        beta = 1.0
        if self.shortest_path_reg:
            sp_matrix = to_scipy_sparse_matrix(edge_index, num_nodes=num_nodes).tocsr()
            shortest_path = torch.tensor(
                csgraph.shortest_path(sp_matrix, directed=False),
                dtype=torch.float32,
                device=x.device,
            )
            beta = torch.zeros_like(shortest_path, dtype=torch.float32)
            nonzero = shortest_path != 0
            beta[nonzero] = 1 / shortest_path[nonzero]
            beta = beta[:, leader_mask].to(dtype=cosine_similarity_dense.dtype)

        s = torch.softmax(cosine_similarity_dense, dim=-1)
        s = beta * s

        s_non_leader = s.clone()
        s_non_leader[leader_mask] = 0.0

        leader_idx = torch.nonzero(leader_mask).squeeze()
        if leader_idx.dim() == 0:
            leader_idx = leader_idx.unsqueeze(0)
        leader_cols = torch.arange(leader_idx.size(0), device=leader_idx.device)

        kronecker_delta = torch.zeros_like(s)
        kronecker_delta[leader_idx, leader_cols] = 1.0
        s = s_non_leader + kronecker_delta

        return s

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}(s_inv_op={self.s_inv_op}, "
            f"shortest_path_reg={self.shortest_path_reg})"
        )
