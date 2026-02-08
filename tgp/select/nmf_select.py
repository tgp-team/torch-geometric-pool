from typing import Optional

import torch
from sklearn.decomposition import non_negative_factorization
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj

from tgp.select import Select, SelectOutput
from tgp.utils.ops import connectivity_to_edge_index, is_dense_adj
from tgp.utils.typing import SinvType


class NMFSelect(Select):
    r"""Select operator that performs Non-negative Matrix Factorization
    pooling as proposed in the paper `"A Non-Negative Factorization approach
    to node pooling in Graph Convolutional Neural Networks"
    <https://arxiv.org/abs/1909.03287>`_ (Bacciu and Di Sotto, AIIA 2019).

    This select operator computes the non-negative matrix factorization

    .. math::
        \mathbf{A} = \mathbf{W}\mathbf{H}

    and returns :math:`\mathbf{H}^\top` as the dense clustering matrix.

    Args:
        k (int):
            Number of clusters or supernodes in the pooler graph.
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    is_dense: bool = True

    def __init__(self, k: int, s_inv_op: SinvType = "transpose"):
        super().__init__()

        self.k = k
        self.s_inv_op = s_inv_op

    def _factorize_single_adjacency(
        self,
        adj: Tensor,
    ) -> Tensor:
        r"""Factorize a single dense adjacency and return a dense assignment.

        The returned assignment has shape :math:`[N, \tilde{K}]`, where
        :math:`\tilde{K} = \min(K, N)` (with :math:`K=\texttt{self.k}`).
        """
        num_nodes = adj.size(0)

        if num_nodes == 0:
            return adj.new_zeros((0, 0))

        actual_k = max(1, min(self.k, num_nodes))

        # When k >= N on non-trivial graphs, use the trivial one-node-per-cluster assignment.
        if num_nodes > 1 and actual_k >= num_nodes:
            s = torch.eye(num_nodes, device=adj.device, dtype=adj.dtype)
        elif actual_k == 1:
            s = torch.ones((num_nodes, 1), device=adj.device, dtype=adj.dtype)
        else:
            adj_np = adj.clamp(min=0).cpu().numpy()  # NMF requires non-negative input.
            _, h, _ = non_negative_factorization(
                adj_np,
                n_components=actual_k,
                init="random",
                max_iter=5000,
            )
            s = torch.tensor(h.T, device=adj.device, dtype=adj.dtype)
            s = torch.softmax(s, dim=-1)

        return s

    @staticmethod
    def _pad_assignment(s: Tensor, k: int) -> Tensor:
        """Right-pad assignment columns with zeros to obtain shape :math:`[N, K]`.

        This is needed whenever `k` is fixed globally but some graphs have fewer nodes than `k`.
        """
        if s.size(-1) >= k:
            return s
        pad = s.new_zeros((s.size(0), k - s.size(-1)))
        return torch.cat([s, pad], dim=-1)

    @staticmethod
    def _to_dense_single_sparse_graph(
        edge_index: Tensor,
        edge_weight: Optional[Tensor],
        batch: Optional[Tensor],
        num_nodes: Optional[int],
    ) -> Tensor:
        """Convert a sparse single-graph input to a dense adjacency."""
        # Determine max_num_nodes for to_dense_adj. If num_nodes is provided, use it. Otherwise, infer from edge_index or batch.
        if batch is None or batch.numel() == 0:
            max_num_nodes = num_nodes
            if max_num_nodes is None:
                max_num_nodes = (
                    int(edge_index.max().item()) + 1 if edge_index.numel() > 0 else 0
                )
        else:
            max_num_nodes = batch.size(0)
            if num_nodes is not None:
                max_num_nodes = max(num_nodes, max_num_nodes)

        return to_dense_adj(
            edge_index,
            edge_attr=edge_weight,
            max_num_nodes=max_num_nodes,
        ).squeeze(0)

    def forward(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        mask: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        fixed_k: bool = False,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass of the select operator.

        Args:
            edge_index (~torch_geometric.typing.Adj): Graph connectivity.
                It can be a dense adjacency tensor :math:`[B, N, N]` (or :math:`[N, N]`)
                or a sparse representation (:obj:`edge_index`, SparseTensor, torch COO).
            edge_weight (~torch.Tensor, optional): Edge weights for sparse inputs.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph for dense inputs. (default: :obj:`None`)
            batch (~torch.Tensor, optional): Batch vector for sparse inputs.
                (default: :obj:`None`)
            num_nodes (int, optional): Number of nodes for sparse inputs when it
                cannot be inferred from :obj:`edge_index`. (default: :obj:`None`)
            fixed_k (bool, optional): If :obj:`True`, force assignment width to
                exactly :obj:`k` for single sparse graphs by right-padding zero
                columns. Useful for pre-coarsening where per-sample outputs are
                collated together. (default: :obj:`False`)

        Returns:
           :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        # Dense adjacency path: returns [B, N, K] with fixed K.
        if is_dense_adj(edge_index):
            adj = edge_index
            if adj.dim() == 2:
                adj = adj.unsqueeze(0)
            if adj.dim() != 3:
                raise ValueError(
                    f"Expected dense adjacency with shape [B, N, N], got {adj.size()}."
                )

            B, N, _ = adj.size()
            if mask is None:
                mask = torch.ones((B, N), dtype=torch.bool, device=adj.device)

            s_list = []
            for b in range(B):
                s_b = adj.new_zeros((N, self.k))
                valid_idx = torch.nonzero(mask[b], as_tuple=False).view(-1)
                n_nodes = int(valid_idx.numel())
                if n_nodes > 0:
                    adj_b = adj[b][valid_idx][:, valid_idx]
                    s_valid = self._pad_assignment(
                        self._factorize_single_adjacency(adj_b),
                        self.k,
                    )
                    s_b[valid_idx] = s_valid
                s_list.append(s_b)

            s = torch.stack(s_list, dim=0)
            return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask)

        # Sparse connectivity path.
        edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
            edge_index, edge_weight
        )
        device = edge_index_conv.device

        is_single_graph = (
            batch is None
            or batch.numel() == 0
            or int(batch.min().item()) == int(batch.max().item())
        )

        # Single sparse graph: return [N, actual_k].
        if is_single_graph:
            adj_dense = self._to_dense_single_sparse_graph(
                edge_index=edge_index_conv,
                edge_weight=edge_weight_conv,
                batch=batch,
                num_nodes=num_nodes,
            )
            s = self._factorize_single_adjacency(adj_dense)
            if fixed_k:
                s = self._pad_assignment(s, self.k)
            return SelectOutput(s=s, s_inv_op=self.s_inv_op, batch=batch)

        # Multi-graph sparse batch: factorize each graph independently and return [N, K].
        batch_size = int(batch.max().item()) + 1
        num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
        node_ptr = torch.cat(
            [num_nodes_per_graph.new_zeros(1), num_nodes_per_graph.cumsum(0)], dim=0
        )
        if edge_index_conv.numel() == 0:
            edge_batch = batch.new_empty((0,), dtype=torch.long)
        else:
            edge_batch = batch[edge_index_conv[0]]

        if edge_weight_conv is None:
            dtype = torch.get_default_dtype()
        else:
            dtype = edge_weight_conv.dtype

        s_list = []
        for i in range(batch_size):
            n_nodes = int(num_nodes_per_graph[i].item())
            if n_nodes == 0:
                s_list.append(torch.zeros((0, self.k), dtype=dtype, device=device))
                continue

            edge_mask = edge_batch == i
            edge_index_i = edge_index_conv[:, edge_mask]
            if edge_weight_conv is None:
                edge_weight_i = None
            else:
                edge_weight_i = edge_weight_conv[edge_mask]

            if edge_index_i.numel() == 0:
                adj_dense = torch.zeros((n_nodes, n_nodes), dtype=dtype, device=device)
            else:
                node_start = int(node_ptr[i].item())
                edge_index_i = edge_index_i - node_start
                adj_dense = to_dense_adj(
                    edge_index_i,
                    edge_attr=edge_weight_i,
                    max_num_nodes=n_nodes,
                ).squeeze(0)
            s_i = self._pad_assignment(
                self._factorize_single_adjacency(adj_dense),
                self.k,
            )
            s_list.append(s_i)

        s = (
            torch.cat(s_list, dim=0)
            if s_list
            else torch.zeros((0, self.k), dtype=dtype, device=device)
        )
        return SelectOutput(s=s, s_inv_op=self.s_inv_op, batch=batch)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, s_inv_op={self.s_inv_op})"
