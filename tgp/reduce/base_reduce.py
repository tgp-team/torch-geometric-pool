from typing import Optional

import torch
import torch_sparse
from torch import Tensor, nn
from torch_geometric.utils import scatter, unbatch
from torch_sparse import SparseTensor

from tgp.select import SelectOutput
from tgp.utils.typing import ReduceType


class Reduce(nn.Module):
    r"""A template class for implementing the :math:`\texttt{reduce}` operator."""

    @staticmethod
    def reduce_batch(
        select_output: SelectOutput,
        batch: Optional[Tensor],
    ) -> Optional[Tensor]:
        r"""Computes the batch vector of the coarsened graph.

        Args:
            select_output (~tgp.select.SelectOutput):
                The output of :class:`~tgp.select.Select`.
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)

        Returns:
            :class:`~torch.Tensor` or :obj:`None`: The pooled batch.
        """
        if batch is None:
            return batch

        if isinstance(select_output.s, SparseTensor):
            out = torch.arange(select_output.num_supernodes, device=batch.device)
            return out.scatter_(
                0, select_output.cluster_index, batch[select_output.node_index]
            )
        else:
            # Dense [N, K] tensor with batch vector
            # Each graph in the batch has K supernodes
            K = select_output.num_supernodes

            # Handle empty batch case
            if batch.numel() == 0:
                return batch.new_empty((0,), dtype=batch.dtype)

            batch_size = int(batch.max().item()) + 1

            # batch_pooled assigns each supernode to its graph:
            # - Supernodes 0 to K-1 belong to graph 0
            # - Supernodes K to 2K-1 belong to graph 1
            # - etc.
            batch_pooled = torch.arange(
                batch_size, dtype=batch.dtype, device=batch.device
            ).repeat_interleave(K)

            return batch_pooled

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(
        self,
        x: Tensor,
        so: SelectOutput,
        *,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (~torch.Tensor):
                The node feature matrix. For a sparse pooler, :obj:`x` has shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and :math:`F` is the number of node features.
                For a dense pooler, :obj:`x` has shape :math:`[B, N, F]`, where :math:`B` is the batch size.
            so (~tgp.select.SelectOutput): The output of the :math:`\texttt{select}` operator.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)

        Returns:
            ~torch.Tensor: The pooled features :math:`\mathbf{X}_{pool}` of the supernodes.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseReduce(Reduce):
    r"""The basic :math:`\texttt{reduce}` operator to compute the node features
    of the :math:`k`-th pooled node.

    .. math::
        (\mathbf{x}_k)_\text{pool} = \text{aggr}(s_{1,k} \mathbf{x}_1, \ldots, s_{N,k} \mathbf{x}_N),

    where aggr is an aggregation function such as the sum or the mean.

    Args:
        reduce_red_op (~tgp.utils.typing.ReduceType, optional):
            The aggregation function to be applied to nodes in the same cluster. Can be
            any string admitted by :obj:`~torch_geometric.utils.scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`~tgp.utils.typing.ReduceType`.
            (default: :obj:`None`)
    """

    def __init__(self, reduce_op: ReduceType = "sum"):
        super().__init__()

        self.operation = reduce_op

    def forward(
        self,
        x: Tensor,
        so: SelectOutput,
        batch: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x (~torch.Tensor):
                The node feature matrix. For a sparse pooler, :obj:`x` has shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and :math:`F` is the number of node features.
                For a dense pooler, :obj:`x` has shape :math:`[B, N, F]`, where :math:`B` is the batch size.
            so (~tgp.select.SelectOutput): The output of the :math:`\texttt{select}` operator.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
        """
        # If batch is not provided, try to retrieve it from SelectOutput
        # This is necessary for multi-graph batches with dense [N, K] tensors,
        # where we need to know which nodes belong to which graph
        if batch is None and hasattr(so, "batch") and so.batch is not None:
            batch = so.batch

        if isinstance(so.s, SparseTensor):
            if self.operation == "any":
                x_pool = scatter(
                    x[so.node_index], so.cluster_index, dim=0, reduce="any"
                )
                if so.s.storage.value() is not None:
                    x_pool = x_pool * so.s.storage.value().view(-1, 1)
            else:
                x_pool = torch_sparse.matmul(so.s.t(), x, reduce=self.operation)
        else:
            # Dense assignment matrix
            if so.s.dim() == 3:
                # Dense [B, N, K] tensor (standard dense pooler format)
                x_pool = so.s.transpose(-2, -1).matmul(x)
            elif batch is not None and batch.numel() > 0:
                # Check if multi-graph batch
                batch_min = int(batch.min().item())
                batch_max = int(batch.max().item())
                if batch_min != batch_max:
                    # Multi-graph batch with dense [N, K] tensor
                    # Process each graph separately and concatenate
                    unbatched_s = unbatch(so.s, batch)  # list of [N_i, K] tensors
                    unbatched_x = unbatch(x, batch)  # list of [N_i, F] tensors

                    x_pool_list = []
                    for s_i, x_i in zip(unbatched_s, unbatched_x):
                        # x_pool_i = S_i^T @ X_i: [K, N_i] @ [N_i, F] = [K, F]
                        x_pool_i = s_i.t().matmul(x_i)
                        x_pool_list.append(x_pool_i)

                    x_pool = torch.cat(x_pool_list, dim=0)  # [B*K, F]
                else:
                    # Single-graph batch: simple matmul
                    x_pool = so.s.transpose(-2, -1).matmul(x)
            else:
                # Single graph without batch: simple matmul
                x_pool = so.s.transpose(-2, -1).matmul(x)

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(reduce_op={self.operation})"
