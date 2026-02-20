from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, unbatch

from tgp.select import SelectOutput


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

        if select_output.s.is_sparse:
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
    ) -> Tuple[Tensor, Optional[Tensor]]:
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
    r"""The basic :math:`\texttt{reduce}` operator that computes :math:`\mathbf{S}^\top \mathbf{X}`.

    For **sparse** assignment :math:`\mathbf{S}`, this is implemented as a sum over nodes
    in each cluster (with optional weighting by :obj:`so.s.values()`). For **dense**
    assignment, it is a matrix multiply :math:`\mathbf{S}^\top \mathbf{X}`.

    For dense multi-graph batches (dense :math:`[N, K]` with a batch vector), each graph
    is processed separately (unbatch then per-graph matmul) for memory efficiency when
    using unbatched dense poolers (:obj:`batched=False`).

    :obj:`return_batched` is only used in the unbatched (sparse) path when returning
    multi-graph results; in the dense path it is ignored and output shape follows
    input (dense in :math:`\Rightarrow` dense out).
    """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        x: Tensor,
        so: SelectOutput,
        *,
        batch: Optional[Tensor] = None,
        return_batched: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward pass computing :math:`\mathbf{S}^\top \mathbf{X}`.

        Args:
            x (~torch.Tensor):
                The node feature matrix. For a sparse pooler, :obj:`x` has shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and :math:`F` is the number of node features.
                For a dense pooler, :obj:`x` has shape :math:`[B, N, F]`, where :math:`B` is the batch size.
            so (~tgp.select.SelectOutput): The output of the :math:`\texttt{select}` operator.
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            return_batched (bool, optional):
                Only used in the **unbatched (sparse) path** for multi-graph batches with
                dense :math:`[N, K]` assignment. If :obj:`True`, returns shape :math:`[B, K, F]`;
                otherwise :math:`[B \cdot K, F]`. Ignored in the dense path (dense in
                :math:`\Rightarrow` dense out). (default: :obj:`False`)
        """
        if batch is None and hasattr(so, "batch") and so.batch is not None:
            batch = so.batch

        if so.s.is_sparse:
            if return_batched:
                raise ValueError(
                    "return_batched=True is only supported for dense assignment matrices."
                )
            src = x[so.node_index]
            values = so.s.values()
            if values is not None:
                src = src * values.view(-1, 1)
            x_pool = scatter(
                src,
                so.cluster_index,
                dim=0,
                dim_size=so.num_supernodes,
                reduce="sum",
            )
        else:
            # Dense assignment: S^T @ x
            if so.s.dim() == 3:
                # Dense [B, N, K] tensor (standard dense pooler format)
                x_pool = so.s.transpose(-2, -1).matmul(x)
            elif batch is not None and batch.numel() > 0:
                # Check if multi-graph batch
                batch_min = int(batch.min().item())
                batch_max = int(batch.max().item())
                if batch_min != batch_max:
                    # Multi-graph batch with dense [N, K] tensor
                    # Process each graph separately and concatenate or stack
                    unbatched_s = unbatch(so.s, batch)  # list of [N_i, K] tensors
                    unbatched_x = unbatch(x, batch)  # list of [N_i, F] tensors

                    x_pool_list = []
                    for s_i, x_i in zip(unbatched_s, unbatched_x):
                        # x_pool_i = S_i^T @ X_i: [K, N_i] @ [N_i, F] = [K, F]
                        x_pool_i = s_i.t().matmul(x_i)
                        x_pool_list.append(x_pool_i)
                    if return_batched:
                        x_pool = torch.stack(x_pool_list, dim=0)  # [B, K, F]
                    else:
                        x_pool = torch.cat(x_pool_list, dim=0)  # [B*K, F]
                else:
                    # Single-graph batch: simple matmul
                    x_pool = so.s.transpose(-2, -1).matmul(x)
            else:
                # Single graph without batch: simple matmul
                x_pool = so.s.transpose(-2, -1).matmul(x)
            if return_batched and x_pool.dim() == 2:
                x_pool = x_pool.unsqueeze(0)

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
