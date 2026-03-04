from typing import Optional, Tuple

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, unbatch

from tgp.select import SelectOutput
from tgp.utils.ops import build_pooled_batch, is_multi_graph_batch


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
            return None

        # Sparse assignment: each selected supernode inherits the graph id of
        # the node that maps to it.
        if select_output.s.is_sparse:
            out = torch.arange(select_output.num_supernodes, device=batch.device)
            return out.scatter_(
                0, select_output.cluster_index, batch[select_output.node_index]
            )

        # Dense assignment: each graph contributes exactly K pooled nodes.
        if batch.numel() == 0:
            return batch.new_empty((0,), dtype=batch.dtype)

        batch_size = int(batch.max().item()) + 1
        return build_pooled_batch(
            batch_size,
            select_output.num_supernodes,
            batch.device,
            dtype=batch.dtype,
        )

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

    For dense unbatched assignments :math:`[N, K]` with multi-graph batches,
    :obj:`return_batched=True` returns :math:`[B, K, F]`; otherwise
    :math:`[B \cdot K, F]`. For dense batched assignments :math:`[B, N, K]`,
    output is always :math:`[B, K, F]`.
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
                For dense unbatched :math:`[N, K]` assignments with multi-graph
                batches, controls output shape:
                :obj:`True` gives :math:`[B, K, F]`, :obj:`False` gives
                :math:`[B \cdot K, F]`. For single-graph :math:`[N, K]`,
                :obj:`True` wraps output as :math:`[1, K, F]`.
                (default: :obj:`False`)
        """
        if batch is None and so.batch is not None:
            batch = so.batch

        # Path 1: sparse assignment matrix (edge list style). Aggregate selected
        # node features into supernodes via scatter.
        if so.s.is_sparse:
            if return_batched:
                raise ValueError(
                    "return_batched=True is only supported for dense assignment matrices."
                )
            src = x[so.node_index] * so.weight.view(-1, 1)
            x_pool = scatter(
                src,
                so.cluster_index,
                dim=0,
                dim_size=so.num_supernodes,
                reduce="sum",
            )
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        # Path 2: dense batched assignment [B, N, K] and dense features [B, N, F].
        if so.s.dim() == 3:
            x_pool = so.s.transpose(-2, -1).matmul(x)  # [B, K, F]
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        if so.s.dim() != 2:
            raise ValueError(
                "Dense SelectOutput.s must be 2D [N, K] or 3D [B, N, K], "
                f"got ndim={so.s.dim()}."
            )

        # Path 3: dense unbatched assignment [N, K] with multi-graph batch.
        if is_multi_graph_batch(batch):
            unbatched_s = unbatch(so.s, batch)  # list of [N_i, K]
            unbatched_x = unbatch(x, batch)  # list of [N_i, F]
            x_pool_per_graph = [
                s_i.t().matmul(x_i) for s_i, x_i in zip(unbatched_s, unbatched_x)
            ]
            x_pool = (
                torch.stack(x_pool_per_graph, dim=0)
                if return_batched
                else torch.cat(x_pool_per_graph, dim=0)
            )
            batch_pool = self.reduce_batch(so, batch)
            return x_pool, batch_pool

        # Path 4: dense unbatched assignment [N, K] for a single graph.
        x_pool = so.s.transpose(-2, -1).matmul(x)  # [K, F]
        if return_batched:
            x_pool = x_pool.unsqueeze(0)  # [1, K, F]

        batch_pool = self.reduce_batch(so, batch)
        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
