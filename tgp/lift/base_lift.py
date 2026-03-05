from typing import Optional

import torch
from torch import Tensor, nn
from torch_geometric.utils import scatter, unbatch

from tgp.select import SelectOutput
from tgp.utils import (
    build_pooled_batch,
    expand_compacted_rows,
    is_multi_graph_batch,
    pseudo_inverse,
)
from tgp.utils.typing import LiftType, ReduceType


class Lift(nn.Module):
    """A template class for the lift operator."""

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        pass

    def forward(self, x_pool: Tensor, so: SelectOutput, **kwargs) -> Tensor:
        r"""Forward pass.

        Args:
            x_pool (~torch.Tensor):
                the pooled node features :math:`\mathbf{X}_{\text{pool}} \in \mathbb{R}^{K \times F}`
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
        """
        raise NotImplementedError

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"


class BaseLift(Lift):
    r"""A base class to lift the features of the supernodes back into the original node space.

    The lift operation is implemented as

    .. math::
        \mathbf{X}_{\text{lift}} = f(\mathbf{S}_{\text{inv}}, \mathbf{X}_{\text{pool}}) \approx \mathbf{X}.

    where

    + :math:`\mathbf{X}_{\text{lift}} \in \mathbb{R}^{N \times F}` are the lifted node features,
    + :math:`\mathbf{S}_{\text{inv}} \in \mathbb{R}^{K \times N}` is the inverse node assignment operator,
    + :math:`\mathbf{X}_{\text{pool}} \in \mathbb{R}^{K \times F}` are the pooled features of the supernodes,
    + :math:`f(\cdot)` is the lifting operation that specifies how :math:`\mathbf{S}_{\text{inv}}` is used to
      compute the lifted features :math:`\mathbf{X}_{\text{lift}}`. In most cases,
      :math:`f(\mathbf{S}_{\text{inv}}, \mathbf{X}_{\text{pool}}) = \mathbf{S}_{\text{inv}}^{\top} \mathbf{X}_{\text{pool}}`.

    It also works for *dense* pooling operators. In that case,
    :math:`\mathbf{X}_{\text{lift}} \in \mathbb{R}^{B \times N \times F}`,
    :math:`\mathbf{S}_{\text{inv}} \in \mathbb{R}^{B \times K \times N}`,
    :math:`\mathbf{X}_{\text{pool}} \in \mathbb{R}^{B \times K \times F}`.

    Args:
        matrix_op (~tgp.utils.typing.LiftType):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - ``"precomputed"`` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the ``"s_inv"`` attribute of the :class:`~tgp.select.SelectOutput`.
            - ``"transpose"``: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - ``"inverse"``: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
        reduce_op (~tgp.utils.typing.ReduceType):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., ``'sum'``, ``'mean'``, ``'max'``)
            (default: ``"sum"``)
    """

    def __init__(
        self, matrix_op: LiftType = "precomputed", reduce_op: ReduceType = "sum"
    ):
        super().__init__()
        self.matrix_op = matrix_op
        self.reduce_op = reduce_op

    def _get_lift_matrix(self, so: SelectOutput) -> Tensor:
        if self.matrix_op == "transpose":
            return so.s

        if self.matrix_op == "precomputed":
            matrix = so.s_inv
        elif self.matrix_op == "inverse":
            matrix = pseudo_inverse(so.s)
        else:
            raise RuntimeError(
                f"'matrix_op' must be one of {list(LiftType.__args__)} ({self.matrix_op} given)"
            )

        matrix = matrix.transpose(-2, -1)
        return matrix.coalesce() if matrix.is_sparse else matrix

    def _lift_sparse(self, lift_matrix: Tensor, x_pool: Tensor) -> Tensor:
        row, col = lift_matrix.indices()
        src = x_pool[col] * lift_matrix.values().view(-1, 1)
        return scatter(
            src,
            row,
            dim=0,
            dim_size=lift_matrix.size(0),
            reduce=self.reduce_op,
        )

    @staticmethod
    def _lift_dense_multi_graph(
        lift_matrix: Tensor,
        x_pool_flat: Tensor,
        batch: Tensor,
        batch_pooled: Tensor,
    ) -> Tensor:
        unbatched_lift = unbatch(lift_matrix, batch)  # list of [N_i, K] tensors
        unbatched_x_pool = unbatch(x_pool_flat, batch_pooled)  # list of [K, F] tensors

        if len(unbatched_lift) != len(unbatched_x_pool):
            raise ValueError(
                "Inconsistent per-graph blocks while lifting dense [N, K] assignments: "
                f"got {len(unbatched_lift)} assignment blocks and {len(unbatched_x_pool)} "
                "pooled feature blocks."
            )

        return torch.cat(
            [
                lift_i.matmul(x_pool_i)
                for lift_i, x_pool_i in zip(unbatched_lift, unbatched_x_pool)
            ],
            dim=0,
        )

    def forward(
        self,
        x_pool: Tensor,
        so: SelectOutput = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward pass of the Lift operation.

        Args:
            x_pool (~torch.Tensor):
                The pooled node features.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.
            batch (~torch.Tensor, optional):
                The batch vector for the original nodes.
                If not provided, ``so.batch`` is used when available.
                (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                The batch vector for the pooled nodes.
                For dense multi-graph lifting with flattened pooled features
                :math:`[B \cdot K, F]`, if not provided it is inferred as contiguous
                graph blocks of size :math:`K`.
                (default: :obj:`None`)

        Returns:
            ~torch.Tensor:
                The lifted node features.
        """
        if batch is None and so.batch is not None:
            batch = so.batch

        lift_matrix = self._get_lift_matrix(so)

        if lift_matrix.is_sparse:
            return self._lift_sparse(lift_matrix, x_pool)

        num_clusters = lift_matrix.size(-1)
        is_multi_graph = is_multi_graph_batch(batch)

        # Case 1: dense unbatched assignment [N, K] with flattened pooled features [B*K, F].
        if lift_matrix.dim() == 2 and x_pool.dim() == 2 and is_multi_graph:
            batch_size = int(batch.max().item()) + 1
            expected_nodes = batch_size * num_clusters

            # Global pooled tensor shared across graphs: [K, F].
            if x_pool.size(0) == num_clusters:
                return lift_matrix.matmul(x_pool)

            if x_pool.size(0) != expected_nodes:
                raise ValueError(
                    "Unexpected pooled feature shape for dense [N, K] lifting with a multi-graph batch: "
                    f"got x_pool.size(0)={x_pool.size(0)}, expected {num_clusters} or {expected_nodes}."
                )

            if batch_pooled is None:
                # Default layout: each graph contributes exactly K pooled rows, stored contiguously.
                batch_pooled = build_pooled_batch(
                    batch_size, num_clusters, x_pool.device
                )
            elif batch_pooled.size(0) != x_pool.size(0):
                raise ValueError(
                    "batch_pooled has an unexpected length for dense [N, K] lifting "
                    f"(got {batch_pooled.size(0)}, expected {x_pool.size(0)})."
                )
            return self._lift_dense_multi_graph(
                lift_matrix, x_pool, batch, batch_pooled
            )

        # Case 2: dense unbatched assignment [N, K] with batched pooled features [B, K, F].
        elif lift_matrix.dim() == 2 and x_pool.dim() == 3:
            if not is_multi_graph:
                return lift_matrix.matmul(x_pool.squeeze(0))

            batch_size = x_pool.size(0)
            expected_nodes = batch_size * num_clusters
            x_pool_flat = x_pool.reshape(expected_nodes, x_pool.size(-1))

            if batch_pooled is None:
                # Same default layout used by reduce/connect for dense [N, K] paths.
                batch_pooled = build_pooled_batch(
                    batch_size, num_clusters, x_pool.device
                )
            elif batch_pooled.size(0) != expected_nodes:
                raise ValueError(
                    "batch_pooled has an unexpected length for dense [N, K] lifting "
                    f"(got {batch_pooled.size(0)}, expected {expected_nodes})."
                )
            return self._lift_dense_multi_graph(
                lift_matrix, x_pool_flat, batch, batch_pooled
            )

        # Case 3: dense batched assignment [B, N, K] with flattened pooled features [B*K, F].
        elif lift_matrix.dim() == 3 and x_pool.dim() == 2:
            batch_size = lift_matrix.size(0)
            expected_nodes = batch_size * num_clusters

            if x_pool.size(0) != expected_nodes:
                x_pool = expand_compacted_rows(
                    x_compact=x_pool,
                    valid_mask=so.out_mask,
                    expected_rows=expected_nodes,
                )

            x_pool = x_pool.view(batch_size, num_clusters, x_pool.size(-1))
            return lift_matrix.matmul(x_pool)

        # Case 4: dense inputs already aligned for direct matmul.
        return lift_matrix.matmul(x_pool)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"matrix_op={self.matrix_op}, "
            f"reduce_op={self.reduce_op})"
        )
