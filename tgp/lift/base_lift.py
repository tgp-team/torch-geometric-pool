from typing import Optional

import torch
import torch_sparse
from torch import Tensor, nn
from torch_geometric.utils import unbatch

from tgp.select import SelectOutput
from tgp.utils import pseudo_inverse
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
    + :math:`f(\cdot)` is the lifting operation that specifies of :math:`\mathbf{S}_{\text{inv}}` is used to
      computed the lifted features :math:`\mathbf{X}_{\text{lift}}`. In most cases,
      :math:`f(\mathbf{S}_{\text{inv}}, \mathbf{X}_{\text{pool}}) = \mathbf{S}_{\text{inv}}^{\top} \mathbf{X}_{\text{pool}}`.

    It works also for *dense* pooling operators. In that case,
    :math:`\mathbf{X}_{\text{lift}} \in \mathbb{R}^{B \times N \times F}`,
    :math:`\mathbf{S}_{\text{inv}} \in \mathbb{R}^{B \times K \times N}`,
    :math:`\mathbf{X}_{\text{pool}} \in \mathbb{R}^{B \times K \times F}`.

    Args:
        matrix_op (~tgp.typing.LiftType):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - :obj:`"precomputed"` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the :obj:`"s_inv"` attribute of the :class:`~tgp.select.SelectOutput`.
            - :obj:`"transpose"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
        reduce_op (~tgp.typing.ReduceType):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
    """

    def __init__(
        self, matrix_op: LiftType = "precomputed", reduce_op: ReduceType = "sum"
    ):
        super().__init__()
        self.matrix_op = matrix_op
        self.reduce_op = reduce_op

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
                If not provided, will use so.batch if available.
                Required for multi-graph batches with dense [N, K] assignment matrices.
                (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                The batch vector for the pooled nodes.
                Required for multi-graph batches with dense [N, K] assignment matrices.
                (default: :obj:`None`)

        Returns:
            x_lift (~torch.Tensor):
                The lifted node features.
        """
        # Use so.batch if batch is not provided
        if batch is None and so.batch is not None:
            batch = so.batch
        if self.matrix_op == "precomputed":
            if isinstance(so.s, Tensor):
                s_inv = so.s_inv.transpose(-2, -1)
            else:
                s_inv = so.s_inv.t()
        elif self.matrix_op == "inverse":
            if isinstance(so.s, Tensor):
                s_inv = pseudo_inverse(so.s).transpose(-2, -1)
            else:
                s_inv = pseudo_inverse(so.s).t()
        elif self.matrix_op == "transpose":
            s_inv = so.s
        else:
            raise RuntimeError(
                f"'matrix_op' must be one of {list(LiftType.__args__)} ({self.matrix_op} given)"
            )

        if isinstance(s_inv, Tensor):
            K = s_inv.size(-1)

            is_multi_graph = (
                batch is not None
                and batch.numel() > 0
                and int(batch.min().item()) != int(batch.max().item())
            )

            # Dense [N, K] + multi-graph batches: pooled features may be flattened as [B*K, F].
            if s_inv.dim() == 2 and x_pool.dim() == 2 and is_multi_graph:
                batch_size = int(batch.max().item()) + 1
                expected_nodes = batch_size * K

                if x_pool.size(0) == expected_nodes:
                    if batch_pooled is None:
                        raise ValueError(
                            "batch_pooled must be provided when lifting with dense [N, K] SelectOutput "
                            "and multi-graph batches. Pass it as: "
                            "pooler(x=x_pool, so=so, lifting=True, batch_pooled=batch_pooled)"
                        )

                    if batch_pooled.size(0) != x_pool.size(0):
                        raise ValueError(
                            "batch_pooled has an unexpected length for dense [N, K] lifting "
                            f"(got {batch_pooled.size(0)}, expected {x_pool.size(0)})."
                        )

                    unbatched_s_inv = unbatch(s_inv, batch)  # list of [N_i, K] tensors
                    unbatched_x_pool = unbatch(
                        x_pool, batch_pooled
                    )  # list of [K, F] tensors

                    x_lifted_list = []
                    for s_inv_i, x_pool_i in zip(unbatched_s_inv, unbatched_x_pool):
                        x_lifted_list.append(s_inv_i.matmul(x_pool_i))

                    x_prime = torch.cat(x_lifted_list, dim=0)  # [N, F]
                elif x_pool.size(0) == K:
                    # Pooling produced a global [K, F] tensor across the batch.
                    x_prime = s_inv.matmul(x_pool)
                else:
                    raise ValueError(
                        "Unexpected pooled feature shape for dense [N, K] lifting with a multi-graph batch: "
                        f"got x_pool.size(0)={x_pool.size(0)}, expected {K} or {expected_nodes}."
                    )
            else:
                x_prime = s_inv.matmul(x_pool)
        else:
            x_prime = torch_sparse.matmul(s_inv, x_pool, reduce=self.reduce_op)

        return x_prime

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"matrix_op={self.matrix_op}, "
            f"reduce_op={self.reduce_op})"
        )
