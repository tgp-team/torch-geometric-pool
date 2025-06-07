import torch_sparse
from torch import Tensor, nn

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

    def forward(self, x_pool: Tensor, so: SelectOutput = None, **kwargs) -> Tensor:
        r"""Forward pass of the Lift operation.

        Args:
            x_pool (~torch.Tensor):
                The pooled node features.
            so (~tgp.select.SelectOutput):
                The output of the :math:`\texttt{select}` operator.

        Returns:
            x_lift (~torch.Tensor):
                The lifted node features.
        """
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
