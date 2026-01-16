from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import MLP

from tgp.select import Select, SelectOutput
from tgp.utils.typing import SinvType


class DenseSelect(Select):
    r"""The :math:`\texttt{select}` operator used by most of the dense pooling methods.

    It computes a dense assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
    from the node features :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`:

    .. math::
        \mathbf{S} = \mathrm{softmax}(\texttt{MLP}(\mathbf{X}))

    Args:
        in_channels (int, list of int):
            Number of hidden units for each hidden layer in the
            :class:`~torch_geometric.nn.models.mlp.MLP` used to
            compute cluster assignments.
            The first integer must match the size of the node features.
        k (int):
            Number of clusters or supernodes in the pooler graph.
        act (str or Callable, optional):
            Activation function in the hidden layers of the
            :class:`~torch_geometric.nn.models.mlp.MLP`.
        dropout (float, optional): Dropout probability in the
            :class:`~torch_geometric.nn.models.mlp.MLP`.
            (default: :obj:`0.0`)
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    is_dense_batched: bool = True

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,
        act: str = None,
        dropout: float = 0.0,
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__()

        in_channels = [in_channels] if isinstance(in_channels, int) else in_channels

        self.mlp = MLP(in_channels + [k], act=act, norm=None, dropout=dropout)
        self.s_inv_op = s_inv_op
        self.in_channels = in_channels
        self.k = k
        self.act = act
        self.dropout = dropout

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the node assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}` is
                being created within this method.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x

        s = self.mlp(x)
        s = torch.softmax(s, dim=-1)  # has shape (B, N, K)

        if mask is not None:
            s = s * mask.unsqueeze(-1)

        return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"k={self.k}, "
            f"act={self.act}, "
            f"dropout={self.dropout}, "
            f"s_inv_op={self.s_inv_op})"
        )
