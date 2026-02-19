from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.models.mlp import MLP

from tgp.select import Select, SelectOutput
from tgp.utils.typing import SinvType


class MLPSelect(Select):
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
        batched_representation (bool, optional):
            If :obj:`True`, expects batched input :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`
            and returns assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`.
            If :obj:`False`, expects unbatched input :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`
            where :math:`N` is the total number of nodes across all graphs, and returns
            assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{N \times K}`.
            (default: :obj:`True`)
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

    is_dense: bool = True

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,
        batched_representation: bool = True,
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
        self.batched_representation = batched_representation
        self.act = act
        self.dropout = dropout

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.mlp.reset_parameters()

    def _prepare_inputs(self, x: Tensor) -> Tensor:
        """Prepare inputs according to the expected representation."""
        if self.batched_representation:
            return x.unsqueeze(0) if x.dim() == 2 else x
        assert x.dim() == 2, "x must be of shape [N, F] for unbatched mode"
        return x

    def _apply_mask(self, s: Tensor, mask: Optional[Tensor]) -> Tensor:
        """Apply a node mask to batched assignment matrices when provided."""
        if mask is not None:
            s = s * mask.unsqueeze(-1)
        return s

    def _build_output(
        self,
        s: Tensor,
        *,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **extra,
    ) -> SelectOutput:
        """Create a SelectOutput with the correct batched/unbatched fields."""
        if self.batched_representation:
            return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask, **extra)
        return SelectOutput(s=s, s_inv_op=self.s_inv_op, batch=batch, **extra)

    def forward(
        self,
        x: Tensor,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor.
                If :obj:`batched_representation=True`, expected shape is
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                If :obj:`batched_representation=False`, expected shape is
                :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`, where :math:`N`
                is the total number of nodes across all graphs in the batch.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. Only used when
                :obj:`batched_representation=True`. (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. Only used when
                :obj:`batched_representation=False`. (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
                If :obj:`batched_representation=True`, the assignment matrix :math:`\mathbf{S}`
                has shape :math:`\mathbb{R}^{B \times N \times K}`.
                If :obj:`batched_representation=False`, the assignment matrix :math:`\mathbf{S}`
                has shape :math:`\mathbb{R}^{N \times K}`.
        """
        x = self._prepare_inputs(x)
        s = self.mlp(x)
        s = torch.softmax(s, dim=-1)

        if self.batched_representation:
            s = self._apply_mask(s, mask)
            return self._build_output(s, mask=mask)

        return self._build_output(s, batch=batch)

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"k={self.k}, "
            f"act={self.act}, "
            f"dropout={self.dropout}, "
            f"s_inv_op={self.s_inv_op})"
        )
