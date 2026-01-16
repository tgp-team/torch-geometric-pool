from typing import Optional

import torch
from sklearn.decomposition import non_negative_factorization
from torch import Tensor

from tgp.select import Select, SelectOutput
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

    is_dense_batched: bool = True

    def __init__(self, k: int, s_inv_op: SinvType = "transpose"):
        super().__init__()

        self.k = k
        self.s_inv_op = s_inv_op

    def forward(
        self, edge_index: Tensor, mask: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        r"""Forward pass of the select operator.

        Args:
            edge_index (~torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph. (default: :obj:`None`)

        Returns:
           :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        device = edge_index.device

        B, N, M = edge_index.size()
        edge_index = edge_index.permute(1, 0, 2).reshape(edge_index.size(1), -1)
        A = edge_index.cpu().numpy()
        _, H, _ = non_negative_factorization(
            A, n_components=self.k, init="random", max_iter=5000
        )
        H = torch.tensor(H, device=device)

        H = H.reshape(H.size(0), B, M).permute(1, 2, 0)
        H = torch.softmax(H, dim=-1)

        if mask is not None:
            H = H * mask.unsqueeze(-1)

        so = SelectOutput(s=H, s_inv_op=self.s_inv_op, mask=mask)

        return so

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k}, s_inv_op={self.s_inv_op})"
