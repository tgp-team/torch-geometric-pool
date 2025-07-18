from typing import List, Optional, Union

from torch import Tensor

from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DenseSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import asym_norm_loss, totvar_loss
from tgp.utils.typing import LiftType, SinvType


class AsymCheegerCutPooling(DenseSRCPooling):
    r"""The asymmetric cheeger cut pooling layer from the paper `"Total Variation
    Graph Neural Networks" <https://arxiv.org/abs/2211.06218>`_
    (Hansen & Bianchi, ICML 2023).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DenseSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer optimizes two auxiliary losses:

    + the total variation loss (:class:`~tgp.utils.losses.totvar_loss`),
    + the asymmetric norm loss (:class:`~tgp.utils.losses.asym_norm_loss`).

    Args:
        in_channels (int, list of int):
            Number of hidden units for each hidden layer in the MLP
            of the :math:`\texttt{select}` operator.
            The first integer must match the size of the node features.
        k (int):
            Number of clusters or supernodes in the pooler graph.
        act (str or Callable, optional):
            Activation function in the hidden layers of the MLP
            of the :math:`\texttt{select}` operator.
        dropout (float, optional):
            Dropout probability in the MLP of the :math:`\texttt{select}` operator.
            (default: :obj:`0.0`)
        totvar_coeff (float):
            Coefficient for graph total variation loss term. (default: :obj:`1.0`)
        balance_coeff (float):
            Coefficient for asymmetric norm loss term. (default: :obj:`1.0`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the adjacency matrix.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        adj_transpose (bool, optional):
            If :obj:`True`, the preprocessing step in :class:`~tgp.src.DenseSRCPooling` and
            the :class:`~tgp.connect.DenseConnect` operation returns transposed
            adjacency matrices, so that they could be passed "as is" to the dense
            message-passing layers.
            (default: :obj:`True`)
        lift (~tgp.typing.LiftType, optional):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - :obj:`"precomputed"` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the :obj:`"s_inv"` attribute of the :class:`~tgp.select.SelectOutput`.
            - :obj:`"transpose"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,
        act: str = None,
        dropout: float = 0.0,
        totvar_coeff: float = 1.0,
        balance_coeff: float = 1.0,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__(
            selector=DenseSelect(
                in_channels=in_channels,
                k=k,
                act=act,
                dropout=dropout,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
            ),
            adj_transpose=adj_transpose,
        )

        self.k = k
        self.totvar_coeff = totvar_coeff
        self.balance_coeff = balance_coeff

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (~torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            # Select
            so = self.select(x=x, mask=mask)

            # Reduce
            x_pooled, _ = self.reduce(x=x, so=so)

            # Connect
            adj_pooled, _ = self.connect(edge_index=adj, so=so)

            loss = self.compute_loss(adj, so.s)

            out = PoolingOutput(x=x_pooled, edge_index=adj_pooled, so=so, loss=loss)

            return out

    def compute_loss(self, adj: Tensor, S: Tensor) -> dict:
        r"""Computes the auxiliary loss terms.

        Args:
            adj (~torch.Tensor): The dense adjacency matrix.
            S (~torch.Tensor): The dense assignment matrix.

        Returns:
            dict: A dictionary with the different terms of
            the auxiliary loss.
        """
        tv_loss = totvar_loss(S, adj, batch_reduction="mean")
        bal_loss = asym_norm_loss(S, self.k, batch_reduction="mean")
        return {
            "total_variation_loss": tv_loss * self.totvar_coeff,
            "balance_loss": bal_loss * self.balance_coeff,
        }

    def extra_repr_args(self) -> dict:
        return {"totvar_coeff": self.totvar_coeff, "balance_coeff": self.balance_coeff}
