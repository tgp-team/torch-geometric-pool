from typing import List, Optional, Union

import torch
from torch import Tensor

from tgp.connect import DenseConnect
from tgp.data import NormalizeAdj
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DenseSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import just_balance_loss
from tgp.utils.typing import LiftType, SinvType


class JustBalancePooling(DenseSRCPooling):
    r"""The Just Balance pooling operator from the paper `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ (Bianchi et al., NLDL 2023).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DenseSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer optimizes an auxiliary balance loss (:func:`~tgp.utils.losses.just_balance_loss`)

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
        normalize_loss (bool, optional): If set to :obj:`True`, the loss is normalized by the number of nodes
            (default: :obj:`True`)
        loss_coeff (float, optional): Coefficient for the loss (default: :obj:`1.0`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the adjacency matrix.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
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
        normalize_loss: bool = True,
        loss_coeff: float = 1.0,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
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
                edge_weight_norm=edge_weight_norm,
            ),
            adj_transpose=adj_transpose,
        )

        self.normalize_loss = normalize_loss
        self.loss_coeff = loss_coeff

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

            loss = self.compute_loss(so.s, mask, so.num_nodes, so.num_supernodes)

            out = PoolingOutput(x=x_pooled, edge_index=adj_pooled, so=so, loss=loss)

            return out

    def compute_loss(
        self,
        S: Tensor,
        mask: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        num_supernodes: Optional[int] = None,
    ) -> dict:
        r"""Computes the auxiliary loss term.

        Args:
            S (~torch.Tensor): The dense assignment matrix.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)
            num_nodes (int, optional): The number of nodes.
            (default: :obj:`None`)
            num_supernodes (int, optional): The number of clusters.
            (default: :obj:`None`)

        Returns:
            dict: A dictionary containing the balance loss.
        """
        loss = just_balance_loss(
            S,
            mask,
            num_nodes=num_nodes,
            num_supernodes=num_supernodes,
            normalize_loss=self.normalize_loss,
            batch_reduction="mean",
        )

        if torch.isnan(loss):
            raise ValueError("Loss is NaN")

        return {"balance_loss": loss * self.loss_coeff}

    @staticmethod
    def data_transforms():
        r"""Transforms the adjacency matrix :math:`\mathbf{A}`
        by applying the following transformation:

        .. math::
            \mathbf{A} \to \mathbf{I} - \delta \mathbf{L}

        where :math:`\mathbf{L}` is the normalized Laplacian
        of the graph and :math:`\delta` is a scaling factor.
        By default, :math:`\delta` is set to :math:`0.85`.
        """
        return NormalizeAdj(delta=0.85)

    def extra_repr_args(self) -> dict:
        return {
            "loss_coeff": self.loss_coeff,
            "normalize_loss": self.normalize_loss,
        }
