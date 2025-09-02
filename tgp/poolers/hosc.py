from typing import List, Optional, Union

import torch
from torch import Tensor

from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DenseSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import hosc_orthogonality_loss, mincut_loss, orthogonality_loss
from tgp.utils.typing import LiftType, SinvType


class HOSCPooling(DenseSRCPooling):
    r"""The high-order pooling operator from the paper
    `"Higher-order clustering and pooling for Graph Neural Networks"
    <http://arxiv.org/abs/2209.03473>`_ (Duval & Malliaros, CIKM 2022)..

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DenseSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer optimizes a combination of the following auxiliary losses:

    + the mincut loss (:class:`~tgp.utils.losses.mincut_loss`),
    + the orthogonality loss (:class:`~tgp.utils.losses.orthogonality_loss`),
    + the hosc orthogonality loss (:class:`~tgp.utils.losses.hosc_orthogonality_loss`),

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
        mu (float, optional):
            A scalar that controls the importance given to regularization loss.
            (default: :obj:`0.1`)
        alpha (float, optional):
            A scalar in [0,1] controlling the importance granted
            to higher-order information in the loss function.
            (default: :obj:`0.5`)
        hosc_ortho (bool, optional):
            Specifies either to use the hosc_orthogonality_loss or the
            orthogonality_loss.
            (default: :obj:`False`)
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
        mu: float = 0.1,
        alpha: float = 0.5,
        hosc_ortho: bool = False,
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

        self.k = k
        self.mu = mu
        self.alpha = alpha
        self.hosc_ortho = hosc_ortho

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
            adj_pool = self.connector.dense_connect(adj=adj, s=so.s)

            loss = self.compute_loss(adj, so.s, adj_pool, mask)

            # Normalize coarsened adjacency matrix
            adj_pool = self.connector.postprocess_adj_pool(
                adj_pool,
                remove_self_loops=self.connector.remove_self_loops,
                degree_norm=self.connector.degree_norm,
                adj_transpose=self.connector.adj_transpose,
                edge_weight_norm=self.connector.edge_weight_norm,
            )

            out = PoolingOutput(x=x_pooled, edge_index=adj_pool, so=so, loss=loss)

            return out

    def compute_loss(
        self, adj: Tensor, S: Tensor, adj_pool: Tensor, mask: Optional[Tensor] = None
    ) -> Optional[dict]:
        r"""Computes the auxiliary loss terms.

        Args:
            adj (~torch.Tensor): The dense adjacency matrix.
            S (~torch.Tensor): The dense assignment matrix.
            adj_pool (~torch.Tensor): The pooled adjacency matrix.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        Returns:
            dict: A dictionary with the different terms of
            the auxiliary loss.
        """
        # Motif adj matrix - not sym. normalised
        motif_adj = torch.matmul(torch.matmul(adj, adj), adj)
        motif_adj_pool = torch.matmul(torch.matmul(S.transpose(1, 2), motif_adj), S)

        cut_loss = ho_cut_loss = 0
        # 1st order MinCUT loss
        if self.alpha < 1:
            cut_loss = mincut_loss(adj, S, adj_pool)
            cut_loss = 1 / self.k * cut_loss

        # Higher order cut
        if self.alpha > 0:
            ho_cut_loss = mincut_loss(motif_adj, S, motif_adj_pool)
            ho_cut_loss = 1 / self.k * ho_cut_loss

        # Combine ho and fo mincut loss.
        hosc_loss = (1 - self.alpha) * cut_loss + self.alpha * ho_cut_loss

        # Orthogonality loss
        if self.mu == 0:
            ortho_loss = torch.tensor(0)
        elif self.hosc_ortho:
            # Hosc orthogonality regularization
            ortho_loss = hosc_orthogonality_loss(S, mask, batch_reduction="mean")
        else:
            # Standard orthogonality regularization of MinCutPool
            ortho_loss = orthogonality_loss(S, batch_reduction="mean")

        return {"hosc_loss": hosc_loss, "ortho_loss": self.mu * ortho_loss}

    def extra_repr_args(self) -> dict:
        return {
            "mu": self.mu,
            "alpha": self.alpha,
            "hosc_ortho": self.hosc_ortho,
        }
