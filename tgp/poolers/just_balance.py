from typing import List, Optional, Union

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import DenseConnect
from tgp.data import NormalizeAdj
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import MLPSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import just_balance_loss
from tgp.utils.typing import LiftType, SinvType


class JustBalancePooling(DenseSRCPooling):
    r"""The Just Balance pooling operator from the paper `"Simplifying Clustering with
    Graph Neural Networks" <https://arxiv.org/abs/2207.08779>`_ (Bianchi et al., NLDL 2023).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.MLPSelect`.
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
        cache_preprocessing (bool, optional):
            If :obj:`True`, caches the dense adjacency produced during preprocessing.
            This should only be enabled when the same graph is reused across iterations.
            (default: :obj:`False`)
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
        batched: bool = True,
        sparse_output: bool = False,
        cache_preprocessing: bool = False,
    ):
        super().__init__(
            selector=MLPSelect(
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
                sparse_output=sparse_output,
            ),
            adj_transpose=adj_transpose,
            cache_preprocessing=cache_preprocessing,
            batched=batched,
            sparse_output=sparse_output,
        )

        self.normalize_loss = normalize_loss
        self.loss_coeff = loss_coeff

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
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
                the valid nodes in each graph. Only used when inputs are already
                dense/padded. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(
                x_pool=x, so=so, batch=batch, batch_pooled=batch_pooled
            )
            return x_lifted

        if not self.batched:
            raise NotImplementedError(
                "JustBalance unbatched mode is not implemented yet."
            )

        x, adj, mask = self._ensure_batched_inputs(
            x=x,
            edge_index=adj,
            edge_weight=edge_weight,
            batch=batch,
            mask=mask,
        )

        # Select
        so = self.select(x=x, mask=mask)

        # Reduce
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect
        adj_pooled, _ = self.connect(
            edge_index=adj,
            so=so,
            edge_weight=edge_weight,
            batch=batch,
            batch_pooled=batch_pooled,
        )

        loss = self.compute_loss(so.s, mask, so.num_nodes, so.num_supernodes)

        if self.sparse_output:
            x_pooled, edge_index_pooled, edge_weight_pooled, batch_pooled = (
                self._finalize_sparse_output(
                    x_pool=x_pooled,
                    adj_pool=adj_pooled,
                    batch=batch,
                    batch_pooled=batch_pooled,
                    so=so,
                )
            )
            return PoolingOutput(
                x=x_pooled,
                edge_index=edge_index_pooled,
                edge_weight=edge_weight_pooled,
                batch=batch_pooled,
                so=so,
                loss=loss,
            )

        return PoolingOutput(x=x_pooled, edge_index=adj_pooled, so=so, loss=loss)

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
            "batched": self.batched,
            "sparse_output": self.sparse_output,
            "loss_coeff": self.loss_coeff,
            "normalize_loss": self.normalize_loss,
        }
