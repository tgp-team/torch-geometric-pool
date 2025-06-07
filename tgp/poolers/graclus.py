from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import GraclusSelect, SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils.typing import ConnectionType, LiftType, ReduceType, SinvType


class GraclusPooling(SRCPooling):
    r"""The Graclus pooling operator inspired by the paper `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <https://ieeexplore.ieee.org/document/4302760>`_
    (Dhillon et al., TPAMI 2007).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.GraclusSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        lift (~tgp.utils.typing.LiftType, optional):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - :obj:`"precomputed"` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the :obj:`"s_inv"` attribute of the :class:`~tgp.select.SelectOutput`.
            - :obj:`"transpose"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.

        s_inv_op (~tgp.utils.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.

        reduce_red_op (~tgp.utils.typing.ReduceType, optional):
            The aggregation function to be applied to nodes in the same cluster. Can be
            any string admitted by :obj:`~torch_geometric.utils.scatter` (e.g., :obj:`'sum'`, :obj:`'mean'`,
            :obj:`'max'`) or any :class:`~tgp.utils.typing.ReduceType`.
            (default: :obj:`sum`)
        lift_red_op (~tgp.typing.ReduceType, optional):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
        cached (bool, optional):
            If set to :obj:`True`, the output of the :math:`\texttt{select}` and :math:`\texttt{select}`
            operations will be cached, so that they do not need to be recomputed.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
        cached: bool = False,
    ):
        super().__init__(
            selector=GraclusSelect(s_inv_op=s_inv_op),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=SparseConnect(reduce_op=connect_red_op),
            cached=cached,
        )
        self.cached = cached

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        so: Optional[SelectOutput] = None,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (torch.Tensor): The node feature matrix.
            adj (torch.Tensor): The edge indices.
            so (SelectOutput, optional): The output of the select operator.
                (default: :obj:`None`)
            edge_weight (torch.Tensor, optional): The edge weights.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector.
                (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the lifting operation will be applied.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The pooled node features and the pooled edge indices
            and edge attributes.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            # Select
            if isinstance(adj, SparseTensor):
                row, col, edge_weight = adj.coo()
                edge_index = torch.stack([row, col])
            else:
                edge_index = adj
            so = self.select(
                edge_index=edge_index, edge_weight=edge_weight, num_nodes=x.size(0)
            )

            # Reduce
            x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

            # Connect
            edge_index_pooled, edge_weight_pooled = self.connect(
                edge_index=adj, so=so, edge_weight=edge_weight
            )

            out = PoolingOutput(
                x=x_pooled,
                edge_index=edge_index_pooled,
                edge_weight=edge_weight_pooled,
                batch=batch_pooled,
                so=so,
            )
            return out

    def extra_repr_args(self) -> dict:
        return {
            "cached": self.cached,
        }
