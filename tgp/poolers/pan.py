from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch.nn import Parameter
from torch_geometric.typing import SparseTensor
from torch_geometric.utils import scatter

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import SelectOutput, TopkSelect
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils.typing import LiftType, ReduceType, SinvType


class PANPooling(SRCPooling):
    r"""The path integral based pooling operator from the paper
    `"Path Integral Based Convolution and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2006.16811>`_ (Ma et al., NeurIPS 2020).

    PAN pooling performs top-:math:`k` pooling where global node importance is
    measured based on node features :math:`\mathbf{X}` and the MET matrix :math:`\mathbf{M}`:

    .. math::
        {\rm score} = \beta_1 \mathbf{X} \cdot \mathbf{p} + \beta_2
        {\rm deg}(\mathbf{M})

    The MET matrix must be computed by the
    :class:`~torch_geometric.nn.conv.PANConv` layer.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.TopkSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        in_channels (int):
            Size of each input sample.
        ratio (float):
            Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional):
            Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{s}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is ignored.
            (default: :obj:`None`)
        multiplier (float, optional):
            Coefficient by which features gets multiplied after pooling.
            This can be useful for large graphs and when :obj:`min_score` is used.
            (default: :obj:`1.0`)
        nonlinearity (str or callable, optional):
            The non-linearity to use when computing the score.
            (default: :obj:`"tanh"`)
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
        connect_red_op (~tgp.typing.ConnectionType, optional):
            The aggregation function to be applied to all edges connecting nodes assigned
            to supernodes :math:`i` and :math:`j`.
            Can be any string of class :class:`~tgp.utils.typing.ConnectionType` admitted by
            :obj:`~torch_geometric.utils.coalesce`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`) (default: :obj:`"sum"`)
        lift_red_op (~tgp.typing.ReduceType, optional):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`) (default: :obj:`"sum"`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the adjacency matrix.
            (default: :obj:`False`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`False`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: float = 0.5,
        min_score: Optional[float] = None,
        multiplier: float = 1.0,
        nonlinearity: Union[str, Callable] = "tanh",
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ReduceType = "sum",
        lift_red_op: ReduceType = "sum",
        remove_self_loops: bool = False,
        degree_norm: bool = False,
        edge_weight_norm: bool = False,
    ):
        super().__init__(
            selector=TopkSelect(
                ratio=ratio, min_score=min_score, act=nonlinearity, s_inv_op=s_inv_op
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=SparseConnect(
                remove_self_loops=remove_self_loops,
                reduce_op=connect_red_op,
                degree_norm=degree_norm,
                edge_weight_norm=edge_weight_norm,
            ),
        )

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        self.multiplier = multiplier

        self.p = Parameter(torch.empty(in_channels))
        self.beta = Parameter(torch.empty(2))

        self.reset_own_parameters()

    def reset_own_parameters(self):
        r"""Resets :math:`p` and :math:`\beta` learnable parameters."""
        self.p.data.fill_(1)
        self.beta.data.fill_(0.5)

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.reset_own_parameters()
        super().reset_parameters()

    def forward(
        self,
        x: Tensor,
        adj: Optional[SparseTensor] = None,
        so: Optional[SelectOutput] = None,
        batch: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (SparseTensor):
                The MET matrix :math:`\mathbf{M}` from the
                :class:`~torch_geometric.nn.conv.PANConv` layer.
                It has a (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            _, col, edge_weight = adj.coo()
            assert edge_weight is not None

            # Select
            score1 = (x * self.p).sum(dim=-1)
            score2 = scatter(edge_weight, col, 0, dim_size=x.size(0), reduce="sum")
            score = (self.beta[0] * score1 + self.beta[1] * score2).view(-1, 1)
            so = self.select(x=score, batch=batch)

            # Reduce
            x, batch_pooled = self.reduce(x=x, so=so, batch=batch)
            x = self.multiplier * x if self.multiplier != 1 else x

            # Connect
            adj_pool, _ = self.connect(edge_index=adj, so=so, batch_pooled=batch_pooled)

            out = PoolingOutput(
                x=x, edge_index=adj_pool, edge_weight=None, batch=batch_pooled, so=so
            )
            return out

    def extra_repr_args(self) -> dict:
        return {
            "multiplier": self.multiplier,
        }
