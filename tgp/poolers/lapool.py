from typing import Optional

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import DenseConnectUnbatched
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import LaPoolSelect, SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils.typing import LiftType, ReduceType, SinvType


class LaPooling(SRCPooling):
    r"""The LaPool pooling operator from the paper `Towards Interpretable Sparse Graph Representation Learning
    with Laplacian Pooling <https://arxiv.org/abs/1905.11577>`_ (Noutahi et al., 2019).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.LaPoolSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnectUnbatched`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        shortest_path_reg (bool, optional):
            If :obj:`True`, applies the shortest path
            regularization to the selection matrix (this can be expensive since it
            runs on CPU).
            (default: :obj:`False`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, normalize the pooled adjacency matrix by the
            nodes' degree.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
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
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
        lift_red_op (~tgp.typing.ReduceType, optional):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
    """

    def __init__(
        self,
        shortest_path_reg: bool = False,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        lift_red_op: ReduceType = "sum",
    ):
        super().__init__(
            selector=LaPoolSelect(
                shortest_path_reg=shortest_path_reg, s_inv_op=s_inv_op
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=DenseConnectUnbatched(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                edge_weight_norm=edge_weight_norm,
            ),
        )

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape  :math:`[E]` or :math:`[E, 1]`
                containing the weights of the edges.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            batch (torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            batch_pooled (torch.Tensor, optional): The batch vector for the pooled nodes.
                Required when lifting with dense :math:`[N, K]` SelectOutput on multi-graph
                batches. Pass `out.batch` from the pooling call. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            batch_orig = batch if batch is not None else so.batch
            x_lifted = self.lift(
                x_pool=x, so=so, batch=batch_orig, batch_pooled=batch_pooled
            )
            return x_lifted

        else:
            # Select
            so = self.select(
                x=x,
                edge_index=adj,
                edge_weight=edge_weight,
                batch=batch,
                num_nodes=x.size(0),
            )
            # batch is now stored in SelectOutput by the selector

            # Reduce
            x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

            # Connect
            edge_index_pooled, edge_weight_pooled = self.connect(
                edge_index=adj,
                so=so,
                edge_weight=edge_weight,
                batch=batch,
                batch_pooled=batch_pooled,
            )

            out = PoolingOutput(
                x=x_pooled,
                edge_index=edge_index_pooled,
                edge_weight=edge_weight_pooled,
                batch=batch_pooled,
                so=so,
            )
            return out
