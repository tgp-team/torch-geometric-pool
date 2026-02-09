from typing import Optional

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import SelectOutput, SEPSelect
from tgp.src import BasePrecoarseningMixin, PoolingOutput, SRCPooling
from tgp.utils.typing import ConnectionType, LiftType, ReduceType, SinvType


class SEPPooling(BasePrecoarseningMixin, SRCPooling):
    r"""The SEPPooling operator from the paper
    "Structural Entropy Guided Graph Hierarchical Pooling"
    <https://proceedings.mlr.press/v162/wu22b/wu22b.pdf>`_ (Wu et al., ICML 2022).

    SEP performs graph pooling by **optimizing cluster assignments globally**, in a single
    shot, with the goal of minimizing **structural entropy** over the whole graph.

    WARNING: This pooling operator works only if it is the only pooling operator in the pipeline.

    Args:
        cached (bool, optional):
            If :obj:`True`, cache :class:`~tgp.select.SelectOutput`. (default: :obj:`False`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops after coarsening. (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, symmetrically normalize pooled adjacency. (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize pooled edge weights. (default: :obj:`False`)
        lift (~tgp.utils.typing.LiftType, optional):
            Kept for API compatibility. EigenPooling always uses eigenvector-based
            lifting and ignores this option. (default: :obj:`"precomputed"`)
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            Operation used to compute :math:`\\mathbf{S}_\text{inv}` in
            :class:`~tgp.select.SelectOutput`. (default: :obj:`"transpose"`)
    """

    def __init__(
        self,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
        cached: bool = False,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
    ):
        super().__init__(
            selector=SEPSelect(s_inv_op=s_inv_op),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=SparseConnect(
                reduce_op=connect_red_op,
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                edge_weight_norm=edge_weight_norm,
            ),
            cached=cached,
        )

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
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
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        # Select (if not precomputed)
        if so is None:
            # Select
            so = self.select(
                edge_index=adj,
                edge_weight=edge_weight,
                batch=batch,
                num_nodes=x.size(0),
            )

        # Reduce
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=adj,
            so=so,
            edge_weight=edge_weight,
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

    def extra_repr_args(self) -> dict:
        # TODO: i am not sure  what we should put here
        return {"cached": self.cached}
