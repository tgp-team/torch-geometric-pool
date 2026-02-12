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

    def multi_level_precoarsening(
        self,
        levels: int,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> list[PoolingOutput]:
        """Compute multiple SEP pre-coarsening levels from a single tree hierarchy."""
        if levels < 1:
            raise ValueError(f"'levels' must be >= 1, got {levels}.")
        if edge_index is None:
            raise ValueError("edge_index cannot be None for pre-coarsening.")

        clear_cache = getattr(self, "clear_cache", None)
        if callable(clear_cache):
            clear_cache()

        if levels == 1:
            pooled_levels = [
                self.precoarsening(
                    edge_index=edge_index,
                    edge_weight=edge_weight,
                    batch=batch,
                    num_nodes=num_nodes,
                    **kwargs,
                )
            ]
            if callable(clear_cache):
                clear_cache()
            return pooled_levels

        so_levels = self.selector.multi_level_select(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            levels=levels,
            **kwargs,
        )
        if len(so_levels) != levels:
            raise RuntimeError(
                f"SEPSelect returned {len(so_levels)} levels, expected {levels}."
            )

        pooled_levels = []
        current_edge_index = edge_index
        current_edge_weight = edge_weight
        current_batch = batch
        current_num_nodes = num_nodes

        for so in so_levels:
            if current_num_nodes is not None and int(current_num_nodes) != int(
                so.num_nodes
            ):
                raise RuntimeError(
                    "Inconsistent hierarchy sizes in multi-level SEP pre-coarsening: "
                    f"expected {int(current_num_nodes)} nodes, got {int(so.num_nodes)}."
                )

            pooled = self._precoarsening_from_select_output(
                so=so,
                edge_index=current_edge_index,
                edge_weight=current_edge_weight,
                batch=current_batch,
                **kwargs,
            )
            pooled_levels.append(pooled)

            pooled_data = pooled.as_data()
            current_edge_index = pooled_data.edge_index
            current_edge_weight = pooled_data.edge_weight
            current_batch = pooled_data.batch
            current_num_nodes = pooled_data.num_nodes

        if callable(clear_cache):
            clear_cache()
        return pooled_levels

    def extra_repr_args(self) -> dict:
        return {"cached": self.cached}
