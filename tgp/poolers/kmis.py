from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select.kmis_select import KMISSelect, SelectOutput
from tgp.src import BasePrecoarseningMixin, PoolingOutput, SRCPooling
from tgp.utils.typing import ConnectionType, LiftType, ReduceType, SinvType


class KMISPooling(BasePrecoarseningMixin, SRCPooling):
    r"""The Maximal :math:`k`-Independent Set (:math:`k`-MIS) pooling operator from the
    paper `"Generalizing Downsampling from Regular Data to Graphs"
    <https://arxiv.org/abs/2208.03523>`_ (Bacciu et al., AAAI 2023).

    The :math:`k`-MIS pooling method selects a subset of nodes based on their score and
    a maximum independent set strategy. The pooling operates by first scoring nodes and
    then selecting a maximal independent set of nodes, where the score of each node is
    computed using one of the provided methods in the attribute :attr:`scorer`. The
    selected nodes are then pooled using the specified aggregation functions, with
    options to lift the node features using different matrix inversion strategies.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.KMISSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        in_channels (int, optional):
            Size of each input sample. Ignored if :obj:`scorer` is not
            :obj:`"linear"`. (default: :obj:`None`)
        order_k (int):
            The :math:`k`-th order for the independent set. (default: :obj:`1`)
        scorer (str or Callable):
            A function that computes a score for each node. Nodes with higher score
            have a higher chance of being selected for pooling. It can be one of:

            - :obj:`"linear"` (default): Uses a sigmoid-activated linear layer to
              compute the scores. :obj:`in_channels` and :obj:`score_passthrough`
              must be set when using this option.
            - :obj:`"random"`: Assigns a random score in :math:`[0, 1]` to each
              node.
            - :obj:`"constant"`: Assigns a constant score of :math:`1` to each node.
            - :obj:`"canonical"`: Assigns the score :math:`-i` to the :math:`i`-th
              node.
            - :obj:`"first"` (or :obj:`"last"`): Uses the first (or last) feature
              dimension of :math:`\mathbf{X}` as the node scores.
            - :obj:`"degree"`: Uses the degree of each node as the score.
            - A custom function: Accepts the arguments
              :obj:`(x, edge_index, edge_weight, batch)` and must return a
              one-dimensional :class:`~torch.FloatTensor`.
        score_heuristic (str, optional):
            Heuristic to increase the total score of selected nodes. Given an initial
            score vector :math:`\mathbf{s} \in \mathbb{R}^n`, options include:

            - :obj:`None`: No heuristic applied.
            - :obj:`"greedy"` (default): Computes the updated score
              :math:`\mathbf{s}'` as

              .. math::
                  \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} + \mathbf{I})^k
                  \mathbf{1}

              where :math:`\oslash` is element-wise division.
            - :obj:`"w-greedy"`: Computes the updated score :math:`\mathbf{s}'` as

              .. math::
                  \mathbf{s}' = \mathbf{s} \oslash (\mathbf{A} + \mathbf{I})^k
                  \mathbf{s}
        force_undirected (bool, optional):
            Whether to force the input graph to be undirected.
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
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`False`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
        cached (bool, optional):
            If set to :obj:`True`, the output of the :math:`\texttt{select}` and :math:`\texttt{select}`
            operations will be cached, so that they do not need to be recomputed.
            If :obj:`True`, the scorer cannot be :obj:`"linear"`.
            (default: :obj:`False`)
        node_dim (int, optional):
            The node dimension in the input feature matrix. (default: :obj:`-2`)
    """

    def __init__(
        self,
        in_channels: Optional[int] = None,
        order_k: int = 1,
        scorer: str = "linear",
        score_heuristic: Optional[str] = "greedy",
        force_undirected: bool = False,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: Optional[ReduceType] = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
        remove_self_loops: bool = True,
        degree_norm: bool = False,
        edge_weight_norm: bool = False,
        cached: bool = False,
        node_dim: int = -2,
    ):
        super().__init__(
            selector=KMISSelect(
                in_channels=in_channels,
                order_k=order_k,
                scorer=scorer,
                score_heuristic=score_heuristic,
                force_undirected=force_undirected,
                s_inv_op=s_inv_op,
                node_dim=node_dim,
            ),
            reducer=BaseReduce(
                reduce_op=reduce_red_op if reduce_red_op is not None else "sum"
            ),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=SparseConnect(
                reduce_op=connect_red_op,
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                edge_weight_norm=edge_weight_norm,
            ),
            cached=cached,
        )

        self.reduce_red_op = reduce_red_op
        self.cached = cached
        self.precoarsenable = scorer in ["random", "constant", "canonical", "degree"]

        if cached and scorer == "linear" or callable(scorer):
            raise Exception(
                "Caching should be disabled when using a linear scorer or a callable scorer."
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
            x (~torch.Tensor):
                The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                It can either be a :obj:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape
                :math:`[E]` containing the weights of the edges.
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

        else:
            # Select
            so = self.select(x=x, edge_index=adj, edge_weight=edge_weight, batch=batch)

            # Reduce
            if self.reduce_red_op is None:
                x_pooled = torch.index_select(x, index=so.mis, dim=self.node_dim)
                x_pooled = x_pooled * so.weight[so.mis].view(-1, 1)
                batch_pooled = None if batch is None else batch[so.mis]
            else:
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
        return {"cached": self.cached}
