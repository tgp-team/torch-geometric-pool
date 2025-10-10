from typing import Callable, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.nn import Linear
from torch_geometric.nn.conv import LEConv
from torch_geometric.typing import Adj
from torch_geometric.utils import scatter, softmax
from torch_sparse import SparseTensor

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import SelectOutput, TopkSelect
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils import add_remaining_self_loops, check_and_filter_edge_weights
from tgp.utils.typing import LiftType, ReduceType, SinvType


class ASAPooling(SRCPooling):
    r"""The Adaptive Structure Aware Pooling operator from the paper
    `"ASAP: Adaptive Structure Aware Pooling for Learning Hierarchical
    Graph Representations" <https://arxiv.org/abs/1911.07979>`_ (Ranjan et al., AAAI 2020).

    + The :math:`\texttt{select}` operator is implemented by passing a special score
      to :class:`~tgp.select.TopkSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        in_channels (int): Size of each input sample.
        ratio (float or int): Graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`. (default: :obj:`0.5`)
        GNN (~torch.nn.Module, optional): A graph neural network layer for
            using intra-cluster properties.
            Especially helpful for graphs with higher degree of neighborhood
            (one of :class:`~torch_geometric.nn.conv.GraphConv`,
            :class:`~torch_geometric.nn.conv.GCNConv` or
            any GNN which supports the :obj:`edge_weight` parameter).
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability of the normalized
            attention coefficients which exposes each node to a stochastically
            sampled neighborhood during training. (default: :obj:`0`)
        negative_slope (float, optional): LeakyReLU angle of the negative
            slope. (default: :obj:`0.2`)
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
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
        lift_red_op (~tgp.typing.ReduceType, optional):
            The aggregation function to be applied to the lifted node features.
            Can be any string of class :class:`~tgp.utils.typing.ReduceType` admitted by
            :obj:`~torch_geometric.utils.scatter`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the adjacency matrix.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`False`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
        **kwargs (optional): Additional parameters for initializing the
            graph neural network layer.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        GNN: Optional[torch.nn.Module] = None,
        dropout: float = 0.0,
        negative_slope: float = 0.2,
        add_self_loops: bool = False,
        nonlinearity: Union[str, Callable] = "sigmoid",
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ReduceType = "sum",
        lift_red_op: ReduceType = "sum",
        remove_self_loops: bool = True,
        degree_norm: bool = False,
        edge_weight_norm: bool = False,
        **kwargs,
    ):
        if remove_self_loops and add_self_loops:
            raise ValueError("remove_self_loops and add_self_loops cannot be both True")

        super().__init__(
            selector=TopkSelect(ratio=ratio, act=nonlinearity, s_inv_op=s_inv_op),
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
        self.negative_slope = negative_slope
        self.dropout = dropout
        self.GNN = GNN
        self.select_scorer = LEConv(in_channels, 1)
        self.add_self_loops = add_self_loops

        self.lin = Linear(in_channels, in_channels)
        self.att = Linear(2 * in_channels, 1)
        if self.GNN is not None:
            # keep only the kwargs that are used in the GNN
            kwargs = {
                k: v
                for k, v in kwargs.items()
                if k in GNN.__init__.__code__.co_varnames
            }
            self.gnn_intra_cluster = GNN(self.in_channels, self.in_channels, **kwargs)
        else:
            self.gnn_intra_cluster = None

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        self.lin.reset_parameters()
        self.att.reset_parameters()
        if self.gnn_intra_cluster is not None:
            self.gnn_intra_cluster.reset_parameters()
        super().reset_parameters()

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
        r"""The forward pass of the pooling operator.

        Args:
            x (~torch.Tensor):
                The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (~torch_geometric.typing.Adj, optional):
                The connectivity matrix.
                It can either be a :class:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape  :math:`[E]` or :math:`[E, 1]`
                containing the weights of the edges.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional):
                The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            lifting (bool, optional):
                If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            N = x.size(0)
            x = x.unsqueeze(-1) if x.dim() == 1 else x

            edge_weight = check_and_filter_edge_weights(edge_weight)
            adj, edge_weight = add_remaining_self_loops(
                adj, edge_weight, fill_value=1.0, num_nodes=N
            )

            x_pool = x
            if self.gnn_intra_cluster is not None:
                x_pool = self.gnn_intra_cluster(
                    x=x, edge_index=adj, edge_weight=edge_weight
                )

            # Convert to edge_index if needed
            if isinstance(adj, SparseTensor):
                row, col, edge_weight = adj.coo()
                edge_index = torch.stack([row, col])
            else:
                edge_index = adj

            if batch is None:
                batch = edge_index.new_zeros(x.size(0))

            x_pool_j = x_pool[edge_index[0]]
            x_q = scatter(x_pool_j, edge_index[1], dim=0, reduce="max")
            x_q = self.lin(x_q)[edge_index[1]]

            score = self.att(torch.cat([x_q, x_pool_j], dim=-1)).view(-1)
            score = F.leaky_relu(score, self.negative_slope)
            score = softmax(score, edge_index[1], num_nodes=N)

            # Sample attention coefficients stochastically.
            score = F.dropout(score, p=self.dropout, training=self.training)

            v_j = x[edge_index[0]] * score.view(-1, 1)
            x = scatter(v_j, edge_index[1], dim=0, reduce="sum")
            score = self.select_scorer(x, edge_index=adj, edge_weight=edge_weight)

            # Select
            so = self.select(x=score, batch=batch)

            # Reduce
            x, batch_pooled = self.reduce(x=x, so=so, batch=batch)

            # Connect
            edge_index_pooled, pooled_edge_weight = self.connect(
                edge_index=adj,
                so=so,
                edge_weight=edge_weight,
                batch_pooled=batch_pooled,
            )

            out = PoolingOutput(
                x=x,
                edge_index=edge_index_pooled,
                edge_weight=pooled_edge_weight,
                batch=batch_pooled,
                so=so,
            )
            return out

    def extra_repr_args(self) -> dict:
        return {
            "ratio": self.ratio,
            "GNN": self.GNN.__class__.__name__ if self.GNN is not None else "None",
            "add_self_loops": self.add_self_loops,
        }
