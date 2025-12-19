from typing import Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import MaxCutSelect, SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils.losses import maxcut_loss
from tgp.utils.ops import connectivity_to_edge_index
from tgp.utils.typing import ConnectionType, LiftType, ReduceType, SinvType


class MaxCutPooling(SRCPooling):
    r"""The MaxCut pooling operator from the paper
    `"MaxCutPool: differentiable feature-aware Maxcut for pooling in graph neural networks"
    <https://arxiv.org/abs/2409.05100>`_ (Abate & Bianchi, ICLR 2025).

    This pooling layer uses a differentiable MaxCut objective to learn
    node assignments. It is particularly effective for heterophilic graphs
    and provides robust pooling through graph topology-aware scoring.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.MaxCutSelect`,
      which computes MaxCut-aware node scores and performs top-k selection.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer provides one auxiliary loss:

    + the MaxCut loss (:class:`~tgp.utils.losses.maxcut_loss`).

    Args:
        in_channels (int): Size of each input sample.
        ratio (Union[float, int]): Graph pooling ratio for top-k selection.
            (default: :obj:`0.5`)
        assign_all_nodes (bool, optional): Whether to create assignment matrices that map
            all nodes to the closest supernode (True) or perform standard top-k selection (False).
            (default: :obj:`True`)
        max_iter (int, optional): Maximum distance for the closest node assignment.
            (default: :obj:`5`)
        loss_coeff (float, optional): Coefficient for the MaxCut auxiliary loss.
            (default: :obj:`1.0`)
        mp_units (list, optional): List of hidden units for message passing layers.
            (default: :obj:`[32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8]`)
        mp_act (str, optional): Activation function for message passing layers.
            (default: :obj:`"tanh"`)
        mlp_units (list, optional): List of hidden units for MLP layers.
            (default: :obj:`[16, 16]`)
        mlp_act (str, optional): Activation function for MLP layers.
            (default: :obj:`"relu"`)
        act (str, optional): Activation function for the final score.
            (default: :obj:`"tanh"`)
        delta (float, optional): Delta parameter for propagation matrix computation.
            (default: :obj:`2.0`)
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
            (default: :obj:`"sum"`)
        connect_red_op (~tgp.utils.typing.ConnectionType, optional):
            The aggregation function to be applied to all edges connecting nodes assigned
            to supernodes :math:`i` and :math:`j`.
            Can be any string of class :class:`~tgp.utils.typing.ConnectionType` admitted by
            :obj:`~torch_geometric.utils.coalesce`,
            e.g., :obj:`'sum'`, :obj:`'mean'`, :obj:`'max'`)
            (default: :obj:`"sum"`)
        lift_red_op (~tgp.utils.typing.ReduceType, optional):
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
            (default: :obj:`True`)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        assign_all_nodes: bool = True,
        max_iter: int = 5,
        loss_coeff: float = 1.0,
        mp_units: list = [32, 32, 32, 32],
        mp_act: str = "tanh",
        mlp_units: list = [16, 16],
        mlp_act: str = "relu",
        act: str = "tanh",
        delta: float = 2.0,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
        remove_self_loops: bool = True,
        degree_norm: bool = False,
        edge_weight_norm: bool = True,
    ):
        super().__init__(
            selector=MaxCutSelect(
                in_channels=in_channels,
                ratio=ratio,
                assign_all_nodes=assign_all_nodes,
                max_iter=max_iter,
                mp_units=mp_units,
                mp_act=mp_act,
                mlp_units=mlp_units,
                mlp_act=mlp_act,
                act=act,
                delta=delta,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            connector=SparseConnect(
                reduce_op=connect_red_op,
                edge_weight_norm=edge_weight_norm,
                degree_norm=degree_norm,
                remove_self_loops=remove_self_loops,
            ),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
        )

        self.in_channels = in_channels
        self.ratio = ratio
        self.assign_all_nodes = assign_all_nodes
        self.max_iter = max_iter
        self.loss_coeff = loss_coeff
        self.mp_units = mp_units
        self.mp_act = mp_act
        self.mlp_units = mlp_units
        self.mlp_act = mlp_act
        self.act = act
        self.delta = delta

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
        r"""Forward pass of the MaxCut pooling operator.

        Args:
            x (~torch.Tensor): Node features of shape :math:`(N, F)`.
            adj (~torch_geometric.typing.Adj, optional): Graph connectivity.
                Can be edge_index tensor of shape :math:`(2, E)` or SparseTensor.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the select operator.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): Batch assignments of shape :math:`(N,)`.
                (default: :obj:`None`)
            lifting (bool, optional): If :obj:`True`, perform lift operation.
                (default: :obj:`False`)

        Returns:
            PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            if so is None:
                raise ValueError("SelectOutput (so) cannot be None for lifting")
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        # Select
        so = self.select(x=x, edge_index=adj, edge_weight=edge_weight, batch=batch)
        loss = self.compute_loss(so.scores, adj, edge_weight, batch)

        # Reduce
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect (it is always based on the full assignment)
        if not self.assign_all_nodes:
            full_so = so.assign_all_nodes(
                adj=adj,
                weight=None,
                max_iter=self.max_iter,
                batch=batch,
                closest_node_assignment=True,
            )
        else:
            full_so = so
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=adj,
            so=full_so,
            edge_weight=edge_weight,
            batch_pooled=batch_pooled,
        )

        return PoolingOutput(
            x=x_pooled,
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
            loss=loss,
        )

    def compute_loss(
        self,
        scores: Tensor,
        adj: Adj,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
    ) -> dict:
        """Compute the auxiliary MaxCut loss.

        Args:
            scores (~torch.Tensor): Node scores computed by the MaxCut selector.
            adj (~torch_geometric.typing.Adj): Graph connectivity.
                Can be edge_index tensor of shape :math:`(2, E)` or SparseTensor.
            edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): Batch assignments of shape :math:`(N,)`.
                (default: :obj:`None`)

        Returns:
            dict: A dictionary with the MaxCut loss term.
        """
        edge_index, edge_weight = connectivity_to_edge_index(adj, edge_weight)

        # Compute MaxCut loss
        maxcut_loss_val = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            batch_reduction="mean",
        )

        return {"maxcut_loss": maxcut_loss_val * self.loss_coeff}

    @property
    def has_loss(self) -> bool:
        """Returns True if this pooler computes auxiliary losses."""
        return True

    def extra_repr_args(self) -> dict:
        """Additional representation arguments for debugging."""
        return {
            "loss_coeff": self.loss_coeff,
        }
