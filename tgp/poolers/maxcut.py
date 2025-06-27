from typing import Optional, Union

import torch
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
        assignment_mode (bool, optional): Whether to create assignment matrices that map
            ALL nodes to supernodes (True) or perform standard top-k selection (False).
            When True, mimics the original MaxCutPool "expressive" mode.
            (default: :obj:`True`)
        loss_coeff (float): Coefficient for the MaxCut auxiliary loss.
            (default: :obj:`1.0`)
        mp_units (list, optional): List of hidden units for message passing layers.
            (default: :obj:`[32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8]`)
        mp_act (str, optional): Activation function for message passing layers.
            (default: :obj:`"tanh"`)
        mlp_units (list, optional): List of hidden units for MLP layers.
            (default: :obj:`[16, 16]`)
        mlp_act (str, optional): Activation function for MLP layers.
            (default: :obj:`"relu"`)
        delta (float, optional): Delta parameter for propagation matrix computation.
            (default: :obj:`2.0`)
        lift (LiftType): Matrix operation for lifting pooled features.
            (default: :obj:`"precomputed"`)
        s_inv_op (SinvType): Operation for computing :math:`\mathbf{S}^{-1}`.
            (default: :obj:`"transpose"`)
        reduce_red_op (ReduceType): Aggregation function for node reduction.
            (default: :obj:`"sum"`)
        connect_red_op (ReduceType): Aggregation function for edge connection.
            (default: :obj:`"sum"`)
        lift_red_op (ReduceType): Aggregation function for lifting operation.
            (default: :obj:`"sum"`)

    Example:
        >>> pooler = MaxCutPooling(in_channels=64, ratio=0.5, assignment_mode=True, loss_coeff=1.0)
        >>> x = torch.randn(100, 64)  # 100 nodes, 64 features
        >>> edge_index = torch.randint(0, 100, (2, 200))  # 200 edges
        >>> out = pooler(x=x, edge_index=edge_index)
        >>> print(out.x.shape, out.edge_index.shape)
        >>> print(f"MaxCut loss: {out.get_loss_value('maxcut_loss')}")
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[float, int] = 0.5,
        assign_all_nodes: bool = True,
        loss_coeff: float = 1.0,
        mp_units: list = [32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8],
        mp_act: str = "tanh",
        mlp_units: list = [16, 16],
        mlp_act: str = "relu",
        delta: float = 2.0,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ConnectionType = "sum",
        lift_red_op: ReduceType = "sum",
    ):
        super().__init__(
            selector=MaxCutSelect(
                in_channels=in_channels,
                ratio=ratio,
                assign_all_nodes=assign_all_nodes,
                mp_units=mp_units,
                mp_act=mp_act,
                mlp_units=mlp_units,
                mlp_act=mlp_act,
                delta=delta,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            connector=SparseConnect(reduce_op=connect_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
        )

        self.in_channels = in_channels
        self.ratio = ratio
        self.assign_all_nodes = assign_all_nodes
        self.loss_coeff = loss_coeff
        self.mp_units = mp_units
        self.mp_act = mp_act
        self.mlp_units = mlp_units
        self.mlp_act = mlp_act
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
            return x_lifted  # type: ignore

        # Select phase
        so = self.select(x=x, edge_index=adj, edge_weight=edge_weight, batch=batch)

        if adj is None:
            # If no adjacency matrix is provided, we can't compute the MaxCut loss
            # Set loss to 0 and continue with the pooling operation
            loss = {"maxcut_loss": torch.tensor(0.0, device=x.device)}
        else:
            # Compute auxiliary loss - check if scores exist in SelectOutput
            scores = getattr(so, 'scores', None)
            if scores is not None:
                loss = self.compute_loss(scores, adj, edge_weight, batch)
            else:
                # Fallback: set loss to 0 if scores are not available
                loss = {"maxcut_loss": torch.tensor(0.0, device=x.device)}

        # Reduce phase
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect phase  
        # Note: adj cannot be None here because selection phase would have failed first
        assert adj is not None  # for type checker
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=adj, so=so, edge_weight=edge_weight
        )

        return PoolingOutput(
            x=x_pooled,
            edge_index=edge_index_pooled,  # type: ignore
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
            loss=loss,
        )

    def compute_loss(self, scores: Tensor, adj: Adj, edge_weight: Optional[Tensor] = None, batch: Optional[Tensor] = None) -> dict:
        """Compute the auxiliary MaxCut loss.

        Args:
            so (~tgp.select.SelectOutput): The output of the select operator,
                which should contain scores, edge_index, and edge_weight.
            batch (~torch.Tensor, optional): Batch assignments.

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
            "in_channels": self.in_channels,
            "ratio": self.ratio,
            "assign_all_nodes": self.assign_all_nodes,
            "loss_coeff": self.loss_coeff,
            "mp_units": self.mp_units,
            "mp_act": self.mp_act,
            "mlp_units": self.mlp_units,
            "mlp_act": self.mlp_act,
            "delta": self.delta,
        } 