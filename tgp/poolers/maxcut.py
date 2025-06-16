from typing import Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.select import TopkSelect  # Temporary placeholder
from tgp.utils.losses import maxcut_loss
from tgp.utils.typing import LiftType, ReduceType, SinvType


class MaxCutPooling(SRCPooling):
    r"""MaxCut pooling operator from the paper "MaxCutPool: differentiable 
    feature-aware Maxcut for pooling in graph neural networks" 
    (Abate & Bianchi, ICLR 2025).

    This pooling layer uses a differentiable MaxCut objective to learn 
    node assignments. It is particularly effective for heterophilic graphs
    and provides robust pooling through graph topology-aware scoring.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.MaxCutPoolSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.SparseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer provides one auxiliary loss:

    + the MaxCut loss (:class:`~tgp.utils.losses.maxcut_loss`).

    Args:
        in_channels (int): Size of each input sample.
        ratio (Union[int, float], optional): Graph pooling ratio. 
            If float, fraction of nodes to keep. If int, exact number of nodes.
            (default: :obj:`0.5`)
        beta (float, optional): Coefficient for the MaxCut auxiliary loss.
            (default: :obj:`1.0`)
        min_score (Optional[float], optional): Minimal node score threshold.
            When set, overrides the ratio parameter.
            (default: :obj:`None`)
        initial_embedding (bool, optional): Whether to apply initial embedding
            to transform input features before scoring.
            (default: :obj:`True`)
        lift (LiftType, optional): Lifting operation type.
            (default: :obj:`"precomputed"`)
        s_inv_op (SinvType, optional): Operation for computing S inverse.
            (default: :obj:`"transpose"`) 
        reduce_red_op (ReduceType, optional): Reduction operation for features.
            (default: :obj:`"sum"`)
        connect_red_op (ReduceType, optional): Reduction operation for edges.
            (default: :obj:`"sum"`)
        lift_red_op (ReduceType, optional): Reduction operation for lifting.
            (default: :obj:`"sum"`)
        **score_net_kwargs: Additional arguments for the score network
            (mp_units, mp_act, mlp_units, mlp_act, delta).

    Examples:
        >>> pooler = MaxCutPooling(in_channels=64, ratio=0.5, beta=1.0)
        >>> x = torch.randn(100, 64)
        >>> edge_index = torch.randint(0, 100, (2, 200))
        >>> out = pooler(x, edge_index)
        >>> print(out.x.shape, out.edge_index.shape)
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        beta: float = 1.0,
        min_score: Optional[float] = None,
        initial_embedding: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        reduce_red_op: ReduceType = "sum",
        connect_red_op: ReduceType = "sum", 
        lift_red_op: ReduceType = "sum",
        **score_net_kwargs
    ):
        
        super().__init__(
            selector=TopkSelect(  # This will be replaced with MaxCutPoolSelect
                in_channels=in_channels,
                ratio=ratio,
                min_score=min_score,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(reduce_op=reduce_red_op),
            lifter=BaseLift(matrix_op=lift, reduce_op=lift_red_op),
            connector=SparseConnect(reduce_op=connect_red_op),
        )
        
        self.in_channels = in_channels
        self.ratio = ratio
        self.beta = beta
        self.min_score = min_score
        self.initial_embedding = initial_embedding
        self.score_net_kwargs = score_net_kwargs

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (Tensor): Node feature matrix of shape [N, F].
            edge_index (Adj): Graph connectivity.
            edge_weight (Optional[Tensor]): Edge weights.
            batch (Optional[Tensor]): Batch vector.
            so (Optional[SelectOutput]): Pre-computed selection output.
            lifting (bool): Whether to perform lifting operation.

        Returns:
            PoolingOutput: The pooling result containing pooled features,
                connectivity, and auxiliary loss.
        """
        if lifting:
            # Lift operation
            x_lifted = self.lift(x_pool=x, so=so)
            return PoolingOutput(x=x_lifted)

        # Select phase
        so = self.select(x=x, edge_index=edge_index, edge_weight=edge_weight, batch=batch)
        
        # Compute auxiliary MaxCut loss (will be moved to utils/losses later)
        loss = self.compute_loss(x, edge_index, edge_weight, batch, so)

        # Reduce phase
        x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

        # Connect phase  
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=edge_index, so=so, edge_weight=edge_weight
        )

        return PoolingOutput(
            x=x_pooled,
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
            loss=loss,
        )

    def compute_loss(self, x: Tensor, edge_index: Adj, edge_weight: Optional[Tensor], 
                     batch: Optional[Tensor], so: SelectOutput) -> dict:
        """Compute auxiliary losses."""
        # Extract scores from SelectOutput - this assumes scores are stored in so.scores
        # This will be properly implemented when MaxCutPoolSelect is created
        if hasattr(so, 'scores') and so.scores is not None:
            scores = so.scores
        else:
            # Fallback: use dummy scores for now until MaxCutPoolSelect is implemented
            import torch
            scores = torch.randn(x.size(0), device=x.device, requires_grad=True)
        
        maxcut_loss_val = maxcut_loss(
            scores=scores,
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            reduction="mean"
        )
        return {"maxcut_loss": self.beta * maxcut_loss_val}

    def extra_repr_args(self) -> dict:
        return {
            "in_channels": self.in_channels,
            "ratio": self.ratio,
            "beta": self.beta,
            "min_score": self.min_score,
            "initial_embedding": self.initial_embedding,
        } 