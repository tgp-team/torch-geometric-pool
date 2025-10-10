from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from tgp.select.base_select import SelectOutput
from tgp.select.topk_select import TopkSelect
from tgp.utils.ops import (
    check_and_filter_edge_weights,
    connectivity_to_edge_index,
    delta_gcn_matrix,
)
from tgp.utils.typing import SinvType


class MaxCutScoreNet(torch.nn.Module):
    r"""Score network for MaxCut pooling that computes node-level scores
    using graph convolutions followed by MLP layers.

    Args:
        in_channels (int): Size of each input feature.
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
    """

    def __init__(
        self,
        in_channels: int,
        mp_units: list = [32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8],
        mp_act: str = "tanh",
        mlp_units: list = [16, 16],
        mlp_act: str = "relu",
        act: str = "tanh",
        delta: float = 2.0,
        **kwargs,  # Accept and ignore extra kwargs for compatibility
    ):
        super().__init__()

        self.initial_layer = Linear(in_channels, in_channels)  # initial embedding

        # Message passing layers
        if mp_act.lower() in ["identity", "none"]:
            self.mp_act = lambda x: x
        else:
            self.mp_act = activation_resolver(mp_act)

        self.mp_convs = torch.nn.ModuleList()
        in_units = in_channels
        for out_units in mp_units:
            self.mp_convs.append(
                GCNConv(in_units, out_units, normalize=False, cached=False)
            )
            in_units = out_units

        # MLP layers
        if mlp_act.lower() in ["identity", "none"]:
            self.mlp_act = lambda x: x
        else:
            self.mlp_act = activation_resolver(mlp_act)

        self.mlp = torch.nn.ModuleList()
        for out_units in mlp_units:
            self.mlp.append(Linear(in_units, out_units))
            in_units = out_units

        self.final_layer = Linear(in_units, 1)
        if act.lower() in ["identity", "none"]:
            self.act = lambda x: x
        else:
            self.act = activation_resolver(act)
        self.delta = delta

    def reset_parameters(self):
        """Reset parameters of all layers."""
        for conv in self.mp_convs:
            conv.reset_parameters()
        for layer in self.mlp:
            layer.reset_parameters()
        self.final_layer.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
    ) -> Tensor:
        r"""Compute MaxCut scores for each node.

        Args:
            x (~torch.Tensor): Node features of shape :math:`(N, F)`.
            edge_index (~torch.Tensor): Graph connectivity in COO format of shape :math:`(2, E)`.
            edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
                (default: :obj:`None`)

        Returns:
            ~torch.Tensor: Node scores of shape :math:`(N, 1)`, normalized to :math:`[-1, 1]` via tanh.
        """
        # Get Delta-GCN propagation matrix for heterophilic message passing
        edge_index, edge_weight = delta_gcn_matrix(
            edge_index, edge_weight, delta=self.delta
        )

        x = self.initial_layer(x)  # initial embedding

        # Message passing layers
        for mp_conv in self.mp_convs:
            x = mp_conv(x, edge_index, edge_weight)
            x = self.mp_act(x)

        # MLP layers
        for mlp_layer in self.mlp:
            x = mlp_layer(x)
            x = self.mlp_act(x)

        # Final score computation
        score = self.final_layer(x)
        return self.act(score)


class MaxCutSelect(TopkSelect):
    r"""The MaxCut :math:`\texttt{select}` operator from the paper
    `"MaxCutPool: differentiable feature-aware Maxcut for pooling in graph neural networks"
    <https://arxiv.org/abs/2409.05100>`_ (Abate & Bianchi, ICLR 2025).

    This operator computes node scores using a specialized neural network that optimizes
    the MaxCut objective, then performs top-k selection based on these scores.

    The MaxCut scoring process consists of:

    1. **Score Computation**: A graph neural network computes node-level scores
       :math:`\mathbf{s} \in [-1, 1]^N` via:

       .. math::
           \mathbf{s} = \tanh(\text{MLP}(\text{GNN}(\mathbf{X}, \mathbf{A})))

    2. **Top-k Selection**: Select top-k nodes based on scores:

       .. math::
           \mathbf{i} = \text{top}_k(|\mathbf{s}|)

    The computed scores are stored in the :class:`~tgp.select.SelectOutput` and can be
    accessed by the pooler for loss computation.

    Args:
        in_channels (int): Size of each input feature.
        ratio (Union[int, float]): Graph pooling ratio for top-k selection.
            (default: :obj:`0.5`)
        assign_all_nodes (bool, optional): Whether to create assignment matrices that map
            all nodes to the closest supernode (True) or perform standard top-k selection (False).
            (default: :obj:`True`)
        max_iter (int, optional): Maximum distance for the closest node assignment.
            (default: :obj:`5`)
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
        min_score (float, optional): Minimal node score threshold.
            Inherited from TopkSelect. (default: :obj:`None`)
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        assign_all_nodes: bool = True,
        max_iter: int = 5,
        mp_units: list = [32, 32, 32, 32, 16, 16, 16, 16, 8, 8, 8, 8],
        mp_act: str = "tanh",
        mlp_units: list = [16, 16],
        mlp_act: str = "relu",
        act: str = "tanh",
        delta: float = 2.0,
        min_score: Optional[float] = None,
        s_inv_op: SinvType = "transpose",
    ):
        # Initialize TopkSelect with None in_channels since we'll compute scores ourselves
        super().__init__(
            in_channels=None,  # We'll provide scores directly
            ratio=ratio,
            min_score=min_score,
            act="identity",  # No additional activation on scores
            s_inv_op=s_inv_op,
        )

        self.in_channels = in_channels
        self.mp_units = mp_units
        self.mp_act = mp_act
        self.mlp_units = mlp_units
        self.mlp_act = mlp_act
        self.score_act = act
        self.delta = delta
        self.assign_all_nodes = assign_all_nodes
        self.max_iter = max_iter

        # Score network - initialize after calling super().__init__
        self.score_net = MaxCutScoreNet(
            in_channels=in_channels,
            mp_units=mp_units,
            mp_act=mp_act,
            mlp_units=mlp_units,
            mlp_act=mlp_act,
            act=self.score_act,
            delta=delta,
        )

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        # Call parent reset_parameters (which handles the weight parameter if needed)
        super().reset_parameters()

        # Reset score network parameters (only if score_net exists)
        if hasattr(self, "score_net") and hasattr(self.score_net, "reset_parameters"):
            self.score_net.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass of the MaxCut selector.

        Args:
            x (~torch.Tensor): Node features of shape :math:`(N, F)`.
            edge_index (~torch.Tensor): Graph connectivity in COO format of shape :math:`(2, E)`.
            edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): Batch assignments of shape :math:`(N,)`.
                (default: :obj:`None`)

        Returns:
            SelectOutput: Selection output containing node indices, weights, and scores.
        """
        if edge_index is None:
            edge_index = torch.tensor([[], []], dtype=torch.long)
            edge_weight = None

        # Convert SparseTensor to edge_index format if needed
        if isinstance(edge_index, SparseTensor):
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )
        if edge_weight is not None:
            edge_weight = check_and_filter_edge_weights(edge_weight)

        scores = self.score_net(x, edge_index, edge_weight)  # Shape: (N, 1)

        # Perform top-k selection using computed scores - call parent forward
        topk_select_output = super().forward(x=scores, batch=batch)

        if self.assign_all_nodes:
            select_output = topk_select_output.assign_all_nodes(
                adj=edge_index,
                weight=scores.squeeze(-1),
                max_iter=self.max_iter,
                batch=batch,
                closest_node_assignment=True,
            )
        else:
            select_output = topk_select_output

        # Add scores to the select output to be used in the loss computation
        setattr(select_output, "scores", scores.squeeze(-1))
        select_output._extra_args.add("scores")

        return select_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"ratio={self.ratio}, "
            f"assign_all_nodes={self.assign_all_nodes}, "
            f"mp_units={self.mp_units}, "
            f"mp_act='{self.mp_act}', "
            f"mlp_units={self.mlp_units}, "
            f"mlp_act='{self.mlp_act}', "
            f"act='{self.score_act}', "
            f"delta={self.delta}, "
            f"max_iter={self.max_iter})"
        )
