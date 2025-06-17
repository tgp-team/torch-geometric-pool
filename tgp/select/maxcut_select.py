from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn import GCNConv, Linear
from torch_geometric.nn.resolver import activation_resolver
from torch_sparse import SparseTensor

from tgp.select.base_select import Select, SelectOutput
from tgp.select.topk_select import TopkSelect
from tgp.utils.ops import delta_gcn_matrix
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
        delta: float = 2.0,
        **kwargs,  # Accept and ignore extra kwargs for compatibility
    ):
        super().__init__()

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
        edge_index: Tensor,
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
        edge_index, edge_weight = delta_gcn_matrix(edge_index, edge_weight, delta=self.delta)

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
        return torch.tanh(score)


class MaxCutSelect(Select):
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
        score_net (Optional[Callable]): Custom score network. If :obj:`None`,
            uses :class:`MaxCutScoreNet`. (default: :obj:`None`)
        s_inv_op (SinvType): Operation for computing :math:`\mathbf{S}^{-1}`.
            (default: :obj:`"transpose"`)
        **score_net_kwargs: Additional arguments for :class:`MaxCutScoreNet`.
    """

    def __init__(
        self,
        in_channels: int,
        ratio: Union[int, float] = 0.5,
        score_net: Optional[Callable] = None,
        s_inv_op: SinvType = "transpose",
        **score_net_kwargs,
    ):
        super().__init__()

        self.in_channels = in_channels
        self.ratio = ratio

        # Score network
        if score_net is None:
            self.score_net = MaxCutScoreNet(
                in_channels=in_channels, **score_net_kwargs
            )
        else:
            self.score_net = score_net

        # TopK selector for actual selection
        self.topk_selector = TopkSelect(
            in_channels=None,  # We'll provide scores directly
            ratio=ratio,
            act="identity",  # No additional activation on scores
            s_inv_op=s_inv_op,
        )

    def reset_parameters(self):
        r"""Resets all learnable parameters of the module."""
        if hasattr(self.score_net, 'reset_parameters'):
            self.score_net.reset_parameters()
        self.topk_selector.reset_parameters()

    def forward(
        self,
        x: Tensor,
        edge_index: Union[Tensor, SparseTensor],
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
        # Convert SparseTensor to edge_index format if needed
        if isinstance(edge_index, SparseTensor):
            row, col, edge_weight = edge_index.coo()
            edge_index = torch.stack([row, col], dim=0)
        elif edge_index is None:
            raise ValueError("edge_index cannot be None for MaxCutSelect")
        
        # Compute MaxCut scores
        scores = self.score_net(x, edge_index, edge_weight)  # Shape: (N, 1)

        # Perform top-k selection using computed scores
        select_output = self.topk_selector(x=scores, batch=batch)

        # Store the scores and connectivity info in SelectOutput for loss computation
        # The pooler will access these to compute the MaxCut loss
        select_output.scores = scores.squeeze(-1)  # Store as (N,) for loss function
        select_output.edge_index = edge_index
        select_output.edge_weight = edge_weight
        select_output._extra_args.update({'scores', 'edge_index', 'edge_weight'})

        return select_output

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"ratio={self.ratio})"
        ) 