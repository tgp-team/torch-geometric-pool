from typing import Callable, Optional, Union

import torch
from torch import Tensor
from torch_geometric.nn.inits import uniform
from torch_geometric.nn.pool.select.topk import topk
from torch_geometric.nn.resolver import activation_resolver
from torch_geometric.utils import softmax

from tgp.select import Select, SelectOutput
from tgp.utils.typing import SinvType


class TopkSelect(Select):
    r"""The top-:math:`k` :math:`\texttt{select}` operator used by
    scoring-based pooling methods.

    It learns a score vector :math:`\mathbf{s} \in \mathbb{R}^{N}`,
    which assigns a score to each node :math:`i` in the graph.
    If :obj:`min_score` is :obj:`None`, computes:

    .. math::
        \mathbf{s} &= \sigma \left( \frac{\mathbf{X}\mathbf{p}}{\|
        \mathbf{p} \|} \right)

        \mathbf{i} &= \mathrm{top}_k(\mathbf{s})

    If :obj:`min_score` is a value :math:`\tilde{\alpha} \in [0,1]`,
    computes:

    .. math::
        \mathbf{s} &= \mathrm{softmax}(\mathbf{X}\mathbf{p})

        \mathbf{i} &= \mathbf{s}_i > \tilde{\alpha}

    where :math:`\mathbf{p}` is the learnable projection vector.

    The :class:`~tgp.select.SelectOutput` contains a sparse assignment
    matrix :math:`\mathbf{S}` that can be thought as dropping all the columns
    :math:`j \notin \mathbf{i}` of the diagonal matrix :math:`\text{diag}(\mathbf{s})`.

    Args:
        in_channels (int):
            Size of each input sample.
        ratio (float or int):
            The graph pooling ratio, which is used to compute
            :math:`k = \lceil \mathrm{ratio} \cdot N \rceil`, or the value
            of :math:`k` itself, depending on whether the type of :obj:`ratio`
            is :obj:`float` or :obj:`int`.
            This value is ignored if :obj:`min_score` is not :obj:`None`.
            (default: :obj:`0.5`)
        min_score (float, optional):
            Minimal node score :math:`\tilde{\alpha}`
            which is used to compute indices of pooled nodes
            :math:`\mathbf{i} = \mathbf{s}_i > \tilde{\alpha}`.
            When this value is not :obj:`None`, the :obj:`ratio` argument is ignored.
            (default: :obj:`None`)
        act (str or callable, optional):
            The non-linearity :math:`\sigma` to use when computing the score.
            (default: :obj:`"tanh"`)
        s_inv_op (~tgp.typing.SinvType, optional):
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
        in_channels: Optional[int] = None,
        ratio: Union[int, float] = 0.5,
        min_score: Optional[float] = None,
        act: Union[str, Callable] = "tanh",
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__()

        if ratio is None and min_score is None:
            raise ValueError(
                f"At least one of the 'ratio' and 'min_score' "
                f"parameters must be specified in "
                f"'{self.__class__.__name__}'"
            )

        self.in_channels = in_channels
        self.ratio = ratio
        self.min_score = min_score
        if act in ["linear", "identity", "none", None]:
            self.act = lambda x: x
        else:
            self.act = activation_resolver(act)
        self.s_inv_op = s_inv_op

        if in_channels is None or in_channels <= 1:
            self.register_parameter("weight", None)
        else:
            self.weight = torch.nn.Parameter(torch.empty(1, in_channels))

        self.reset_parameters()

    def reset_parameters(self):
        if self.weight is not None:
            uniform(self.in_channels, self.weight)

    def forward(
        self, x: Tensor, *, batch: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of the :math:`\texttt{select}` operator.
        """
        if batch is None:
            batch = x.new_zeros(x.size(0), dtype=torch.long)

        if self.weight is None:
            score = x if x.dim() == 1 else x.view(-1)
        else:
            x = x.view(-1, 1) if x.dim() == 1 else x
            score = (x * self.weight).sum(dim=-1)
            if self.min_score is None:
                score = score / self.weight.norm(p=2, dim=-1)

        score = self.act(score) if self.min_score is None else softmax(score, batch)

        node_index = topk(score, self.ratio, batch, self.min_score)

        return SelectOutput(
            node_index=node_index,
            num_nodes=x.size(-2),
            cluster_index=torch.arange(node_index.size(0), device=x.device),
            num_clusters=node_index.size(0),
            weight=score[node_index],
            s_inv_op=self.s_inv_op,
        )

    def __repr__(self) -> str:
        if self.min_score is None:
            arg = f"ratio={self.ratio}"
        else:
            arg = f"min_score={self.min_score}"
        return (
            f"{self.__class__.__name__}("
            f"in_channels={self.in_channels}, "
            f"{arg}, "
            f"act={self.act}, "
            f"s_inv_op={self.s_inv_op})"
        )
