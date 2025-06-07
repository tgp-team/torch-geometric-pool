from typing import Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import NMFSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.typing import LiftType, SinvType


class NMFPooling(DenseSRCPooling):
    r"""The Non-negative Matrix Factorization
    pooling as proposed in the paper `"A Non-Negative Factorization approach
    to node pooling in Graph Convolutional Neural Networks"
    <https://arxiv.org/abs/1909.03287>`_ (Bacciu and Di Sotto, AIIA 2019).

    NMF pooling performs a Nonnegative Matrix Factorization of the adjacency matrix

    .. math::
        \mathbf{A} \approx \mathbf{W} \mathbf{H}

    where :math:`\mathbf{H}` is the soft cluster assignment matrix
    and :math:`\mathbf{W}` is the cluster centroid matrix.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.NMFSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    Args:
        k (int):
            Number of clusters or supernodes in the pooler graph.
        cached (bool, optional):
            If set to :obj:`True`, the output of the :math:`\texttt{select}` and :math:`\texttt{select}`
            operations will be cached, so that they do not need to be recomputed.
            (default: :obj:`False`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, normalize the pooled adjacency matrix by the
            nodes' degree.
            (default: :obj:`True`)
        adj_transpose (bool, optional):
            If :obj:`True`, the preprocessing step in :class:`~tgp.src.DenseSRCPooling` and
            the :class:`~tgp.connect.DenseConnect` operation returns transposed
            adjacency matrices, so that they could be passed "as is" to the dense
            message-passing layers.
            (default: :obj:`True`)
        lift (~tgp.typing.LiftType, optional):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - :obj:`"precomputed"` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the :obj:`"s_inv"` attribute of the :class:`~tgp.select.SelectOutput`.
            - :obj:`"transpose"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
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
        k: int,
        cached: bool = False,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        adj_transpose: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
    ):
        super().__init__(
            selector=NMFSelect(k=k, s_inv_op=s_inv_op),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
            ),
            cached=cached,
            adj_transpose=adj_transpose,
        )

        self.cached = cached

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> Union[PoolingOutput, Tensor]:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (~torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            # Select
            so = self.select(edge_index=adj, mask=mask)

            # Reduce
            x_pooled, _ = self.reduce(x=x, so=so)

            # Connect
            adj_pooled, _ = self.connect(edge_index=adj, so=so)

            out = PoolingOutput(x=x_pooled, edge_index=adj_pooled, so=so)

            return out

    def extra_repr_args(self) -> dict:
        return {
            "cached": self.cached,
        }
