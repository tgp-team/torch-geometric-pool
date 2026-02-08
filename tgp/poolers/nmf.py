import warnings
from typing import Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import NMFSelect, SelectOutput
from tgp.src import BasePrecoarseningMixin, DenseSRCPooling, PoolingOutput
from tgp.utils.typing import LiftType, SinvType


class NMFPooling(BasePrecoarseningMixin, DenseSRCPooling):
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

    Notes:
        - This implementation supports sparse inputs and multi-graph batches via
          :obj:`edge_index` + :obj:`batch`.
        - Dense padded batched inputs (:math:`[B, N, N]`) are not supported.

    Args:
        k (int):
            Number of clusters or supernodes in the pooler graph.
        cached (bool, optional):
            If set to :obj:`True`, the output of the :math:`\texttt{select}` and :math:`\texttt{select}`
            operations will be cached, so that they do not need to be recomputed.
            (default: :obj:`False`)
        cache_preprocessing (bool, optional):
            If :obj:`True`, caches the dense adjacency produced during preprocessing.
            This should only be enabled when the same graph is reused across iterations.
            (default: :obj:`False`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops from the graph after coarsening.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, normalize the pooled adjacency matrix by the
            nodes' degree.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
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
        batched (bool, optional):
            Kept for API compatibility. Dense padded batched mode is unsupported
            and this option is ignored.
            (default: :obj:`False`)
        sparse_output (bool, optional):
            If :obj:`True`, return sparse pooled connectivity. (default: :obj:`False`)
    """

    def __init__(
        self,
        k: int,
        cached: bool = False,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
        adj_transpose: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        batched: bool = False,
        sparse_output: bool = False,
        cache_preprocessing: bool = False,
    ):
        if batched:
            warnings.warn(
                "NMFPooling does not support dense padded batched inputs. "
                "Use sparse edge_index with a batch vector.",
                UserWarning,
            )

        super().__init__(
            selector=NMFSelect(k=k, s_inv_op=s_inv_op),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
                edge_weight_norm=edge_weight_norm,
                sparse_output=sparse_output,
            ),
            cached=cached,
            cache_preprocessing=cache_preprocessing,
            adj_transpose=adj_transpose,
            batched=False,
            sparse_output=sparse_output,
        )

        self.cached = cached

        # Connector used in the precoarsening step (always sparse output).
        self.preconnector = DenseConnect(
            remove_self_loops=remove_self_loops,
            degree_norm=degree_norm,
            edge_weight_norm=edge_weight_norm,
            sparse_output=True,
        )

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> Union[PoolingOutput, Tensor]:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor.
                Node features :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                Sparse connectivity in one of the formats supported by
                :class:`~torch_geometric.typing.Adj`.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): Edge weights for sparse inputs.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Unused, kept for API compatibility.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): Batch vector
                :math:`\mathbf{b} \in \{0,\ldots,B-1\}^{N}` for sparse inputs.
                (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional): Batch vector for pooled nodes.
                Required when lifting from dense :math:`[N, K]` assignments on
                multi-graph batches. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(
                x_pool=x, so=so, batch=batch, batch_pooled=batch_pooled
            )
            return x_lifted

        if so is None:
            # Select
            so = self.select(
                edge_index=adj,
                edge_weight=edge_weight,
                batch=batch,
                num_nodes=x.size(0),
            )

        # Reduce
        return_batched = not self.sparse_output
        x_pooled, batch_pooled = self.reduce(
            x=x, so=so, batch=batch, return_batched=return_batched
        )

        # Connect
        edge_index_pooled, edge_weight_pooled = self.connect(
            edge_index=adj,
            so=so,
            edge_weight=edge_weight,
            batch=batch,
            batch_pooled=batch_pooled,
        )

        return PoolingOutput(
            x=x_pooled,
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
        )

    def precoarsening(
        self,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> PoolingOutput:
        # Keep assignment width fixed to k across samples during dataset pre-transform.
        return super().precoarsening(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            fixed_k=True,
            **kwargs,
        )

    def extra_repr_args(self) -> dict:
        return {
            "batched": self.batched,
            "cached": self.cached,
        }
