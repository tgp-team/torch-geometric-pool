from typing import List, Optional, Union

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import DenseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import MLPSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import (
    mincut_loss,
    orthogonality_loss,
    sparse_mincut_loss,
    unbatched_orthogonality_loss,
)
from tgp.utils.ops import connectivity_to_edge_index, postprocess_adj_pool_dense
from tgp.utils.typing import LiftType, SinvType


class MinCutPooling(DenseSRCPooling):
    r"""The MinCut pooling operator from the paper `"Spectral Clustering in Graph
    Neural Networks for Graph Pooling" <https://arxiv.org/abs/1907.00481>`_
    (Bianchi et al., ICML 2020).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.MLPSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer optimizes two auxiliary losses:

    + the mincut loss (:func:`~tgp.utils.losses.mincut_loss` for batched,
      :func:`~tgp.utils.losses.sparse_mincut_loss` for unbatched),
    + the orthogonality loss (:func:`~tgp.utils.losses.orthogonality_loss` for batched,
      :func:`~tgp.utils.losses.unbatched_orthogonality_loss` for unbatched).

    Args:
        in_channels (int, list of int):
            Number of hidden units for each hidden layer in the MLP
            of the :math:`\texttt{select}` operator.
            The first integer must match the size of the node features.
        k (int):
            Number of clusters or supernodes in the pooler graph.
        act (str or Callable, optional):
            Activation function in the hidden layers of the MLP
            of the :math:`\texttt{select}` operator.
        dropout (float, optional):
            Dropout probability in the MLP of the :math:`\texttt{select}` operator.
            (default: :obj:`0.0`)
        cut_loss_coeff (float, optional):
            Coefficient for the MinCut loss (default: :obj:`1.0`)
        ortho_loss_coeff (float, optional):
            Coefficient for the orthogonality loss (default: :obj:`1.0`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the adjacency matrix.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
        adj_transpose (bool, optional):
            If :obj:`True`, the preprocessing step in :class:`tgp.src.DenseSRCPooling` and
            the :class:`tgp.connect.DenseConnect` operation returns transposed
            adjacency matrices, so that they could be passed "as is" to the dense
            message-passing layers.
            (default: :obj:`True`)
        cache_preprocessing (bool, optional):
            If :obj:`True`, caches the dense adjacency produced during preprocessing.
            This should only be enabled when the same graph is reused across iterations.
            (default: :obj:`False`)
        lift (~tgp.typing.LiftType, optional):
            Defines how to compute the matrix :math:`\mathbf{S}_\text{inv}` to lift the pooled node features.

            - :obj:`"precomputed"` (default): Use as :math:`\mathbf{S}_\text{inv}` what is
              already stored in the :obj:`"s_inv"` attribute of the :class:`tgp.select.SelectOutput`.
            - :obj:`"transpose"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Recomputes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.

        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.

        batched (bool, optional):
            If :obj:`True`, uses the batched dense path which converts sparse inputs
            to dense padded tensors. If :obj:`False`, uses the unbatched path which
            operates on sparse adjacency matrices without padding, providing better
            memory efficiency for graphs with varying sizes.
            (default: :obj:`True`)
        sparse_output (bool, optional):
            If :obj:`True`, returns block-diagonal sparse outputs. If :obj:`False`,
            returns batched dense outputs. (default: :obj:`False`)
    """

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,
        act: str = None,
        dropout: float = 0.0,
        cut_loss_coeff: float = 1.0,
        ortho_loss_coeff: float = 1.0,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
        adj_transpose: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
        batched: bool = True,
        sparse_output: bool = False,
        cache_preprocessing: bool = False,
    ):
        super().__init__(
            selector=MLPSelect(
                in_channels=in_channels,
                k=k,
                batched_representation=batched,
                act=act,
                dropout=dropout,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
                edge_weight_norm=edge_weight_norm,
                sparse_output=sparse_output,
            ),
            adj_transpose=adj_transpose,
            cache_preprocessing=cache_preprocessing,
            batched=batched,
            sparse_output=sparse_output,
        )

        self.cut_loss_coeff = cut_loss_coeff
        self.ortho_loss_coeff = ortho_loss_coeff

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
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor.
                For batched mode: :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`,
                with batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                For unbatched mode: :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`,
                where :math:`N` is the total number of nodes across all graphs.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                For batched mode: it can be either sparse connectivity
                (:obj:`edge_index`, :obj:`~torch_sparse.SparseTensor`, or torch COO),
                which is internally converted to a dense padded tensor of shape
                :math:`[B, N, N]`, or an already dense tensor of shape
                :math:`[B, N, N]`.
                For unbatched mode: Sparse connectivity matrix in one of the formats
                supported by :class:`~torch_geometric.typing.Adj` (edge_index, SparseTensor, etc.).
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape :math:`[E]` or
                :math:`[E, 1]` containing the weights of the edges (unbatched mode only).
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph. Only used when inputs are already
                dense/padded. (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional): The batch vector for the pooled nodes.
                Required when lifting with dense :math:`[N, K]` SelectOutput on multi-graph
                batches. Pass `out.batch` from the pooling call. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            batch_orig = batch if batch is not None else so.batch
            x_lifted = self.lift(
                x_pool=x, so=so, batch=batch_orig, batch_pooled=batch_pooled
            )
            return x_lifted

        # === Batched path ===
        if self.batched:
            x, adj, mask = self._ensure_batched_inputs(
                x=x,
                edge_index=adj,
                edge_weight=edge_weight,
                batch=batch,
                mask=mask,
            )

            # Select
            so = self.select(x=x, mask=mask)

            # Reduce
            x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

            # Connect
            adj_pool = self.connector.dense_connect(adj=adj, s=so.s)

            loss = self.compute_loss(adj, so.s, adj_pool)

            # Normalize coarsened adjacency matrix
            adj_pool = postprocess_adj_pool_dense(
                adj_pool,
                remove_self_loops=self.connector.remove_self_loops,
                degree_norm=self.connector.degree_norm,
                adj_transpose=self.connector.adj_transpose,
                edge_weight_norm=self.connector.edge_weight_norm,
            )

            if self.sparse_output:
                x_pooled, edge_index_pooled, edge_weight_pooled, batch_pooled = (
                    self._finalize_sparse_output(
                        x_pool=x_pooled,
                        adj_pool=adj_pool,
                        batch=batch,
                        batch_pooled=batch_pooled,
                        so=so,
                    )
                )
                return PoolingOutput(
                    x=x_pooled,
                    edge_index=edge_index_pooled,
                    edge_weight=edge_weight_pooled,
                    batch=batch_pooled,
                    so=so,
                    loss=loss,
                )

            return PoolingOutput(x=x_pooled, edge_index=adj_pool, so=so, loss=loss)

        # === Unbatched (sparse-loss) path ===
        # Select
        so = self.select(x=x, batch=batch)

        # Compute sparse loss
        loss = self.compute_sparse_loss(adj, edge_weight, so.s, batch)

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
            loss=loss,
        )

    def compute_loss(self, adj: Tensor, S: Tensor, adj_pooled: Tensor) -> dict:
        """Computes the auxiliary loss terms for batched (dense) mode.

        Args:
            adj (~torch.Tensor): The dense adjacency matrix of shape :math:`(B, N, N)`.
            S (~torch.Tensor): The dense assignment matrix of shape :math:`(B, N, K)`.
            adj_pooled (~torch.Tensor): The pooled adjacency matrix of shape :math:`(B, K, K)`.

        Returns:
            dict: A dictionary with the different terms of the auxiliary loss:
                - :obj:`'cut_loss'`: The mincut loss weighted by :attr:`cut_loss_coeff`.
                - :obj:`'ortho_loss'`: The orthogonality loss weighted by :attr:`ortho_loss_coeff`.
        """
        cut_loss = mincut_loss(adj, S, adj_pooled, batch_reduction="mean")
        ortho_loss = orthogonality_loss(S, batch_reduction="mean")

        return {
            "cut_loss": cut_loss * self.cut_loss_coeff,
            "ortho_loss": ortho_loss * self.ortho_loss_coeff,
        }

    def compute_sparse_loss(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor],
        S: Tensor,
        batch: Optional[Tensor],
    ) -> dict:
        """Computes the auxiliary loss terms for unbatched (sparse) mode.

        This method is used when :attr:`batched=False` and operates on sparse
        adjacency matrices without requiring padding or densification.

        Args:
            edge_index (~torch_geometric.typing.Adj): Graph connectivity in sparse format.
            edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
            S (~torch.Tensor): The dense assignment matrix of shape :math:`(N, K)`.
            batch (~torch.Tensor, optional): Batch vector of shape :math:`(N,)`.

        Returns:
            dict: A dictionary with the different terms of the auxiliary loss:
                - :obj:`'cut_loss'`: The sparse mincut loss weighted by :attr:`cut_loss_coeff`.
                - :obj:`'ortho_loss'`: The unbatched orthogonality loss weighted by :attr:`ortho_loss_coeff`.
        """
        edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
            edge_index, edge_weight
        )

        cut_loss = sparse_mincut_loss(
            edge_index_conv, S, edge_weight_conv, batch, batch_reduction="mean"
        )
        ortho_loss = unbatched_orthogonality_loss(S, batch, batch_reduction="mean")

        return {
            "cut_loss": cut_loss * self.cut_loss_coeff,
            "ortho_loss": ortho_loss * self.ortho_loss_coeff,
        }

    def extra_repr_args(self) -> dict:
        return {
            "batched": self.batched,
            "cut_loss_coeff": self.cut_loss_coeff,
            "ortho_loss_coeff": self.ortho_loss_coeff,
        }
