import warnings
from typing import Optional, Union

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import EigenPoolConnect
from tgp.lift import EigenPoolLift
from tgp.reduce import EigenPoolReduce
from tgp.select import EigenPoolSelect, SelectOutput
from tgp.src import BasePrecoarseningMixin, DenseSRCPooling, PoolingOutput
from tgp.utils.typing import LiftType, SinvType


class EigenPooling(BasePrecoarseningMixin, DenseSRCPooling):
    r"""The EigenPooling operator from
    `"Graph Convolutional Networks with EigenPooling"
    <https://arxiv.org/abs/1904.13107>`_ (Ma et al., KDD 2019).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.EigenPoolSelect`.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.EigenPoolReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.EigenPoolConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.EigenPoolLift`.

    Let:

    + :math:`\mathbf{X} \in \mathbb{R}^{N \times F}` be node features;
    + :math:`\mathbf{S} \in \{0,1\}^{N \times K}` be the hard assignment matrix
      produced by :class:`~tgp.select.EigenPoolSelect`;
    + :math:`\boldsymbol{\Omega} := \mathbf{S}` (same matrix, connectivity notation);
    + :math:`\mathbf{A}_{\text{ext}} \in \mathbb{R}^{N \times N}` be the input
      (possibly block-diagonal) adjacency used by the connector;
    + :math:`H` be the number of eigenvector modes.

    EigenPooling first partitions nodes into :math:`K` clusters via spectral
    clustering, then builds a multi-mode pooling matrix
    :math:`\boldsymbol{\Theta} \in \mathbb{R}^{N \times (K\cdot H)}` from
    Laplacian eigenvectors of each cluster-induced subgraph. Features are pooled as:

    .. math::
        \mathbf{X}_{\text{pool,raw}} = \boldsymbol{\Theta}^{\top}\mathbf{X},

    then reshaped from :math:`[H\!\cdot\!K, F]` to :math:`[K, H\!\cdot\!F]`.

    Connectivity is coarsened as:

    .. math::
        \mathbf{A}_{\text{coar}} = \boldsymbol{\Omega}^{\top}\mathbf{A}_{\text{ext}}\boldsymbol{\Omega}.

    Notes:
        - This implementation supports sparse inputs and multi-graph batches via
          :obj:`edge_index` + :obj:`batch`.
        - Dense padded batched inputs (:math:`[B, N, N]`) are **not** supported.

    Args:
        k (int):
            Number of clusters (supernodes) in the pooled graph.
        num_modes (int, optional):
            Number of eigenvector modes :math:`H`. (default: :obj:`5`)
        normalized (bool, optional):
            If :obj:`True`, use the normalized Laplacian. (default: :obj:`True`)
        cached (bool, optional):
            If :obj:`True`, cache :class:`~tgp.select.SelectOutput`. (default: :obj:`False`)
        remove_self_loops (bool, optional):
            Whether to remove self-loops after coarsening. (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, symmetrically normalize pooled adjacency. (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize pooled edge weights. (default: :obj:`False`)
        adj_transpose (bool, optional):
            Passed to the connector for adjacency post-processing. (default: :obj:`True`)
        lift (~tgp.utils.typing.LiftType, optional):
            Kept for API compatibility. EigenPooling always uses eigenvector-based
            lifting and ignores this option. (default: :obj:`"precomputed"`)
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            Operation used to compute :math:`\mathbf{S}_\text{inv}` in
            :class:`~tgp.select.SelectOutput`. (default: :obj:`"transpose"`)
        batched (bool, optional):
            Kept for API compatibility. Dense batched mode is unsupported and this
            option is ignored. Use sparse inputs with :obj:`batch` instead.
            (default: :obj:`False`)
        sparse_output (bool, optional):
            If :obj:`True`, return sparse pooled connectivity. (default: :obj:`False`)
        cache_preprocessing (bool, optional):
            Passed to :class:`~tgp.src.DenseSRCPooling`; has no practical effect for
            this sparse-oriented path. (default: :obj:`False`)
    """

    def __init__(
        self,
        k: int,
        num_modes: int = 5,
        normalized: bool = True,
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
                "EigenPooling does not support dense padded batched inputs. "
                "Use batched=False with a sparse edge_index and batch vector.",
                UserWarning,
            )
        if lift != "precomputed":
            warnings.warn(
                "EigenPooling ignores the 'lift' argument and always uses "
                "eigenvector-based lifting.",
                UserWarning,
            )
        # EigenPooling always uses unbatched mode
        # because spectral clustering operates on individual graphs
        super().__init__(
            selector=EigenPoolSelect(
                k=k,
                s_inv_op=s_inv_op,
                num_modes=num_modes,
                normalized=normalized,
            ),
            reducer=EigenPoolReduce(
                num_modes=num_modes,
            ),
            lifter=EigenPoolLift(
                num_modes=num_modes,
            ),
            connector=EigenPoolConnect(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                adj_transpose=adj_transpose,
                edge_weight_norm=edge_weight_norm,
                sparse_output=sparse_output,
            ),
            cached=cached,
            cache_preprocessing=cache_preprocessing,
            adj_transpose=adj_transpose,
            batched=False,  # Always use unbatched mode
            sparse_output=sparse_output,
        )

        self.k = k
        self.num_modes = num_modes
        self.normalized = normalized
        self.cached = cached

        # Connector for precoarsening (always sparse output)
        self.preconnector = EigenPoolConnect(
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
            x (~torch.Tensor):
                Node features :math:`\mathbf{X} \in \mathbb{R}^{N \times F}`.
                During lifting, accepts pooled features
                :math:`\mathbf{X}_{\text{pool}} \in \mathbb{R}^{K \times (H\cdot F)}`.
            adj (~torch_geometric.typing.Adj, optional):
                Sparse graph connectivity (edge index, :class:`~torch_sparse.SparseTensor`,
                or torch COO tensor). Internally interpreted as
                :math:`\mathbf{A}_{\text{ext}}`; required when :obj:`lifting=False`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                Edge weights associated with :obj:`adj`. (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional):
                Pre-computed selection output. (default: :obj:`None`)
            mask (~torch.Tensor, optional):
                Unused, kept for API compatibility. (default: :obj:`None`)
            batch (~torch.Tensor, optional):
                Batch vector for sparse multi-graph inputs. (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                Batch vector for pooled nodes, used during lifting.
                (default: :obj:`None`)
            lifting (bool, optional):
                If :obj:`True`, apply :math:`\texttt{lift}` instead of pooling.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput or ~torch.Tensor:
                Pooled output if :obj:`lifting=False`, otherwise lifted features.
        """
        if lifting:
            # Lift expects 2D x_pool [N, F]; flatten if we got dense batched [B, K, F]
            x_pool = x
            if x.dim() == 3:
                B, K, F = x.shape
                x_pool = x.view(-1, F)
                if batch_pooled is None:
                    batch_pooled = torch.arange(
                        B, dtype=torch.long, device=x.device
                    ).repeat_interleave(K)
            return self.lift(
                x_pool=x_pool,
                so=so,
                batch=batch,
                batch_pooled=batch_pooled,
            )

        # Select (if not precomputed)
        if so is None:
            so = self.select(edge_index=adj, edge_weight=edge_weight, batch=batch)

        # Reduce
        x_pooled, pooled_batch = self.reduce(x=x, so=so, batch=batch)

        # Connect
        adj_pooled, edge_weight_pooled = self.connect(
            edge_index=adj,
            so=so,
            edge_weight=edge_weight,
            batch=batch,
            batch_pooled=pooled_batch,
        )

        # When dense output and multiple graphs: reshape x_pooled to [B, K, F]
        if (
            not self.sparse_output
            and pooled_batch is not None
            and pooled_batch.numel() > 0
        ):
            batch_size = int(pooled_batch.max().item()) + 1
            num_clusters = so.s.size(-1)
            x_pooled = x_pooled.view(batch_size, num_clusters, -1)

        return PoolingOutput(
            x=x_pooled,
            edge_index=adj_pooled,
            edge_weight=edge_weight_pooled,
            batch=pooled_batch,
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
        # In pre-coarsening, fix the assignment width to k across samples so
        # batched collation can concatenate dense SelectOutput.s safely.
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
            "k": self.k,
            "num_modes": self.num_modes,
            "normalized": self.normalized,
            "cached": self.cached,
        }
