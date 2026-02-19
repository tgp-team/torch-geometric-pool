from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import unbatch

from tgp.reduce.base_reduce import Reduce
from tgp.select import SelectOutput
from tgp.utils.typing import ReduceType


class EigenPoolReduce(Reduce):
    r"""The :math:`\texttt{reduce}` operator for EigenPooling.

    It uses the pooling matrix :math:`\boldsymbol{\Theta}` computed by
    :class:`~tgp.select.EigenPoolSelect` and stored in :obj:`so.theta`.
    For each graph:

    .. math::
        \mathbf{X}_{\text{pool,raw}} = \boldsymbol{\Theta}^{\top}\mathbf{X},

    then :math:`\mathbf{X}_{\text{pool,raw}}` is reshaped from mode-major layout
    :math:`[H \cdot K, F]` to :math:`[K, H \cdot F]`.

    Args:
        num_modes (int, optional):
            Number of eigenvector modes :math:`H`. Kept for API symmetry with the
            EigenPool components. (default: :obj:`5`)
        reduce_op (~tgp.utils.typing.ReduceType, optional):
            Kept for API compatibility with :class:`~tgp.reduce.Reduce`.
            (default: :obj:`"sum"`)
    """

    def __init__(
        self,
        num_modes: int = 5,
        reduce_op: ReduceType = "sum",
    ):
        super().__init__()
        self.num_modes = num_modes
        self.reduce_op = reduce_op

    @staticmethod
    def _pool_with_theta(theta: Tensor, x: Tensor) -> Tensor:
        if theta.is_sparse:
            return torch.sparse.mm(theta.t(), x)
        return theta.t().matmul(x)

    @staticmethod
    def _reshape_mode_major_to_feature_blocks(
        x_pool_raw: Tensor,
        num_clusters: int,
    ) -> Tensor:
        r"""Reshape pooled features from mode-major :math:`[H\cdot K, F]` to
        cluster-major feature blocks :math:`[K, H\cdot F]`.
        """
        num_modes = x_pool_raw.size(0) // num_clusters
        feat_dim = x_pool_raw.size(-1)
        return (
            x_pool_raw.view(num_modes, num_clusters, feat_dim)
            .permute(1, 0, 2)
            .reshape(num_clusters, num_modes * feat_dim)
        )

    def forward(
        self,
        x: Tensor,
        so: SelectOutput,
        *,
        batch: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        return_batched: bool = False,
        **kwargs,
    ) -> Tuple[Tensor, Optional[Tensor]]:
        r"""Forward pass.

        Args:
            x (~torch.Tensor):
                Node feature matrix :math:`\mathbf{X}` of shape :math:`[N, F]`.
            so (~tgp.select.SelectOutput):
                Output of the :math:`\texttt{select}` operator with dense
                assignment matrix :obj:`so.s` and pooling matrix :obj:`so.theta`.
            batch (~torch.Tensor, optional):
                Batch vector for sparse multi-graph inputs. If :obj:`None`, this
                method uses :obj:`so.batch` when available.
                (default: :obj:`None`)
            edge_index (~torch.Tensor, optional):
                Unused for EigenPooling; kept for API compatibility.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                Unused for EigenPooling; kept for API compatibility.
                (default: :obj:`None`)
            return_batched (bool, optional):
                If :obj:`True`, returns :math:`[B, K, H \cdot F]` for multi-graph
                batches and :math:`[1, K, H \cdot F]` for single graphs.
                (default: :obj:`False`)

        Returns:
            (~torch.Tensor, ~torch.Tensor or :obj:`None`):
                Pooled features and pooled batch vector.
        """
        if batch is None and so.batch is not None:
            batch = so.batch

        num_clusters = so.s.size(-1)
        theta = so.theta

        is_multi_graph = (
            batch is not None
            and batch.numel() > 0
            and int(batch.min().item()) != int(batch.max().item())
        )

        # Single graph case: directly pool with theta and reshape
        if not is_multi_graph:
            x_pool = self._pool_with_theta(theta, x)
            x_pool = self._reshape_mode_major_to_feature_blocks(x_pool, num_clusters)
            batch_pool = super().reduce_batch(so, batch)
            if return_batched:
                x_pool = x_pool.unsqueeze(0)
            return x_pool, batch_pool

        # Multi-graph batch case: unbatch theta and x, pool each graph separately, then concatenate results.
        theta_list = theta if isinstance(theta, list) else unbatch(theta, batch=batch)
        x_list = unbatch(x, batch=batch)

        pooled_features = []
        for theta_b, x_b in zip(theta_list, x_list):
            x_pool_b = self._pool_with_theta(theta_b, x_b)
            pooled_features.append(
                self._reshape_mode_major_to_feature_blocks(x_pool_b, num_clusters)
            )

        x_pool = torch.cat(pooled_features, dim=0)
        batch_pool = super().reduce_batch(so, batch)

        if return_batched:
            x_pool = x_pool.view(len(theta_list), num_clusters, -1)

        return x_pool, batch_pool

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_modes={self.num_modes})"
