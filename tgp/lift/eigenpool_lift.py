from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import unbatch

from tgp.lift.base_lift import Lift
from tgp.select import SelectOutput
from tgp.utils.typing import ReduceType


class EigenPoolLift(Lift):
    r"""The :math:`\texttt{lift}` operator for EigenPooling.

    It uses the pooling matrix :math:`\boldsymbol{\Theta}` stored in
    :obj:`so.theta` and lifts pooled features back to node space as:

    .. math::
        \mathbf{X}_{\text{lift}} =
        \boldsymbol{\Theta}\mathbf{X}_{\text{pool,raw}},

    where :math:`\mathbf{X}_{\text{pool,raw}}` is the mode-major version of the
    pooled features.

    Args:
        num_modes (int, optional):
            Number of eigenvector modes :math:`H`. Kept for API symmetry with the
            EigenPool components. (default: :obj:`5`)
        reduce_op (~tgp.utils.typing.ReduceType, optional):
            Kept for API compatibility with :class:`~tgp.lift.Lift`.
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
    def _reshape_feature_blocks_to_mode_major(
        x_pool: Tensor,
        num_clusters: int,
        num_modes: int,
    ) -> Tensor:
        r"""Reshape pooled features from :math:`[K, H\cdot F]` to
        mode-major :math:`[H\cdot K, F]`.
        """
        feat_dim = x_pool.size(-1) // num_modes
        return (
            x_pool.view(num_clusters, num_modes, feat_dim)
            .permute(1, 0, 2)
            .reshape(num_modes * num_clusters, feat_dim)
        )

    @classmethod
    def _lift_with_theta(
        cls,
        theta: Tensor,
        x_pool: Tensor,
        num_clusters: int,
    ) -> Tensor:
        num_modes = theta.size(-1) // num_clusters
        x_pool_mode_major = cls._reshape_feature_blocks_to_mode_major(
            x_pool=x_pool,
            num_clusters=num_clusters,
            num_modes=num_modes,
        )

        if theta.is_sparse:
            return torch.sparse.mm(theta, x_pool_mode_major)
        return theta.matmul(x_pool_mode_major)

    def forward(
        self,
        x_pool: Tensor,
        so: SelectOutput = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        edge_index: Optional[Tensor] = None,
        edge_weight: Optional[Tensor] = None,
        **kwargs,
    ) -> Tensor:
        r"""Forward pass.

        Args:
            x_pool (~torch.Tensor):
                Pooled feature matrix of shape :math:`[K, H\cdot F]`,
                :math:`[B\cdot K, H\cdot F]`, or :math:`[B, K, H\cdot F]`.
            so (~tgp.select.SelectOutput, optional):
                Output of the :math:`\texttt{select}` operator with dense
                assignment matrix :obj:`so.s` and pooling matrix :obj:`so.theta`.
            batch (~torch.Tensor, optional):
                Batch vector for original nodes. If :obj:`None`, this method uses
                :obj:`so.batch` when available. (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional):
                Batch vector for pooled nodes in multi-graph lifting.
                (default: :obj:`None`)
            edge_index (~torch.Tensor, optional):
                Unused for EigenPooling; kept for API compatibility.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                Unused for EigenPooling; kept for API compatibility.
                (default: :obj:`None`)

        Returns:
            ~torch.Tensor:
                Lifted node features of shape :math:`[N, F]`.
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

        # Single graph case.
        if not is_multi_graph:
            x_pool_mat = x_pool.squeeze(0) if x_pool.dim() == 3 else x_pool
            return self._lift_with_theta(
                theta=theta, x_pool=x_pool_mat, num_clusters=num_clusters
            )

        # Multi-graph case: unbatch pooled features and theta, lift each graph, then concatenate.
        batch_size = int(batch.max().item()) + 1
        if batch_pooled is None:
            batch_pooled = torch.arange(
                batch_size, dtype=batch.dtype, device=batch.device
            ).repeat_interleave(num_clusters)

        x_pool_flat = x_pool.view(-1, x_pool.size(-1)) if x_pool.dim() == 3 else x_pool
        x_pool_list = unbatch(x_pool_flat, batch=batch_pooled)
        theta_list = theta if isinstance(theta, list) else unbatch(theta, batch=batch)

        x_lift_list = [
            self._lift_with_theta(
                theta=theta_b, x_pool=x_pool_b, num_clusters=num_clusters
            )
            for theta_b, x_pool_b in zip(theta_list, x_pool_list)
        ]
        return torch.cat(x_lift_list, dim=0)

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(num_modes={self.num_modes})"
