"""EigenPool lift operator with eigenvector-based unpooling.

This module uses eigenvector-based pooling matrices computed during Select
and stored in SelectOutput.theta.
"""

from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils import unbatch

from tgp.lift.base_lift import Lift
from tgp.select import SelectOutput
from tgp.utils.typing import ReduceType


class EigenPoolLift(Lift):
    """EigenPool lift operator with eigenvector-based unpooling.

    This operator uses eigenvector-based pooling matrices computed in Select
    and stored in SelectOutput.theta.
    It expects pooled features of shape [K, H*d] and produces lifted features
    of shape [N, d].

    Args:
        num_modes: Number of eigenvector modes (H) to use for lifting. (default: 5)
        normalized: If True, use normalized Laplacian for eigenvector computation.
            (default: True)
        reduce_op: Aggregation function (unused, kept for API). (default: "sum")
    """

    def __init__(
        self,
        num_modes: int = 5,
        normalized: bool = True,
        reduce_op: ReduceType = "sum",
    ):
        super().__init__()
        self.num_modes = num_modes
        self.normalized = normalized
        self.reduce_op = reduce_op

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
        """Forward pass with eigenvector-based unpooling.

        Args:
            x_pool: Pooled features [K, H*d].
            so: SelectOutput containing the one-hot assignment matrix [N, K].
            batch: Batch vector for original nodes.
            batch_pooled: Batch vector for pooled nodes.
            edge_index: Unused for EigenPooling (kept for API compatibility).
            edge_weight: Unused for EigenPooling (kept for API compatibility).

        Returns:
            Lifted features [N, d].
        """
        if batch is None and hasattr(so, "batch") and so.batch is not None:
            batch = so.batch

        device = x_pool.device
        dtype = x_pool.dtype

        num_clusters = so.s.size(-1)
        is_multi_graph = (
            batch is not None
            and batch.numel() > 0
            and int(batch.min().item()) != int(batch.max().item())
        )

        if not hasattr(so, "theta") or so.theta is None:
            raise ValueError(
                "SelectOutput.theta is required for EigenPoolLift. "
                "Make sure EigenPoolSelect computes and stores it."
            )

        theta = so.theta

        def reshape_feature_blocks_to_mode_major(
            x_pool_mat: Tensor, n_clusters: int, n_modes: int, feat_dim: int
        ) -> Tensor:
            """Convert [K, H*d] (mode blocks per cluster) to [H*K, d] (mode-major)."""
            return (
                x_pool_mat.view(n_clusters, n_modes, feat_dim)
                .permute(1, 0, 2)
                .reshape(n_modes * n_clusters, feat_dim)
            )

        if is_multi_graph:
            if x_pool.dim() == 3:
                B, K, Hd = x_pool.shape
                x_pool_flat = x_pool.view(B * K, Hd)
            elif x_pool.dim() == 2:
                x_pool_flat = x_pool
            else:
                raise ValueError(
                    f"Expected x_pool to be 2D or 3D, got shape {x_pool.shape}"
                )

            batch_size = int(batch.max().item()) + 1
            expected_nodes = batch_size * num_clusters
            if x_pool_flat.size(0) != expected_nodes:
                raise ValueError(
                    "Unexpected pooled feature shape for multi-graph lifting: "
                    f"got x_pool.size(0)={x_pool_flat.size(0)}, expected {expected_nodes}."
                )

            if batch_pooled is None:
                batch_pooled = torch.arange(
                    batch_size, dtype=batch.dtype, device=batch.device
                ).repeat_interleave(num_clusters)

            unbatched_x_pool = unbatch(x_pool_flat, batch=batch_pooled)
            if isinstance(theta, list):
                theta_list = theta
            else:
                if batch is None:
                    raise ValueError(
                        "batch is required to unbatch SelectOutput.theta for batched lifting."
                    )
                theta_list = unbatch(theta, batch=batch)
            if len(theta_list) != batch_size:
                raise ValueError(
                    "SelectOutput.theta has an unexpected number of graphs for batched lifting: "
                    f"got {len(theta_list)}, expected {batch_size}."
                )

            x_lift_list = []
            for b in range(batch_size):
                theta_b = theta_list[b]
                if not isinstance(theta_b, Tensor):
                    theta_b = torch.tensor(theta_b, dtype=dtype, device=device)
                n_nodes = theta_b.size(0)
                actual_num_modes = (
                    theta_b.size(-1) // num_clusters if num_clusters > 0 else 1
                )
                x_pool_b = unbatched_x_pool[b]
                Hd = x_pool_b.size(-1)
                d = Hd // actual_num_modes if actual_num_modes > 0 else 0
                if n_nodes == 0:
                    x_lift_list.append(torch.zeros((0, d), dtype=dtype, device=device))
                    continue

                x_pool_reshaped = reshape_feature_blocks_to_mode_major(
                    x_pool_b,
                    num_clusters,
                    actual_num_modes,
                    d,
                )
                if theta_b.is_sparse:
                    x_lift_list.append(torch.sparse.mm(theta_b, x_pool_reshaped))
                else:
                    x_lift_list.append(theta_b.matmul(x_pool_reshaped))

            return torch.cat(x_lift_list, dim=0)

        theta_mat = theta
        if not isinstance(theta_mat, Tensor):
            theta_mat = torch.tensor(theta_mat, dtype=dtype, device=device)
        actual_num_modes = theta_mat.size(-1) // num_clusters
        if x_pool.dim() == 2:
            K, Hd = x_pool.shape
            d = Hd // actual_num_modes
            x_pool_reshaped = reshape_feature_blocks_to_mode_major(
                x_pool,
                K,
                actual_num_modes,
                d,
            )
        else:
            raise ValueError(
                f"Expected x_pool to be 2D [K, H*d], got shape {x_pool.shape}"
            )

        if theta_mat.is_sparse:
            x_lift = torch.sparse.mm(theta_mat, x_pool_reshaped)
        else:
            x_lift = theta_mat.matmul(x_pool_reshaped)

        return x_lift

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_modes={self.num_modes}, "
            f"normalized={self.normalized})"
        )
