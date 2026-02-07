"""EigenPool reduce operator with eigenvector-based pooling matrices.

This module uses eigenvector-based pooling matrices computed during Select
and stored in SelectOutput.theta.
"""

from typing import Optional, Tuple

import torch
from torch import Tensor
from torch_geometric.utils import unbatch

from tgp.reduce.base_reduce import Reduce
from tgp.select import SelectOutput
from tgp.utils.typing import ReduceType


class EigenPoolReduce(Reduce):
    """EigenPool reduce operator with eigenvector-based pooling matrices.

    This operator uses eigenvector-based pooling matrices computed in Select
    and stored in SelectOutput.theta.
    The pooling produces features of shape [K, H*d] where K is the number of
    clusters and H is the number of eigenvector modes.

    Args:
        num_modes: Number of eigenvector modes (H) to use for pooling. (default: 5)
        normalized: If True, use normalized Laplacian for eigenvector computation.
            (default: True)
        reduce_op: The aggregation function (unused for EigenPooling, kept for API).
            (default: "sum")
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

    @staticmethod
    def reduce_batch(
        select_output: SelectOutput,
        batch: Optional[Tensor],
    ) -> Optional[Tensor]:
        """Compute the batch vector for the coarsened graph.

        Args:
            select_output: The output of Select with [N, K] assignment.
            batch: The batch vector for original nodes.

        Returns:
            The pooled batch vector of size K.
        """
        if batch is None:
            return batch

        # Get num_clusters (K) from SelectOutput.s shape
        num_clusters = select_output.s.size(-1)

        # Handle empty batch case
        if batch.numel() == 0:
            return batch.new_empty((0,), dtype=batch.dtype)

        batch_size = int(batch.max().item()) + 1

        # batch_pooled assigns each supernode to its graph
        batch_pooled = torch.arange(
            batch_size, dtype=batch.dtype, device=batch.device
        ).repeat_interleave(num_clusters)

        return batch_pooled

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
        """Forward pass with eigenvector-based pooling.

        Args:
            x: Node feature matrix [N, d].
            so: SelectOutput containing the one-hot assignment matrix [N, K].
            batch: Batch vector for multi-graph batches.
            edge_index: Unused for EigenPooling (kept for API compatibility).
            edge_weight: Unused for EigenPooling (kept for API compatibility).
            return_batched: If True, return batched output [B, K, H*d].

        Returns:
            Tuple of (pooled features [K, H*d] or [total_K, H*d] for batched, pooled batch vector).
        """
        if batch is None and hasattr(so, "batch") and so.batch is not None:
            batch = so.batch

        device = x.device
        dtype = x.dtype
        num_clusters = so.s.size(-1)
        d = x.size(-1)

        if not hasattr(so, "theta") or so.theta is None:
            raise ValueError(
                "SelectOutput.theta is required for EigenPoolReduce. "
                "Make sure EigenPoolSelect computes and stores it."
            )

        theta = so.theta

        def pool_with_theta(theta_mat: Tensor, x_mat: Tensor) -> Tensor:
            if not isinstance(theta_mat, Tensor):
                theta_mat = torch.tensor(theta_mat, dtype=dtype, device=device)
            if theta_mat.is_sparse:
                return torch.sparse.mm(theta_mat.t(), x_mat)
            return theta_mat.t().matmul(x_mat)

        def reshape_mode_major_to_feature_blocks(
            x_pool_raw: Tensor, n_clusters: int
        ) -> Tensor:
            """Convert [H*K, d] (mode-major) to [K, H*d] (mode blocks per cluster)."""
            if n_clusters == 0:
                return x_pool_raw.new_zeros((0, 0))
            if x_pool_raw.size(0) % n_clusters != 0:
                raise ValueError(
                    "Invalid EigenPool reduce shape: "
                    f"got {x_pool_raw.size(0)} rows for {n_clusters} clusters."
                )
            n_modes = x_pool_raw.size(0) // n_clusters
            return (
                x_pool_raw.view(n_modes, n_clusters, d)
                .permute(1, 0, 2)
                .reshape(n_clusters, n_modes * d)
            )

        is_multi_graph = (
            batch is not None
            and batch.numel() > 0
            and int(batch.min().item()) != int(batch.max().item())
        )

        if is_multi_graph:
            if isinstance(theta, list):
                theta_list = theta
            else:
                theta_list = unbatch(theta, batch=batch)

            if batch is not None:
                x_list = unbatch(x, batch=batch)
            else:
                sizes = [t.size(0) for t in theta_list]
                x_list = torch.split(x, sizes, dim=0)

            pooled_features = []
            for theta_b, x_b in zip(theta_list, x_list):
                if not isinstance(theta_b, Tensor):
                    theta_b = torch.tensor(theta_b, dtype=dtype, device=device)
                actual_num_modes = (
                    theta_b.size(-1) // num_clusters if num_clusters > 0 else 0
                )
                if x_b.numel() == 0:
                    pooled_features.append(
                        torch.zeros(
                            num_clusters,
                            actual_num_modes * d,
                            dtype=dtype,
                            device=device,
                        )
                    )
                    continue
                x_pool_b = pool_with_theta(theta_b, x_b)
                x_pool_b = reshape_mode_major_to_feature_blocks(x_pool_b, num_clusters)
                pooled_features.append(x_pool_b)

            x_pool = torch.cat(pooled_features, dim=0)
            batch_pool = self.reduce_batch(so, batch)
            if return_batched:
                batch_size = len(theta_list)
                x_pool = x_pool.view(batch_size, num_clusters, -1)
            return x_pool, batch_pool

        x_pool = pool_with_theta(theta, x)
        x_pool = reshape_mode_major_to_feature_blocks(x_pool, num_clusters)
        batch_pool = self.reduce_batch(so, batch)

        if return_batched and x_pool.dim() == 2:
            x_pool = x_pool.unsqueeze(0)

        return x_pool, batch_pool

    def __repr__(self) -> str:
        return (
            f"{self.__class__.__name__}("
            f"num_modes={self.num_modes}, "
            f"normalized={self.normalized})"
        )
