import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from sklearn.cluster import SpectralClustering
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj

from tgp.select.base_select import Select, SelectOutput
from tgp.utils.ops import connectivity_to_edge_index
from tgp.utils.typing import SinvType


def laplacian(adj: np.ndarray, normalized: bool = True) -> np.ndarray:
    r"""Compute the graph Laplacian from a dense adjacency matrix.

    Given an adjacency matrix :math:`\mathbf{A}`, the function returns either:

    .. math::
        \mathbf{L} = \mathbf{D} - \mathbf{A} \quad \text{(unnormalized)}

    or the symmetric normalized Laplacian:

    .. math::
        \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}.

    Args:
        adj (np.ndarray):
            Dense adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`.
        normalized (bool, optional):
            If :obj:`True`, returns the normalized Laplacian.
            (default: :obj:`True`)

    Returns:
        np.ndarray:
            Laplacian matrix :math:`\mathbf{L} \in \mathbb{R}^{N \times N}`.
    """
    d = adj.sum(axis=0).reshape(-1)

    if not normalized:
        return np.diag(d) - adj

    d = d + np.spacing(np.array(0, dtype=adj.dtype))
    d_inv_sqrt = 1.0 / np.sqrt(d)
    d_inv_sqrt_mat = np.diag(d_inv_sqrt)
    identity = np.eye(d.size, dtype=adj.dtype)
    return identity - d_inv_sqrt_mat @ adj @ d_inv_sqrt_mat


def eigenvectors(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Compute eigenvalues and eigenvectors of a Laplacian matrix.

    Args:
        L (np.ndarray):
            Laplacian matrix :math:`\mathbf{L} \in \mathbb{R}^{N \times N}`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Eigenvalues :math:`\boldsymbol{\lambda} \in \mathbb{R}^{N}`
            - Eigenvectors :math:`\mathbf{U} \in \mathbb{R}^{N \times N}` (columns)
    """
    lamb, U = np.linalg.eigh(L)
    return lamb, U


def _group_nodes_by_cluster(cluster_labels: np.ndarray) -> dict:
    clusters = {}
    for node_idx, label in enumerate(cluster_labels):
        label_int = int(label)
        clusters.setdefault(label_int, []).append(node_idx)
    return {
        label: np.asarray(node_indices, dtype=np.int64)
        for label, node_indices in clusters.items()
    }


def build_pooling_matrix(
    adj_np: np.ndarray,
    cluster_labels: np.ndarray,
    num_modes: int,
    normalized: bool = True,
    expected_num_clusters: Optional[int] = None,
) -> np.ndarray:
    r"""Build the eigenvector-based pooling matrix :math:`\boldsymbol{\Theta}`.

    For each cluster, we compute the Laplacian of the induced subgraph and use
    the first :math:`H` eigenvectors as pooling modes. The resulting matrix is
    assembled as:

    .. math::
        \boldsymbol{\Theta} = [\boldsymbol{\Theta}^{(1)} \; \cdots \; \boldsymbol{\Theta}^{(H)}]
        \in \mathbb{R}^{N \times (K\cdot H)},

    where :math:`\boldsymbol{\Theta}^{(h)}` places the :math:`h`-th eigenvector
    of each cluster on the rows corresponding to its nodes and zeros elsewhere.

    Args:
        adj_np (np.ndarray):
            Dense adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`.
        cluster_labels (np.ndarray):
            Cluster assignment vector of length :math:`N`.
        num_modes (int):
            Number of eigenvector modes :math:`H`.
        normalized (bool, optional):
            If :obj:`True`, uses the normalized Laplacian.
            (default: :obj:`True`)
        expected_num_clusters (int, optional):
            Fixed number of clusters :math:`K` used to allocate columns in the
            output, even when some clusters are empty.

    Returns:
        np.ndarray:
            Pooling matrix
            :math:`\boldsymbol{\Theta} \in \mathbb{R}^{N \times (K \cdot H)}`.
    """
    num_nodes = adj_np.shape[0]
    clusters = _group_nodes_by_cluster(cluster_labels)

    if expected_num_clusters is None:
        label_to_idx = {label: idx for idx, label in enumerate(sorted(clusters))}
        num_clusters = len(label_to_idx)
    else:
        label_to_idx = {label: label for label in clusters}
        num_clusters = expected_num_clusters

    theta = np.zeros((num_nodes, num_clusters * num_modes), dtype=adj_np.dtype)

    for label, node_indices in clusters.items():
        cluster_idx = label_to_idx[label]
        adj_cluster = adj_np[np.ix_(node_indices, node_indices)]
        cluster_size = node_indices.size

        if cluster_size == 1:
            value = float(adj_cluster[0, 0])
            theta[node_indices[0], cluster_idx::num_clusters] = value
            continue

        _, eigvecs = eigenvectors(laplacian(adj_cluster, normalized=normalized))
        max_mode_idx = cluster_size - 1

        for mode_idx in range(num_modes):
            eigvec = eigvecs[:, min(mode_idx, max_mode_idx)]

            # Fix eigenvector sign ambiguity to keep deterministic output.
            if eigvec[0] < 0:
                eigvec = -eigvec

            theta[node_indices, mode_idx * num_clusters + cluster_idx] = eigvec

    return theta


def _cluster_from_adj(adj_np: np.ndarray, k: int) -> Tuple[np.ndarray, int]:
    num_nodes = adj_np.shape[0]
    actual_k = max(1, min(k, num_nodes))

    if actual_k == 1:
        return np.zeros(num_nodes, dtype=np.int64), actual_k

    # When k >= N, using one cluster per node is equivalent and avoids
    # SpectralClustering warnings about eigensolver fallbacks.
    if actual_k >= num_nodes:
        return np.arange(num_nodes, dtype=np.int64), num_nodes

    sc = SpectralClustering(n_clusters=actual_k, affinity="precomputed", n_init=10)
    with warnings.catch_warnings():
        warnings.filterwarnings(
            "ignore",
            message=".*not fully connected.*",
            category=UserWarning,
        )
        sc.fit(adj_np)
    return sc.labels_.astype(np.int64), actual_k


def _select_from_dense_adjacency(
    adj_dense: Tensor,
    k: int,
    num_modes: int,
    normalized: bool,
    num_classes: Optional[int] = None,
) -> Tuple[Tensor, Tensor]:
    adj_np = adj_dense.cpu().numpy()
    cluster_labels, actual_k = _cluster_from_adj(adj_np, k)

    s_num_classes = actual_k if num_classes is None else num_classes
    cluster_index = torch.as_tensor(
        cluster_labels, dtype=torch.long, device=adj_dense.device
    )
    s = F.one_hot(cluster_index, num_classes=s_num_classes).to(dtype=adj_dense.dtype)

    theta_np = build_pooling_matrix(
        adj_np=adj_np,
        cluster_labels=cluster_labels,
        num_modes=num_modes,
        normalized=normalized,
        expected_num_clusters=num_classes,
    )
    theta = torch.as_tensor(theta_np, dtype=adj_dense.dtype, device=adj_dense.device)
    return s, theta


def eigenpool_select(
    edge_index: Adj,
    k: int,
    edge_weight: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    num_nodes: Optional[int] = None,
    fixed_k: bool = False,
    s_inv_op: SinvType = "transpose",
    num_modes: int = 5,
    normalized: bool = True,
) -> SelectOutput:
    r"""Compute EigenPool assignments and eigenvector pooling matrices.

    Given a graph with :math:`N` nodes, this function computes:

    + a hard assignment matrix
      :math:`\mathbf{S} \in \{0,1\}^{N \times K}` via spectral clustering;
    + an eigenvector pooling matrix
      :math:`\boldsymbol{\Theta} \in \mathbb{R}^{N \times (K\cdot H)}`.

    For consistency with the connector notation, :math:`\boldsymbol{\Omega}` used in
    :class:`~tgp.connect.EigenPoolConnect` is the same matrix as
    :math:`\mathbf{S}`.

    Args:
        edge_index (~torch_geometric.typing.Adj):
            Graph connectivity in edge index or dense adjacency format.
        k (int):
            Number of clusters (supernodes).
        edge_weight (~torch.Tensor, optional):
            Edge weights associated with :obj:`edge_index`. (default: :obj:`None`)
        batch (~torch.Tensor, optional):
            Batch vector :math:`\mathbf{b} \in \{0,\dots,B-1\}^N` for multi-graph inputs.
            (default: :obj:`None`)
        num_nodes (int, optional):
            Total number of nodes. Useful when :obj:`edge_index` is empty.
            (default: :obj:`None`)
        fixed_k (bool, optional):
            If :obj:`True`, always use exactly :obj:`k` output clusters
            (allowing empty clusters). If :obj:`False`, single-graph mode
            may reduce the effective number of clusters for tiny graphs.
            (default: :obj:`False`)
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            Operation used to compute :math:`\mathbf{S}_\text{inv}` stored in
            :class:`~tgp.select.SelectOutput`. (default: :obj:`"transpose"`)
        num_modes (int, optional):
            Number of eigenvector modes :math:`H` used to build
            :math:`\boldsymbol{\Theta}`. (default: :obj:`5`)
        normalized (bool, optional):
            If :obj:`True`, uses the normalized Laplacian for eigenvectors.
            (default: :obj:`True`)

    Returns:
        ~tgp.select.SelectOutput:
            Selection output with:

            - :obj:`s`: dense one-hot assignment matrix :math:`[N, K]`
            - :obj:`theta`: pooling matrix :math:`\boldsymbol{\Theta}` (or a list
              of per-graph matrices for multi-graph batches)
    """
    edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
        edge_index, edge_weight
    )

    device = edge_index_conv.device
    # Infer node count from connectivity, then merge with explicit hints.
    # This is important for edgeless graphs where edge_index carries no nodes.
    inferred_num_nodes = (
        int(edge_index_conv.max().item()) + 1 if edge_index_conv.numel() > 0 else 0
    )
    if batch is not None:
        inferred_num_nodes = max(inferred_num_nodes, batch.size(0))
    if num_nodes is None:
        num_nodes = inferred_num_nodes
    else:
        num_nodes = max(int(num_nodes), inferred_num_nodes)

    if num_nodes == 0:
        raise ValueError("Cannot perform eigenpool selection on empty graph.")

    is_multi_graph = (
        batch is not None
        and batch.numel() > 0
        and int(batch.min().item()) != int(batch.max().item())
    )

    # Single graph case: compute one assignment and pooling matrix for the entire graph.
    if not is_multi_graph:
        adj_dense = to_dense_adj(
            edge_index_conv, edge_attr=edge_weight_conv, max_num_nodes=num_nodes
        ).squeeze(0)
        s, theta = _select_from_dense_adjacency(
            adj_dense=adj_dense,
            k=k,
            num_modes=num_modes,
            normalized=normalized,
            # In pre-coarsening we may need a fixed width K across samples
            # to make downstream collation of dense assignments deterministic.
            num_classes=k if fixed_k else None,
        )
        return SelectOutput(
            s=s,
            s_inv_op=s_inv_op,
            batch=batch,
            theta=theta,
        )

    # Multi-graph batch: process each graph separately and return a list of theta matrices.
    batch_size = int(batch.max().item()) + 1
    num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
    # Prefix sums let us convert global node ids to local graph ids when needed.
    node_ptr = torch.cat(
        [num_nodes_per_graph.new_zeros(1), num_nodes_per_graph.cumsum(0)], dim=0
    )
    if edge_index_conv.numel() == 0:
        edge_batch = batch.new_empty((0,), dtype=torch.long)
    else:
        # In COO edge_index, source node graph id is enough because edges do not cross graphs.
        edge_batch = batch[edge_index_conv[0]]

    if edge_weight_conv is None:
        out_dtype = torch.get_default_dtype()
    else:
        out_dtype = edge_weight_conv.dtype

    s_list, theta_list = [], []

    for i, n_nodes_tensor in enumerate(num_nodes_per_graph):
        n_nodes = int(n_nodes_tensor.item())
        if n_nodes == 0:
            # Preserve graph slots for empty graphs to keep list/batch alignment.
            s_list.append(torch.zeros((0, k), dtype=out_dtype, device=device))
            theta_list.append(
                torch.zeros((0, k * num_modes), dtype=out_dtype, device=device)
            )
            continue

        edge_mask = edge_batch == i
        edge_index_i = edge_index_conv[:, edge_mask]
        if edge_weight_conv is None:
            edge_weight_i = None
        else:
            edge_weight_i = edge_weight_conv[edge_mask]

        if edge_index_i.numel() == 0:
            # Graph has nodes but no edges: use all-zero adjacency.
            adj_dense = torch.zeros((n_nodes, n_nodes), dtype=out_dtype, device=device)
        else:
            node_start = int(node_ptr[i].item())
            # Convert global node indices to per-graph local indexing [0, n_nodes).
            edge_index_i = edge_index_i - node_start
            adj_dense = to_dense_adj(
                edge_index_i,
                edge_attr=edge_weight_i,
                max_num_nodes=n_nodes,
            ).squeeze(0)

        s, theta = _select_from_dense_adjacency(
            adj_dense=adj_dense,
            k=k,
            num_modes=num_modes,
            normalized=normalized,
            # Batched mode always uses fixed K so all graphs can be concatenated.
            num_classes=k,
        )
        s_list.append(s.to(dtype=out_dtype))
        theta_list.append(theta.to(dtype=out_dtype))

    s = (
        torch.cat(s_list, dim=0)
        if s_list
        else torch.zeros((0, k), device=device, dtype=out_dtype)
    )
    return SelectOutput(
        s=s,
        s_inv_op=s_inv_op,
        batch=batch,
        theta=theta_list,
    )


class EigenPoolSelect(Select):
    r"""The :math:`\texttt{select}` operator for EigenPooling.

    This operator performs spectral clustering on the adjacency matrix to build
    a dense assignment matrix
    :math:`\mathbf{S} \in \{0,1\}^{N \times K}` and the eigenvector pooling
    matrix :math:`\boldsymbol{\Theta} \in \mathbb{R}^{N \times (K\cdot H)}` used
    by the EigenPooling reduce/lift steps.

    The same assignment matrix may also be denoted as
    :math:`\boldsymbol{\Omega}` in connectivity formulas; in this implementation
    :math:`\boldsymbol{\Omega} = \mathbf{S}`.

    Args:
        k (int):
            Number of clusters (supernodes).
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            Operation used to compute :math:`\mathbf{S}_\text{inv}` from
            :math:`\mathbf{S}`. (default: :obj:`"transpose"`)
        num_modes (int, optional):
            Number of eigenvector modes :math:`H`. (default: :obj:`5`)
        normalized (bool, optional):
            If :obj:`True`, use the normalized Laplacian. (default: :obj:`True`)
    """

    is_dense: bool = True

    def __init__(
        self,
        k: int,
        s_inv_op: SinvType = "transpose",
        num_modes: int = 5,
        normalized: bool = True,
    ):
        super().__init__()
        self.k = k
        self.s_inv_op = s_inv_op
        self.num_modes = num_modes
        self.normalized = normalized

    def forward(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor, optional):
                Node features (unused; kept for API compatibility). (default: :obj:`None`)
            edge_index (~torch_geometric.typing.Adj, optional):
                Graph connectivity.
            edge_weight (~torch.Tensor, optional):
                Edge weights associated with :obj:`edge_index`. (default: :obj:`None`)
            batch (~torch.Tensor, optional):
                Batch vector for multi-graph inputs. (default: :obj:`None`)
            num_nodes (int, optional):
                Number of nodes in the graph. (default: :obj:`None`)

        Returns:
            ~tgp.select.SelectOutput:
                Selection output with:

                - :obj:`s`: assignment matrix :math:`\mathbf{S}`
                - :obj:`theta`: pooling matrix :math:`\boldsymbol{\Theta}`
        """
        return eigenpool_select(
            edge_index=edge_index,
            k=self.k,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            fixed_k=bool(kwargs.pop("fixed_k", False)),
            s_inv_op=self.s_inv_op,
            num_modes=self.num_modes,
            normalized=self.normalized,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"
