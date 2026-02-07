import warnings
from typing import Optional, Tuple

import numpy as np
import torch
import torch.nn.functional as F
from scipy import sparse as sp
from sklearn.cluster import SpectralClustering
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils import to_dense_adj, unbatch, unbatch_edge_index

from tgp.select.base_select import Select, SelectOutput
from tgp.utils.ops import connectivity_to_edge_index
from tgp.utils.typing import SinvType


def laplacian(adj: np.ndarray, normalized: bool = True) -> np.ndarray:
    r"""Computes the graph Laplacian from a dense adjacency matrix.

    Given an adjacency matrix :math:`\mathbf{A}`, this function returns either:

    .. math::
        \mathbf{L} = \mathbf{D} - \mathbf{A} \quad \text{(unnormalized)}

    or the symmetric normalized Laplacian:

    .. math::
        \mathbf{L} = \mathbf{I} - \mathbf{D}^{-1/2}\mathbf{A}\mathbf{D}^{-1/2}.

    Args:
        adj (np.ndarray): Dense adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`.
        normalized (bool, optional): If :obj:`True`, returns the normalized Laplacian.
            (default: :obj:`True`)

    Returns:
        np.ndarray: The Laplacian matrix :math:`\mathbf{L} \in \mathbb{R}^{N \times N}`.
    """
    if sp.issparse(adj):
        d = np.array(adj.sum(axis=0)).flatten()
    else:
        d = np.array(adj.sum(axis=0)).flatten()

    if not normalized:
        D = sp.diags(d, 0) if sp.issparse(adj) else np.diag(d)
        L = D - adj
    else:
        d = d + np.spacing(np.array(0, dtype=adj.dtype))
        d_inv_sqrt = 1.0 / np.sqrt(d)
        D_inv_sqrt = (
            sp.diags(d_inv_sqrt, 0) if sp.issparse(adj) else np.diag(d_inv_sqrt)
        )
        Identity = (
            sp.eye(d.size, dtype=adj.dtype) if sp.issparse(adj) else np.eye(d.size)
        )
        L = Identity - D_inv_sqrt @ adj @ D_inv_sqrt

    return L


def eigenvectors(L: np.ndarray) -> Tuple[np.ndarray, np.ndarray]:
    r"""Computes eigenvalues and eigenvectors of a Laplacian matrix.

    Args:
        L (np.ndarray): Laplacian matrix :math:`\mathbf{L} \in \mathbb{R}^{N \times N}`.

    Returns:
        Tuple[np.ndarray, np.ndarray]:
            - Eigenvalues :math:`\boldsymbol{\lambda} \in \mathbb{R}^{N}`
            - Eigenvectors :math:`\mathbf{U} \in \mathbb{R}^{N \times N}` (columns)
    """
    if sp.issparse(L):
        L = L.toarray()
    lamb, U = np.linalg.eigh(L)
    return lamb, U


def build_pooling_matrix(
    adj_np: np.ndarray,
    cluster_labels: np.ndarray,
    num_modes: int,
    normalized: bool = True,
    expected_num_clusters: Optional[int] = None,
) -> np.ndarray:
    r"""Builds the eigenvector-based pooling matrix :math:`\boldsymbol{\Theta}`.

    For each cluster, we compute the Laplacian of the induced subgraph and use
    the first :math:`H` eigenvectors as pooling modes. The resulting matrix is
    assembled as:

    .. math::
        \boldsymbol{\Theta} = [\boldsymbol{\Theta}^{(1)} \; \cdots \; \boldsymbol{\Theta}^{(H)}]
        \in \mathbb{R}^{N \times (K\cdot H)},

    where :math:`\boldsymbol{\Theta}^{(h)}` places the :math:`h`-th eigenvector
    of each cluster on the rows corresponding to its nodes and zeros elsewhere.

    Notes:
        - For singleton clusters, the assignment defaults to the (possibly zero)
          self-loop value in the adjacency matrix.
        - When :obj:`expected_num_clusters` is provided (batched case), the
          output has a fixed :math:`K` columns even if some clusters are empty.

    Args:
        adj_np (np.ndarray): Dense adjacency matrix :math:`\mathbf{A} \in \mathbb{R}^{N \times N}`.
        cluster_labels (np.ndarray): Cluster assignment vector of length :math:`N`.
        num_modes (int): Number of eigenvector modes :math:`H`.
        normalized (bool, optional): If :obj:`True`, uses the normalized Laplacian.
            (default: :obj:`True`)
        expected_num_clusters (int, optional): Fixed number of clusters :math:`K`
            to allocate in the output (used for batched inputs).

    Returns:
        np.ndarray: The pooling matrix :math:`\boldsymbol{\Theta} \in \mathbb{R}^{N \times (K\cdot H)}`.
    """
    num_nodes = adj_np.shape[0]

    clusters = {}
    for node_idx, label in enumerate(cluster_labels):
        if label not in clusters:
            clusters[label] = []
        clusters[label].append(node_idx)

    if expected_num_clusters is not None:
        num_clusters = expected_num_clusters
    else:
        num_clusters = len(clusters)

    if expected_num_clusters is not None:
        label_to_idx = {label: label for label in clusters.keys()}
    else:
        label_to_idx = {label: idx for idx, label in enumerate(sorted(clusters.keys()))}

    adj_per_cluster = {}
    for label, node_list in clusters.items():
        node_arr = np.array(node_list)
        adj_per_cluster[label] = adj_np[np.ix_(node_arr, node_arr)]

    pooling_matrices = [np.zeros((num_nodes, num_clusters)) for _ in range(num_modes)]

    for label, node_list in clusters.items():
        adj_cluster = adj_per_cluster[label]
        cluster_size = len(node_list)
        cluster_idx = label_to_idx[label]

        if cluster_size > 1:
            L_cluster = laplacian(adj_cluster, normalized=normalized)
            _, U_cluster = eigenvectors(L_cluster)

            for mode_idx in range(num_modes):
                eig_idx = min(mode_idx, cluster_size - 1)
                eigvec = U_cluster[:, eig_idx]

                if eigvec[0] < 0:
                    eigvec = -eigvec

                for local_idx, global_idx in enumerate(node_list):
                    pooling_matrices[mode_idx][global_idx, cluster_idx] = eigvec[
                        local_idx
                    ]
        else:
            node_idx = node_list[0]
            value = float(adj_cluster.reshape(-1)[0]) if adj_cluster.size else 0.0
            for mode_idx in range(num_modes):
                pooling_matrices[mode_idx][node_idx, cluster_idx] = value

    theta = np.concatenate(pooling_matrices, axis=1)
    return theta


def eigenpool_select(
    edge_index: Adj,
    k: int,
    edge_weight: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    s_inv_op: SinvType = "transpose",
    num_modes: int = 5,
    normalized: bool = True,
) -> SelectOutput:
    r"""Compute EigenPool selection via spectral clustering.

    This function partitions the graph into :math:`K` clusters using spectral
    clustering on the (weighted) adjacency matrix and returns a dense assignment
    matrix :math:`\mathbf{S} \in \mathbb{R}^{N \times K}` along with the
    eigenvector-based pooling matrix :math:`\boldsymbol{\Theta}` used by
    :class:`~tgp.reduce.EigenPoolReduce` and :class:`~tgp.lift.EigenPoolLift`.

    For batched inputs, clustering is performed independently for each graph.

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
        s_inv_op (~tgp.typing.SinvType, optional):
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
            The selection output containing:

            - :obj:`s`: a dense one-hot assignment matrix :math:`[N, K]`
            - :obj:`theta`: pooling matrix :math:`\boldsymbol{\Theta}` (or a list
              of per-graph matrices in the batched case)

    Example:
        >>> from tgp.select.eigenpool_select import eigenpool_select
        >>> so = eigenpool_select(edge_index, k=4, num_modes=3)
        >>> so.s.shape
        torch.Size([N, 4])
        >>> so.theta.shape
        torch.Size([N, 12])
    """
    edge_index_conv, edge_weight_conv = connectivity_to_edge_index(
        edge_index, edge_weight
    )

    device = edge_index_conv.device
    num_nodes = (
        int(edge_index_conv.max().item()) + 1 if edge_index_conv.numel() > 0 else 0
    )
    if batch is not None:
        num_nodes = max(num_nodes, batch.size(0))

    if num_nodes == 0:
        raise ValueError("Cannot perform eigenpool selection on empty graph.")

    def _cluster_from_adj(
        adj_np: np.ndarray, num_nodes_i: int
    ) -> Tuple[np.ndarray, int]:
        actual_k = min(k, num_nodes_i)
        if actual_k < 1:
            actual_k = 1

        # If k >= N, SpectralClustering triggers a warning and falls back to a dense eigensolver.
        # In this case, use the trivial assignment where each node is its own cluster.
        if num_nodes_i > 1 and actual_k >= num_nodes_i:
            return np.arange(num_nodes_i, dtype=np.int64), num_nodes_i

        if actual_k == 1:
            return np.zeros(num_nodes_i, dtype=np.int64), actual_k

        sc = SpectralClustering(
            n_clusters=actual_k,
            affinity="precomputed",
            n_init=10,
        )
        with warnings.catch_warnings():
            warnings.filterwarnings(
                "ignore",
                message=".*not fully connected.*",
                category=UserWarning,
            )
            sc.fit(adj_np)
        return sc.labels_, actual_k

    if (
        batch is None
        or batch.numel() == 0
        or int(batch.min().item()) == int(batch.max().item())
    ):
        adj_dense = to_dense_adj(
            edge_index_conv, edge_attr=edge_weight_conv, max_num_nodes=num_nodes
        ).squeeze(0)
        dtype = adj_dense.dtype
        adj_np = adj_dense.cpu().numpy()
        cluster_labels, actual_k = _cluster_from_adj(adj_np, num_nodes)

        cluster_index = torch.tensor(cluster_labels, dtype=torch.long, device=device)
        s = F.one_hot(cluster_index, num_classes=actual_k).to(dtype=dtype)
        theta_np = build_pooling_matrix(
            adj_np=adj_np,
            cluster_labels=cluster_labels,
            num_modes=num_modes,
            normalized=normalized,
        )
        theta = torch.tensor(theta_np, dtype=dtype, device=device)
        return SelectOutput(
            s=s,
            s_inv_op=s_inv_op,
            batch=batch,
            theta=theta,
        )

    batch_size = int(batch.max().item()) + 1
    num_nodes_per_graph = torch.bincount(batch, minlength=batch_size)
    unbatched_edges = unbatch_edge_index(edge_index_conv, batch=batch)

    if edge_weight_conv is None:
        unbatched_weights = [None] * batch_size
        dtype = torch.get_default_dtype()
    else:
        edge_batch = batch[edge_index_conv[0]]
        unbatched_weights = unbatch(edge_weight_conv.view(-1), batch=edge_batch)
        dtype = edge_weight_conv.dtype

    s_list = []
    theta_list = []
    for i in range(batch_size):
        n_nodes = int(num_nodes_per_graph[i].item())
        if n_nodes == 0:
            s_list.append(torch.zeros((0, k), dtype=dtype, device=device))
            theta_list.append(
                torch.zeros((0, k * num_modes), dtype=dtype, device=device)
            )
            continue

        adj_dense = to_dense_adj(
            unbatched_edges[i],
            edge_attr=unbatched_weights[i],
            max_num_nodes=n_nodes,
        ).squeeze(0)
        adj_np = adj_dense.cpu().numpy()
        cluster_labels, _ = _cluster_from_adj(adj_np, n_nodes)
        cluster_index = torch.tensor(cluster_labels, dtype=torch.long, device=device)
        s_list.append(F.one_hot(cluster_index, num_classes=k).to(dtype=dtype))
        theta_np = build_pooling_matrix(
            adj_np=adj_np,
            cluster_labels=cluster_labels,
            num_modes=num_modes,
            normalized=normalized,
            expected_num_clusters=k,
        )
        theta_list.append(torch.tensor(theta_np, dtype=dtype, device=device))

    s = (
        torch.cat(s_list, dim=0)
        if s_list
        else torch.zeros((0, k), device=device, dtype=dtype)
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
    a dense assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{N \times K}` and
    the eigenvector pooling matrix :math:`\boldsymbol{\Theta}` used by the
    EigenPooling reduce/lift steps.

    Args:
        k (int):
            Number of clusters (supernodes).
        s_inv_op (~tgp.typing.SinvType, optional):
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
                Graph connectivity. (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                Edge weights associated with :obj:`edge_index`. (default: :obj:`None`)
            batch (~torch.Tensor, optional):
                Batch vector for multi-graph inputs. (default: :obj:`None`)
            num_nodes (int, optional):
                Number of nodes in the graph. (default: :obj:`None`)

        Returns:
            ~tgp.select.SelectOutput:
                Selection output with :obj:`s` and :obj:`theta`.
        """
        if edge_index is None:
            raise ValueError("edge_index is required for EigenPoolSelect.")

        return eigenpool_select(
            edge_index=edge_index,
            k=self.k,
            edge_weight=edge_weight,
            batch=batch,
            s_inv_op=self.s_inv_op,
            num_modes=self.num_modes,
            normalized=self.normalized,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(k={self.k})"
