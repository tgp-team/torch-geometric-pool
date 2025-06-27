import math
from typing import List, Literal, Optional

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Distribution, kl_divergence
from torch_geometric.utils import scatter
from torch_sparse import SparseTensor

from tgp import eps
from tgp.utils import rank3_diag, rank3_trace

BatchReductionType = Literal["mean", "sum"]


def _batch_reduce_loss(loss: Tensor, batch_reduction: BatchReductionType) -> Tensor:
    if batch_reduction == "mean":
        return torch.mean(loss)
    if batch_reduction == "sum":
        return torch.sum(loss)
    raise ValueError(
        f"Batch reduction {batch_reduction} not allowed, must be one of ['mean', 'sum']."
    )


def mincut_loss(
    adj: Tensor,
    S: Tensor,
    adj_pooled: Tensor,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary mincut loss used by :class:`~tgp.poolers.MinCutPooling` operator
    from the paper `"Spectral Clustering in Graph Neural Networks for Graph Pooling"
    <https://arxiv.org/abs/1907.00481>`_ (Bianchi et al., ICML 2020).

    The loss is computed as

    .. math::
        \mathcal{L}_\text{CUT} = - \frac{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{A}
        \mathbf{S})} {\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{D}
        \mathbf{S})},

    where

    + :math:`\mathbf{A}` is the adjacency matrix,
    + :math:`\mathbf{S}` is the dense cluster assignment matrix,
    + :math:`\mathbf{D} = \mathrm{diag}(\mathbf{A}^{\top}\mathbf{1})` is the degree
      matrix.

    Args:
        adj (~torch.Tensor): The adjacency matrix of shape
            :math:`(B, N, N)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, used to compute :math:`\mathbf{D}`.
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`K` is the number of clusters.
        adj_pooled (~torch.Tensor): The pooled adjacency matrix :math:`\mathbf{S}^{\top}
            \mathbf{A}\mathbf{S}` of shape :math:`(B, K, K)`.
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The mincut loss.
    """
    num = rank3_trace(adj_pooled)
    d_flat = adj.sum(-1)
    d = rank3_diag(d_flat)
    den = rank3_trace(torch.matmul(torch.matmul(S.transpose(-2, -1), d), S))
    cut_loss = -(num / den)
    return _batch_reduce_loss(cut_loss, batch_reduction)


def orthogonality_loss(
    S: Tensor, batch_reduction: BatchReductionType = "mean"
) -> Tensor:
    r"""Auxiliary orthogonality loss used by :class:`~tgp.poolers.MinCutPooling`
    operator from the paper `"Spectral Clustering in Graph Neural Networks for Graph
    Pooling" <https://arxiv.org/abs/1907.00481>`_ (Bianchi et al., ICML 2020).

    The loss is computed as

    .. math::
        \mathcal{L}_O = {\left\| \frac{\mathbf{S}^{\top} \mathbf{S}}
        {{\|\mathbf{S}^{\top} \mathbf{S}\|}_F} -\frac{\mathbf{I}_K}{\sqrt{K}}
        \right\|}_F,

    where

    + :math:`\mathbf{S}` is the dense cluster assignment matrix,
    + :math:`\mathbf{I}_K` is the identity matrix of size :math:`K`,
    + :math:`K` is the number of clusters.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The orthogonality loss.
    """
    STS = torch.matmul(S.transpose(-2, -1), S)
    STS_term = STS / torch.norm(STS, dim=(-2, -1), keepdim=True)
    k = S.size(-1)
    id_k = torch.eye(k, device=S.device, dtype=S.dtype) / math.sqrt(k)
    ortho_loss = torch.norm(STS_term - id_k, dim=(-2, -1))
    return _batch_reduce_loss(ortho_loss, batch_reduction)


def hosc_orthogonality_loss(
    S: Tensor,
    mask: Optional[Tensor] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary orthogonality loss used by :class:`~tgp.poolers.HOSCPooling`
    operator from the paper `"Higher-order Clustering and Pooling for Graph Neural Networks"
    <https://arxiv.org/abs/2209.03473>`_ (Duval & Malliaros, CIKM 2022).

    The loss is computed as

    .. math::
        \mathcal{L}_\text{HO} = \frac{1}{\sqrt{K}-1} \bigg( \sqrt{K} - \frac{1}{\sqrt{N}}\sum_{j=1}^K ||\mathbf{S}_{*j}||_F\bigg),

    where

    + :math:`N` is the number of nodes,
    + :math:`K` is the number of clusters,
    + :math:`\mathbf{S}_{*j}` is the :math:`j`-th column of the cluster assignment matrix :math:`\mathbf{S}`.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The orthogonality loss.
    """
    _, num_nodes, num_clusters = S.size()
    norm = torch.norm(S, p="fro", dim=-2).sum(dim=-1)
    sqrt_k = math.sqrt(num_clusters)
    sqrt_nodes = mask.sum(1).sqrt() if mask is not None else math.sqrt(num_nodes)
    ortho_num = -norm / sqrt_nodes + sqrt_k
    ortho_loss = ortho_num / (sqrt_k - 1)
    return _batch_reduce_loss(ortho_loss, batch_reduction)


def link_pred_loss(S: Tensor, adj: Tensor, normalize_loss: bool = True) -> Tensor:
    r"""Auxiliary link prediction loss used by :class:`~tgp.poolers.DiffPool`
    operator from the paper `"Hierarchical Graph Representation Learning with
    Differentiable Pooling" <https://arxiv.org/abs/1806.08804>`_ (Ying et al., NeurIPS 2018).

    The loss is computed as

    .. math::
        \mathcal{L}_{LP} = {\| \mathbf{A} -
        \mathrm{softmax}(\mathbf{S}) {\mathrm{softmax}(\mathbf{S})}^{\top}
        \|}_F,

    where

    + :math:`\mathbf{A}` is the adjacency matrix,
    + :math:`\mathbf{S}` is the dense cluster assignment matrix.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        adj (~torch.Tensor): The adjacency matrix of shape
            :math:`(B, N, N)`.
        normalize_loss (bool, optional): If set to :obj:`True`, the loss will be
            normalized by the number of elements in the adjacency matrix.
            (default: :obj:`True`)

    Returns:
        ~torch.Tensor: The link prediction loss.
    """
    ss = torch.matmul(S, S.transpose(1, 2))
    link_loss = adj - ss
    link_loss = torch.norm(link_loss, p=2)
    if normalize_loss is True:
        link_loss = link_loss / adj.numel()
    return link_loss


def entropy_loss(S: Tensor, batch_reduction: BatchReductionType = "mean") -> Tensor:
    r"""Auxiliary entropy regularization loss used by :class:`~tgp.poolers.DiffPool`
    operator from the paper `"Hierarchical Graph Representation Learning with
    Differentiable Pooling" <https://arxiv.org/abs/1806.08804>`_ (Ying et al., NeurIPS 2018).

    The loss is computed as

    .. math::
        \mathcal{L}_E = \frac{1}{N} \sum_{n=1}^N H(\mathbf{S}_n),

    where

    + :math:`\mathbf{S}` is the dense cluster assignment matrix,
    + :math:`N` is the number of nodes,
    + :math:`H(\cdot)` is the entropy function.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)` where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The entropy regularization loss.
    """
    entropy = -S * torch.log(S + eps)
    entropy = torch.sum(entropy, dim=-1)
    return _batch_reduce_loss(entropy, batch_reduction)


def totvar_loss(
    S: Tensor, adj: Tensor, batch_reduction: BatchReductionType = "mean"
) -> Tensor:
    r"""The total variation regularization loss used by
    :class:`~tgp.poolers.AsymCheegerCutPooling` operator from the paper
    `"Total Variation Graph Neural Networks" <https://arxiv.org/abs/2211.06218>`_
    (Hansen & Bianchi, ICML 2023).

    The loss is computed as

    .. math::
        \mathcal{L}_\text{GTV} = \frac{\mathcal{L}_\text{GTV}^*}{2E} \in [0, 1],

    with the total variation regularization loss defined as

    .. math::
        \mathcal{L}_\text{GTV}^* = \displaystyle\sum_{k=1}^K\sum_{i=1}^N \sum_{j=i}^N a_{i,j} |s_{i,k} - s_{j,k}|.

    where

    + :math:`N` is the number of vertices,
    + :math:`K` is the number of clusters,
    + :math:`a_{i,j}` is the entry :math:`(i,j)` of the adjacency matrix,
    + :math:`s_{i,k}` is the assignment of vertex :math:`i` to cluster :math:`k`,
    + :math:`E` is the number of edges.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)` where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        adj (~torch.Tensor): The adjacency matrix of shape
            :math:`(B, N, N)`.
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The total variation regularization loss.
    """
    l1_norm = torch.sum(torch.abs(S[..., None, :] - S[:, None, ...]), dim=-1)
    loss = torch.sum(adj * l1_norm, dim=(-1, -2))
    n_edges = torch.count_nonzero(adj, dim=(-1, -2))
    loss *= 1 / (2 * n_edges)
    return _batch_reduce_loss(loss, batch_reduction)


def asym_norm_loss(
    S: Tensor, k: int, batch_reduction: BatchReductionType = "mean"
) -> Tensor:
    r"""Auxiliary asymmetrical norm term used by :class:`~tgp.poolers.AsymCheegerCutPooling`
    operator from the paper `"Total Variation Graph Neural Networks"
    <https://arxiv.org/abs/2211.06218>`_ (Hansen & Bianchi, ICML 2023).

    This term, :math:`\mathcal{L}_{\text{AN}}`, encourages balanced partitions of the graph by penalizing
    large deviations between each assignment vector and its :math:`\rho`-quantile. It is defined as

    .. math::
        \mathcal{L}_{\text{AN}} = \frac{\beta - \mathcal{L}^*_{\text{AN}}}{\beta} \in [0, 1],

    where

    .. math::
        \mathcal{L}^*_{\text{AN}} = \sum_{k=1}^{K} \bigl\|\mathbf{S}_{:,k} \;-\; \mathrm{quant}_\rho\bigl(\mathbf{S}_{:,k}\bigr)\bigr\|_{1,\rho}.

    In this formulation:

    + :math:`\mathbf{S}` is the cluster dense assignment matrix and :math:`\mathbf{S}_{:,k}`
      denotes the :math:`k`-th column of :math:`\mathbf{S}`, i.e., the
      assignments for cluster :math:`k` across all nodes.
    + :math:`\mathrm{quant}_\rho(\mathbf{S}_{:,k})` extracts the :math:`\rho`-quantile of
      :math:`\mathbf{S}_{:,k}`, where :math:`\rho` is a balancing parameter typically set to :math:`K-1`.
    + :math:`\|\cdot\|_{1,\rho}` is the asymmetric :math:`\ell_1` norm:
      :math:`\|\mathbf{x}\|_{1,\rho} = \sum_{i=1}^N |x_i|_{\rho},\,
      |x_i|_{\rho} = \rho x_i \,\text{if } x_i \ge 0,\text{ and } -x_i \text{ if } x_i < 0.`
    + :math:`\beta` is a normalization term ensuring that :math:`\mathcal{L}_{\text{AN}}` stays in :math:`[0,1]`.
      When :math:`\rho = K-1`, :math:`\beta = N\rho`. For other values of :math:`\rho`,
      :math:`\beta = N\rho \min\!\bigl(1, \frac{K}{\rho+1}\bigr)`.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)` where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is
            the number of clusters.
        k (int): The number of clusters (:math:`K`). This is used
            internally to set :math:`\rho = K - 1` if no other
            value of :math:`\rho` is explicitly chosen.
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The asymmetrical norm regularization loss.
    """
    n_nodes = S.size()[-2]

    # k-quantile
    idx = int(math.floor(n_nodes / k))
    quant = torch.sort(S, dim=-2, descending=True)[0][:, idx, :]  # shape [B, K]

    # Asymmetric l1-norm
    loss = S - torch.unsqueeze(quant, dim=1)
    loss = (loss >= 0) * (k - 1) * loss + (loss < 0) * (-loss)
    loss = torch.sum(loss, dim=(-1, -2))  # shape [B]
    loss = 1 / (n_nodes * (k - 1)) * (n_nodes * (k - 1) - loss)
    return _batch_reduce_loss(loss, batch_reduction)


def just_balance_loss(
    S: Tensor,
    mask: Optional[Tensor] = None,
    normalize_loss: bool = True,
    num_nodes: Optional[int] = None,
    num_clusters: Optional[int] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary balance regularization loss used by
    :class:`~tgp.poolers.JustBalancePooling` operator from the paper
    `"Simplifying Clustering with Graph Neural Networks"
    <https://arxiv.org/abs/2207.08779>`_ (Bianchi, NLDL 2023).

    The loss is computed as

    .. math::
        \mathcal{L}_{B} = - \mathrm{Tr}(\sqrt{\mathbf{S}^{\top} \mathbf{S}}),

    where

    + :math:`\mathbf{S}` is the dense cluster assignment matrix,
    + :math:`\mathrm{Tr}(\cdot)` is the trace operator.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        normalize_loss (bool, optional): If set to :obj:`True`, the loss is
            normalized by the number of nodes :math:`N` and the number of clusters :math:`K`.
            (default: :obj:`True`)
        num_nodes (Optional[int]): The number of nodes in the graph. If not provided,
            it is inferred from the shape of :math:`\mathbf{S}`. (default: :obj:`None`)
        num_clusters (Optional[int]): The number of clusters in the graph. If not provided,
            it is inferred from the shape of :math:`\mathbf{S}`. (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The balance regularization loss.
    """
    if num_nodes is None:
        num_nodes = S.size(-2)
    if num_clusters is None:
        num_clusters = S.size(-1)

    ss = torch.matmul(S.transpose(1, 2), S)
    ss_sqrt = torch.sqrt(ss + eps)
    loss = -rank3_trace(ss_sqrt)
    if normalize_loss:
        if mask is None:
            loss = loss / torch.sqrt(torch.tensor(num_nodes * num_clusters))
        else:
            loss = loss / torch.sqrt(mask.sum() / mask.size(0) * num_clusters)

    return _batch_reduce_loss(loss, batch_reduction)


def spectral_loss(
    adj: Tensor,
    S: Tensor,
    adj_pooled: Tensor,
    mask: Optional[Tensor] = None,
    num_clusters: Optional[int] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary spectral regularization loss used by
    :class:`~tgp.poolers.DMoNPooling` operator from the paper
    `"Graph Clustering with Graph Neural Networks"
    <https://arxiv.org/abs/2006.16904>`_ (Tsitsulin et al., JMLR 2023).

    The loss is computed as

    .. math::
        \mathcal{L}_S = - \frac{1}{2m}
        \cdot{\mathrm{Tr}(\mathbf{S}^{\top} \mathbf{B} \mathbf{S})},

    where

    + :math:`\mathbf{B} = \mathbf{A} - \frac{\mathbf{d} \mathbf{d}^{\top}}{2m}`
      is the modularity matrix,
    + :math:`\mathbf{A}` is the adjacency matrix,
    + :math:`\mathbf{d}` is the degree vector,
    + :math:`m = \frac{1}{2} \sum_{i,j} A_{i,j}` is the total number of edges in the graph.

    Args:
        adj (~torch.Tensor): The adjacency matrix.
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        adj_pooled (~torch.Tensor): The pooled adjacency matrix.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        num_clusters (Optional[int]): The number of clusters in the graph. If not provided,
            it is inferred from the shape of :math:`\mathbf{S}`. (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The spectral regularization loss.
    """
    if num_clusters is None:
        num_clusters = S.size(-1)

    if mask is None:
        mask = torch.ones(S.size(0), S.size(1), dtype=torch.bool, device=S.device)

    degrees = torch.einsum("bnm->bn", adj)
    degrees = degrees * mask
    m = degrees.sum(-1) / 2
    m_expand = m.view(-1, 1, 1).expand(-1, num_clusters, num_clusters)
    ca = torch.einsum("bnk, bn -> bk", S, degrees)
    cb = torch.einsum("bn, bnk -> bk", degrees, S)
    normalizer = torch.einsum("bk, bm -> bkm", ca, cb) / 2 / m_expand
    decompose = adj_pooled - normalizer
    spectral_loss = -rank3_trace(decompose) / 2 / m
    return _batch_reduce_loss(spectral_loss, batch_reduction)


def cluster_loss(
    S: Tensor,
    mask: Optional[Tensor] = None,
    num_clusters: Optional[int] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary cluster regularization loss used by
    :class:`~tgp.poolers.DMoNPooling` operator from the paper
    `"Graph Clustering with Graph Neural Networks"
    <https://arxiv.org/abs/2006.16904>`_ (Tsitsulin et al., JMLR 2023).

    The loss is computed as

    .. math::
        \mathcal{L}_C = \frac{\sqrt{K}}{N}
        {\left\|\sum_{i=1}^{N} \mathbf{S}_i^{\top} \right\|}_F - 1,

    where

    + :math:`\mathbf{S}` is the dense cluster assignment matrix,
    + :math:`N` is the number of nodes,
    + :math:`K` is the number of clusters.

    Args:
        S (~torch.Tensor): The dense cluster assignment matrix of shape
            :math:`(B, N, K)`, where :math:`B` is the batch size,
            :math:`N` is the number of nodes, and :math:`K` is the number of clusters.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        num_clusters (Optional[int]): The number of clusters in the graph. If not provided,
            it is inferred from the shape of :math:`\mathbf{S}`. (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The cluster regularization loss.
    """
    if num_clusters is None:
        num_clusters = S.size(-1)

    if mask is None:
        mask = torch.ones(S.size(0), S.size(1), dtype=torch.bool, device=S.device)

    i_s = torch.eye(num_clusters).type_as(S)
    cluster_size = torch.einsum("ijk->ik", S)  # B x K
    cluster_loss = torch.norm(input=cluster_size, dim=1)
    cluster_loss = cluster_loss / mask.sum(dim=1) * torch.norm(i_s) - 1
    return _batch_reduce_loss(cluster_loss, batch_reduction)


def weighted_bce_reconstruction_loss(
    rec_adj: Tensor,
    adj: Tensor,
    mask: Optional[Tensor] = None,
    balance_links: bool = True,
    normalize_loss: bool = False,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Weighted binary cross-entropy reconstruction loss for adjacency matrices.

    This function computes the binary cross-entropy loss between a reconstructed
    adjacency matrix and the true adjacency matrix. When :obj:`balance_links` is :obj:`True`,
    it applies class-balancing weights to handle the imbalance between edges and
    non-edges in sparse graphs.

    The weighted BCE loss is computed as:

    .. math::
        \mathcal{L}_{\text{BCE}} = \text{BCE}(\mathbf{A}_{\text{rec}}, \mathbf{A}, \mathbf{W})

    where the weight matrix :math:`\mathbf{W}` is computed to balance positive and negative samples:

    .. math::
        W_{ij} = \frac{N^2}{n_{\text{edges}}} \cdot A_{ij} + \frac{N^2}{n_{\text{non-edges}}} \cdot (1 - A_{ij})

    with :math:`n_{\text{edges}} = \sum_{i,j} A_{ij}` and :math:`n_{\text{non-edges}} = N^2 - n_{\text{edges}}`.

    When :obj:`normalize_loss` is :obj:`True`, the loss is normalized by :math:`N^2` per graph:

    .. math::
        \mathcal{L}_{\text{normalized}} = \frac{\mathcal{L}_{\text{BCE}}}{N^2}

    Args:
        rec_adj (~torch.Tensor): The reconstructed adjacency matrix (logits) of shape
            :math:`(B, N, N)`, where :math:`B` is the batch size and :math:`N` is
            the number of nodes. Contains the predicted edge probabilities.
        adj (~torch.Tensor): The true adjacency matrix of shape :math:`(B, N, N)`.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        balance_links (bool, optional): Whether to apply class-balancing weights to handle
            edge/non-edge imbalance.
            (default: :obj:`True`)
        normalize_loss (bool, optional): Whether to normalize the loss by the square of the
            number of nodes per graph :math:`N^2`. This ensures consistent scaling across
            graphs of different sizes.
            (default: :obj:`False`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The weighted BCE reconstruction loss.
    """
    pos_weight = None
    if balance_links:
        # Calculate BCE weights to balance positive and negative samples
        if mask is not None:
            N = mask.sum(-1).view(-1, 1, 1)  # has shape B x 1 x 1
        else:
            N = adj.shape[-1]  # Number of nodes
        n_edges = torch.clamp(adj.sum([-1, -2]), min=1).view(
            -1, 1, 1
        )  # this is a vector of size B x 1 x 1
        n_not_edges = torch.clamp(N**2 - n_edges, min=1).view(
            -1, 1, 1
        )  # this is a vector of size B x 1 x 1
        # the clamp is needed to avoid zero division when we have all edges
        pos_weight = (N**2 / n_edges) * adj + (N**2 / n_not_edges) * (1 - adj)

    loss = F.binary_cross_entropy_with_logits(
        rec_adj, adj, weight=pos_weight, reduction="none"
    )

    # Apply mask if provided (create edge mask for adjacency matrices)
    if mask is not None and not torch.all(mask):
        # Create edge mask: (B, N) -> (B, N, N)
        edge_mask = torch.einsum("bn,bm->bnm", mask, mask)
        loss = loss * edge_mask

    # Sum over both spatial dimensions (always the same for adjacency matrices)
    loss = loss.sum((-1, -2))  # Sum over both spatial dimensions -> (B,)

    # Normalize by N^2 if requested
    if normalize_loss:
        if mask is not None:
            N_squared = mask.sum(-1) ** 2  # (B,)
        else:
            N_squared = adj.shape[-1] ** 2  # All nodes are valid
        loss = loss / N_squared

    return _batch_reduce_loss(loss, batch_reduction)


def kl_loss(
    q: Distribution,
    p: Distribution,
    mask: Optional[Tensor] = None,
    node_axis: Optional[int] = None,
    sum_axes: Optional[List[int]] = None,
    normalize_loss: bool = False,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Compute KL divergence between two distributions with flexible axis control.

    This function computes the KL divergence :math:`D_{KL}(q \parallel p)` between
    two distributions, with explicit control over which axes to sum and how to
    apply masking.

    .. math::
        D_{KL}(q \parallel p) = \mathbb{E}_{x \sim q}[\log q(x) - \log p(x)]

    When :obj:`normalize_loss` is True, the loss is normalized by :math:`N^2`:

    .. math::
        D_{KL,\text{normalized}} = \frac{D_{KL}(q \parallel p)}{N^2}

    Args:
        q (~torch.distributions.Distribution): The approximate posterior distribution.
        p (~torch.distributions.Distribution): The prior distribution.
        mask (Optional[~torch.Tensor]): A mask matrix
            :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
            the valid nodes for each graph. (default: :obj:`None`)
        node_axis (Optional[int]): The axis along which nodes are arranged.
            The mask will be applied along this axis. (default: :obj:`None`)
        sum_axes (Optional[list]): List of axes to sum over after masking but before
            final reduction. If :obj:`None`, sums over all axes except batch (axis 0).
            (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The KL divergence loss.

    Examples:
        >>> import torch
        >>> from torch.distributions import Beta
        >>> from tgp.utils.losses import kl_loss
        >>> # Example: Stick-breaking process in BNPool
        >>> # Shape: (B=2, N=4, K-1=3) for 4 nodes, max 4 clusters
        >>> alpha_sb = torch.ones(2, 4, 3) + 0.5  # Posterior Alpha parameters
        >>> beta_sb = torch.ones(2, 4, 3) + 1.0  # Posterior Beta parameters
        >>> q_sb = Beta(alpha_sb, beta_sb)  # Posterior distributions
        >>> # Prior: Beta(1, alpha_DP) for each stick-breaking fraction
        >>> alpha_prior = torch.ones(3)
        >>> beta_prior = torch.ones(3) * 2.0  # alpha_DP = 2.0
        >>> p_sb = Beta(alpha_prior, beta_prior)
        >>> # Node mask for variable-sized graphs
        >>> mask = torch.tensor(
        ...     [[True, True, True, False], [True, True, True, True]], dtype=torch.bool
        ... )
        >>> # Compute KL loss: sum over K-1 components, then over nodes
        >>> loss = kl_loss(q_sb, p_sb, mask=mask, node_axis=1, sum_axes=[2, 1])
    """
    loss = kl_divergence(q, p)

    # Store number of nodes for normalization (before summing changes the shape)
    if normalize_loss and node_axis is not None:
        if mask is not None:
            num_nodes = mask.sum(node_axis)  # (B,) - actual number of nodes per graph
        else:
            num_nodes = loss.shape[node_axis]  # All nodes are valid

    # Apply mask if provided
    if mask is not None and node_axis is not None:
        if not torch.all(mask):
            # Expand mask to match loss dimensions
            mask_shape = [1] * loss.dim()
            mask_shape[node_axis] = mask.shape[node_axis]
            if mask.dim() > 1:  # Handle batch dimension
                mask_shape[0] = mask.shape[0]
            expanded_mask = mask.view(mask_shape)
            loss = loss * expanded_mask

    # Sum over specified axes
    if sum_axes is not None:
        for axis in sorted(
            sum_axes, reverse=True
        ):  # Sum from last to first to maintain indices
            loss = loss.sum(dim=axis)
    else:
        # Default: sum over all non-batch axes
        sum_dims = list(range(1, loss.dim()))
        for axis in reversed(sum_dims):
            loss = loss.sum(dim=axis)

    # Normalize by N^2 if requested
    if normalize_loss:
        N_squared = num_nodes**2  # (B,) - use stored num_nodes
        loss = loss / N_squared

    return _batch_reduce_loss(loss, batch_reduction)


def cluster_connectivity_prior_loss(
    K: Tensor,
    K_mu: Tensor,
    K_var: Tensor,
    normalize_loss: bool = False,
    mask: Optional[Tensor] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Prior loss for cluster connectivity matrix in :class:`~tgp.poolers.BNPool`.

    This function computes the prior loss for the cluster connectivity matrix :math:`\mathbf{K}`,
    which regularizes the learned cluster-cluster connectivity probabilities
    towards a prior distribution. The prior loss is computed as the negative
    log-likelihood of a Gaussian prior:

    .. math::
        \mathcal{L}_{\mathbf{K}} = \frac{1}{2} \sum_{i,j} \frac{(K_{ij} - \mu_{ij})^2}{\sigma^2}

    where :math:`\mathbf{K} \in \mathbb{R}^{C \times C}` is the cluster connectivity matrix,
    :math:`\boldsymbol{\mu} \in \mathbb{R}^{C \times C}` is the prior mean matrix,
    and :math:`\sigma^2` is the prior variance.

    The prior mean :math:`\boldsymbol{\mu}` typically has the structure:

    .. math::
        \mu_{ij} = \begin{cases}
        \mu_{\text{diag}} & \text{if } i = j \text{ (within-cluster connectivity)} \\
        \mu_{\text{off}} & \text{if } i \neq j \text{ (between-cluster connectivity)}
        \end{cases}

    This structure encourages block-diagonal patterns in the reconstructed adjacency matrix
    :math:`\mathbf{A}_{\text{rec}} = \mathbf{S} \mathbf{K} \mathbf{S}^{\top}`, promoting well-separated clusters.

    When :obj:`normalize_loss` is :obj:`True`, the loss is normalized by :math:`N^2` per graph:

    .. math::
        \mathcal{L}_{\text{normalized}} = \frac{\mathcal{L}_{\mathbf{K}}}{N^2}

    Args:
        K (~torch.Tensor): The learnable cluster connectivity matrix of shape :math:`(C, C)`,
            where :math:`C` is the maximum number of clusters. This matrix models the expected
            connectivity patterns between different clusters.
        K_mu (~torch.Tensor): Prior mean matrix of shape :math:`(C, C)` specifying the
            expected values for the connectivity matrix. Usually designed to encourage
            higher within-cluster than between-cluster connectivity.
        K_var (~torch.Tensor): Prior variance parameter :math:`\sigma^2` (scalar tensor).
            Controls the strength of the regularization - smaller values impose stronger
            constraints towards the prior mean.
        normalize_loss (bool, optional): Whether to normalize the loss by the square of the
            number of nodes per graph :math:`N^2`. This ensures consistent scaling across
            graphs of different sizes when used in conjunction with other losses.
            (default: :obj:`False`)
        mask (Optional[~torch.Tensor]): A node mask for normalization of shape :math:`(B, N)`.
            Only used if :obj:`normalize_loss` is :obj:`True` to compute the effective number
            of nodes per graph in the batch.
            (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The cluster connectivity prior loss.

    Note:
        - Typically used with :math:`\mu_{\text{diag}} > 0` and :math:`\mu_{\text{off}} < 0`
        - The loss strength can be controlled through :obj:`K_var`
    """
    prior_loss = (0.5 * (K - K_mu) ** 2 / K_var).sum()

    # Normalize by N^2 if requested
    if normalize_loss:
        if mask is not None:
            N_squared = mask.sum(-1) ** 2  # (B,) - per-graph N^2
        else:
            N_squared = K.shape[-1] ** 2  # All nodes are valid

        prior_loss = prior_loss / N_squared  # scalar / vector = vector

        return _batch_reduce_loss(prior_loss, batch_reduction)

    return prior_loss


def maxcut_loss(
    scores: Tensor,
    edge_index: Tensor,
    edge_weight: Optional[Tensor] = None,
    batch: Optional[Tensor] = None,
    batch_reduction: BatchReductionType = "mean",
) -> Tensor:
    r"""Auxiliary MaxCut loss used by :class:`~tgp.poolers.MaxCutPooling`
    operator from the paper `"MaxCutPool: differentiable feature-aware Maxcut for
    pooling in graph neural networks" <https://arxiv.org/abs/2409.05100>`_
    (Abate & Bianchi, ICLR 2025).

    The MaxCut objective aims to maximize the sum of edge weights crossing a graph partition.
    For differentiable optimization, the loss minimizes the negative normalized MaxCut value:

    .. math::
        \mathcal{L}_{\text{MaxCut}} = -\frac{1}{V} \sum_{(i,j) \in E} w_{ij} \cdot z_i \cdot z_j

    where:

    + :math:`z_i \in [-1, 1]` are the node scores/assignments,
    + :math:`w_{ij}` are the edge weights,
    + :math:`V = \sum_{(i,j) \in E} w_{ij}` is the graph volume (total edge weight),
    + :math:`E` is the edge set.

    The computation is performed efficiently using sparse matrix operations:

    .. math::
        \mathcal{L}_{\text{MaxCut}} = -\frac{\mathbf{z}^{\top} \mathbf{A} \mathbf{z}}{V}

    where :math:`\mathbf{A}` is the weighted adjacency matrix and :math:`\mathbf{z}` contains node scores.

    **Implementation Details:**

    1. Node scores are normalized via :math:`\tanh` to :math:`[-1, 1]` range
    2. Sparse matrix multiplication :math:`\mathbf{A} \mathbf{z}` is computed efficiently
    3. Volume normalization ensures loss comparability across different graph sizes
    4. Batch processing handles multiple graphs simultaneously

    Args:
        scores (~torch.Tensor): Node scores/assignments of shape :math:`(N,)` or :math:`(N, 1)`.
            Typically normalized to :math:`[-1, 1]` via :obj:`tanh` activation.
        edge_index (~torch.Tensor): Graph connectivity in COO format of shape :math:`(2, E)`.
        edge_weight (~torch.Tensor, optional): Edge weights of shape :math:`(E,)`.
            If :obj:`None`, all edges have weight :obj:`1.0`. (default: :obj:`None`)
        batch (~torch.Tensor, optional): Batch assignments for each node of shape :math:`(N,)`.
            If :obj:`None`, assumes single graph. (default: :obj:`None`)
        batch_reduction (str, optional): Reduction method applied to the batch dimension.
            Can be :obj:`'mean'` or :obj:`'sum'`.
            (default: :obj:`"mean"`)

    Returns:
        ~torch.Tensor: The MaxCut loss value (scalar for single graph, or reduced across batch).

    Note:
        The volume normalization :math:`V = \sum_{(i,j) \in E} w_{ij}` ensures that the loss
        magnitude is comparable across graphs of different sizes and densities, making it
        suitable for batched training scenarios.
    """
    # Handle score shapes
    if scores.dim() == 2 and scores.size(1) == 1:
        scores = scores.squeeze(-1)
    elif scores.dim() != 1:
        raise ValueError(
            f"Expected scores to have shape [N] or [N, 1], got {scores.shape}"
        )

    num_nodes = scores.size(0)

    if batch is None:
        batch = torch.zeros(num_nodes, dtype=torch.long, device=scores.device)

    if edge_weight is None:
        edge_weight = torch.ones(edge_index.size(1), device=scores.device)
    else:
        # Ensure edge_weight is 1D - squeeze if it has shape (E, 1)
        if edge_weight.dim() > 1:
            edge_weight = edge_weight.squeeze()

    # Construct sparse adjacency matrix
    adj = SparseTensor(
        row=edge_index[0],
        col=edge_index[1],
        value=edge_weight,
        sparse_sizes=(num_nodes, num_nodes),
    )

    # Compute A * z (adjacency matrix times scores)
    az = adj.matmul(scores.unsqueeze(-1)).squeeze(-1)

    # Compute z^T * A * z for each graph in the batch
    cut_values = scores * az
    cut_losses = scatter(cut_values, batch, dim=0, reduce="sum")

    # Compute volume (total edge weight) for each graph
    edge_batch = batch[edge_index[0]]
    volumes = scatter(edge_weight, edge_batch, dim=0, reduce="sum")

    # Normalize by volume and take mean across graphs
    normalized_cut_losses = cut_losses / volumes

    return _batch_reduce_loss(normalized_cut_losses, batch_reduction)
