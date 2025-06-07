import math
from typing import Literal, Optional

import torch
from torch import Tensor

from tgp import eps
from tgp.utils import rank3_diag, rank3_trace

ReductionType = Literal["mean", "sum", "none"]


def _reduce_loss(loss: Tensor, reduction: ReductionType) -> Tensor:
    if reduction == "mean":
        return torch.mean(loss)
    if reduction == "sum":
        return torch.sum(loss)
    raise ValueError(
        f"Reduction {reduction} not allowed, must be one of ['mean', 'sum']."
    )


def mincut_loss(
    adj: Tensor, S: Tensor, adj_pooled: Tensor, reduction: ReductionType = "none"
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

    Returns:
        ~torch.Tensor: The mincut loss.
    """
    num = rank3_trace(adj_pooled)
    d_flat = adj.sum(-1)
    d = rank3_diag(d_flat)
    den = rank3_trace(torch.matmul(torch.matmul(S.transpose(-2, -1), d), S))
    cut_loss = -(num / den)
    return _reduce_loss(cut_loss, reduction)


def orthogonality_loss(S: Tensor, reduction: ReductionType = "none") -> Tensor:
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

    Returns:
        ~torch.Tensor: The orthogonality loss.
    """
    STS = torch.matmul(S.transpose(-2, -1), S)
    STS_term = STS / torch.norm(STS, dim=(-2, -1), keepdim=True)
    k = S.size(-1)
    id_k = torch.eye(k, device=S.device, dtype=S.dtype) / math.sqrt(k)
    ortho_loss = torch.norm(STS_term - id_k, dim=(-2, -1))
    return _reduce_loss(ortho_loss, reduction)


def hosc_orthogonality_loss(
    S: Tensor, mask: Optional[Tensor] = None, reduction: ReductionType = "none"
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

    Returns:
        ~torch.Tensor: The orthogonality loss.
    """
    _, num_nodes, num_clusters = S.size()
    norm = torch.norm(S, p="fro", dim=-2).sum(dim=-1)
    sqrt_k = math.sqrt(num_clusters)
    sqrt_nodes = mask.sum(1).sqrt() if mask is not None else math.sqrt(num_nodes)
    ortho_num = -norm / sqrt_nodes + sqrt_k
    ortho_loss = ortho_num / (sqrt_k - 1)
    return _reduce_loss(ortho_loss, reduction)


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


def entropy_loss(S: Tensor, reduction: ReductionType = "none") -> Tensor:
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

    Returns:
        ~torch.Tensor: The entropy regularization loss.
    """
    entropy = -S * torch.log(S + eps)
    entropy = torch.sum(entropy, dim=-1)
    return _reduce_loss(entropy, reduction)


def totvar_loss(S: Tensor, adj: Tensor, reduction: ReductionType = "none") -> Tensor:
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

    Returns:
        ~torch.Tensor: The total variation regularization loss.
    """
    l1_norm = torch.sum(torch.abs(S[..., None, :] - S[:, None, ...]), dim=-1)
    loss = torch.sum(adj * l1_norm, dim=(-1, -2))
    n_edges = torch.count_nonzero(adj, dim=(-1, -2))
    loss *= 1 / (2 * n_edges)
    return _reduce_loss(loss, reduction)


def asym_norm_loss(S: Tensor, k: int, reduction: ReductionType = "none") -> Tensor:
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

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
    return _reduce_loss(loss, reduction)


def just_balance_loss(
    S: Tensor,
    mask: Optional[Tensor] = None,
    normalize_loss: bool = True,
    num_nodes: int = None,
    num_clusters: int = None,
    reduction: ReductionType = "none",
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

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

    return _reduce_loss(loss, reduction)


def spectral_loss(
    adj: Tensor,
    S: Tensor,
    adj_pooled: Tensor,
    mask: Optional[Tensor] = None,
    num_clusters: int = None,
    reduction: ReductionType = "none",
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

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
    return _reduce_loss(spectral_loss, reduction)


def cluster_loss(
    S: Tensor,
    mask: Optional[Tensor] = None,
    num_clusters: int = None,
    reduction: ReductionType = "none",
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
        reduction (str, optional): The reduction method to apply to the loss.
            (default: :obj:`"none"`)

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
    return _reduce_loss(cluster_loss, reduction)
