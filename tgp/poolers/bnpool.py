from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Beta

from tgp.connect import DenseConnect, postprocess_adj_pool
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DPSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.losses import (
    cluster_connectivity_prior_loss,
    kl_loss,
    weighted_bce_reconstruction_loss,
)
from tgp.utils.typing import LiftType, SinvType


class BNPool(DenseSRCPooling):
    r"""The BN-Pool operator from the paper `"BN-Pool: Bayesian Nonparametric Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana & Bianchi, 2025).

    BN-Pool implements a Bayesian nonparametric approach to graph pooling using a Dirichlet Process
    with stick-breaking construction for cluster assignment. The method learns both the number of clusters
    and their assignments through variational inference.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DPSelect` to perform variational inference of the stick-breaking process.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    The method uses a truncated stick-breaking representation of the Dirichlet Process:

    .. math::
        v_{ik} \sim \text{Beta}(\alpha_{ik}, \beta_{ik}), \quad i = 1, \ldots, N \quad k = 1, \ldots, K-1

    .. math::
        \pi_{ik} = v_{ik} \prod_{j=1}^{k-1} (1 - v_{ij})

    where :math:`\pi_{ik}` represents the probability of assigning node :math:`i` to cluster :math:`k`.
    The coefficients :math:`\alpha_{ik}` and :math:`\beta_{ik}` are computed by an MLP
    from node features :math:`\mathbf{x}_i`.

    The cluster connectivity is modeled through a learnable matrix :math:`\mathbf{K} \in \mathbb{R}^{K \times K}`
    and the pooled adjacency matrix is computed as:

    .. math::
        \mathbf{A}_{\text{rec}} = \mathbf{S} \mathbf{K} \mathbf{S}^{\top}

    where :math:`S_{ik} = \pi_{ik}`.

    This layer optimizes three auxiliary losses:

    + **Reconstruction loss** (:func:`~tgp.utils.losses.weighted_bce_reconstruction_loss`): Binary cross-entropy loss between the true and reconstructed adjacency matrix :math:`\mathbf{A}_{\text{rec}}`.
    + **KL divergence loss** (:func:`~tgp.utils.losses.kl_loss`): KL divergence between the prior and posterior variational approximation of the stick-breaking variables.
    + **Cluster connectivity prior loss** (:func:`~tgp.utils.losses.cluster_connectivity_prior_loss`): Prior regularization on the cluster connectivity matrix :math:`\mathbf{K}`.

    Args:
        in_channels (Union[int, List[int]]): The number of input node feature channels.
            If a list is provided, it specifies the architecture of the MLP in :class:`~tgp.select.DPSelect`.
        k (int): The maximum number of clusters :math:`K` to be used in the pooling mechanism.
            The actual number of active clusters is learned through the stick-breaking process.
        alpha_DP (float, optional): Prior concentration parameter :math:`\alpha` of the Dirichlet Process.
            Controls the expected number of clusters. Higher values encourage more clusters.
            (default: :obj:`1.0`)
        K_var (float, optional): Variance :math:`\sigma^2` of the Gaussian prior on the cluster connectivity matrix :math:`\mathbf{K}`.
            (default: :obj:`1.0`)
        K_mu (float, optional): Mean parameter for the cluster connectivity prior. The prior mean matrix is constructed as
            :math:`\mathbf{K}_{\mu} = \mu \mathbf{I} - \mu (\mathbf{1}\mathbf{1}^{\top} - \mathbf{I})`.
            (default: :obj:`10.0`)
        K_init (float, optional): Initial value for the cluster connectivity matrix :math:`\mathbf{K}`.
            (default: :obj:`1.0`)
        eta (float, optional): Weights the KL divergence loss term.
            (default: :obj:`1.0`)
        rescale_loss (bool, optional): If :obj:`True`, losses are normalized by the square of the number of nodes :math:`N^2`
            to ensure proper scaling across different graph sizes.
            (default: :obj:`True`)
        balance_links (bool, optional): If :obj:`True`, applies class-balancing weights in the reconstruction loss
            to handle the imbalance between edges and non-edges in sparse graphs.
            (default: :obj:`True`)
        train_K (bool, optional): If :obj:`True`, the cluster connectivity matrix :math:`\mathbf{K}` is learnable.
            If :obj:`False`, :math:`\mathbf{K}` is fixed to its initial value.
            (default: :obj:`True`)
        act (str, optional): Activation function for the MLP in :class:`~tgp.select.DPSelect`.
            (default: :obj:`None`)
        dropout (float, optional): Dropout rate in the MLP of :class:`~tgp.select.DPSelect`.
            (default: :obj:`0.0`)
        adj_transpose (bool, optional):
            If :obj:`True`, the preprocessing step in :class:`tgp.src.DenseSRCPooling` and
            the :class:`tgp.connect.DenseConnect` operation returns transposed
            adjacency matrices, so that they could be passed "as is" to the dense
            message-passing layers.
            (default: :obj:`True`)
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
    """

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,  # hyperparameters of the method
        alpha_DP=1.0,
        K_var=1.0,
        K_mu=10.0,
        K_init=1.0,
        eta=1.0,
        rescale_loss=True,
        balance_links=True,
        train_K=True,  # hyperparameters of the selector
        act: str = None,
        dropout: float = 0.0,
        adj_transpose: bool = True,
        lift: LiftType = "precomputed",
        s_inv_op: SinvType = "transpose",
    ):
        if alpha_DP <= 0:
            raise ValueError("alpha_DP must be positive")

        if K_var <= 0:
            raise ValueError("K_var must be positive")

        if eta <= 0:
            raise ValueError("eta must be positive")

        if k <= 0:
            raise ValueError("max_k must be positive")

        super(BNPool, self).__init__(
            selector=DPSelect(in_channels, k, act, dropout, s_inv_op),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnect(
                remove_self_loops=False, degree_norm=False, adj_transpose=adj_transpose
            ),
            adj_transpose=adj_transpose,
        )

        self.k = k
        self.K_init_val = K_init
        self.alpha_DP = alpha_DP
        self.K_var_val = K_var
        self.K_mu_val = K_mu
        self.rescale_loss = rescale_loss
        self.balance_links = balance_links
        self.train_K = train_K
        self.eta = eta  # coefficient for the kl_loss

        # prior of the Stick Breaking Process
        self.register_buffer("alpha_prior", torch.ones(self.k - 1))
        self.register_buffer("beta_prior", torch.ones(self.k - 1) * alpha_DP)

        # prior of the cluster-cluster prob. matrix
        self.register_buffer("K_var", torch.tensor(K_var))
        self.register_buffer(
            "K_mu",
            K_mu * torch.eye(self.k, self.k) - K_mu * (1 - torch.eye(self.k, self.k)),
        )

        # cluster-cluster prob matrix
        self.K = torch.nn.Parameter(
            K_init * torch.eye(self.k, self.k)
            - K_init * (1 - torch.eye(self.k, self.k)),
            requires_grad=train_K,
        )

    def reset_parameters(self):
        super().reset_parameters()
        self.K.data = self.K_init_val * torch.eye(
            self.k, self.k, device=self.K.device
        ) - self.K_init_val * (1 - torch.eye(self.k, self.k, device=self.K.device))

    def forward(
        self,
        x: Tensor,
        adj: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        mask: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
            adj (~torch.Tensor): Adjacency tensor
                :math:`\mathbf{A} \in \mathbb{R}^{B \times N \times N}`.
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes in each graph. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted

        else:
            # Select
            so = self.select(x=x, mask=mask)

            # Reduce
            x_pooled, _ = self.reduce(x=x, so=so)

            # Connect
            adj_pool, _ = self.connect(edge_index=adj, so=so)

            loss = self.compute_loss(adj, mask, so)

            # Normalize coarsened adjacency matrix
            adj_pool = postprocess_adj_pool(
                adj_pool,
                remove_self_loops=True,
                degree_norm=True,
                adj_transpose=self.adj_transpose,
            )

            out = PoolingOutput(x=x_pooled, edge_index=adj_pool, so=so, loss=loss)

            return out

    def compute_loss(self, adj, mask, so) -> dict:
        r"""Computes the loss components for BN-Pool training.

        This method calculates three loss components that guide the learning of cluster assignments
        and connectivity patterns:

        1. **Reconstruction Loss**: Measures how well the learned cluster connectivity matrix :math:`\mathbf{K}`
           can reconstruct the original adjacency matrix through :math:`\mathbf{A}_{\text{rec}} = \mathbf{S} \mathbf{K} \mathbf{S}^{\top}`.

        2. **KL Divergence Loss**: Regularizes the posterior stick-breaking variables :math:`q(v_k)` towards
           the Dirichlet Process prior :math:`p(v_k)`.

        3. **Cluster Connectivity Prior Loss**: Regularizes the learned connectivity matrix :math:`\mathbf{K}`
           towards the specified prior distribution.

        All losses can be optionally normalized by :math:`N^2` (number of node pairs) when :attr:`rescale_loss=True`
        to ensure consistent scaling across different graph sizes.

        Args:
            adj (~torch.Tensor): True adjacency matrix of shape :math:`(B, N, N)` to reconstruct.
            mask (~torch.Tensor): Boolean node mask of shape :math:`(B, N)` indicating valid nodes.
                Used to handle variable-sized graphs within batches.
            so (:class:`~tgp.select.SelectOutput`): Selection output containing:
                - :attr:`s`: Soft assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
                - :attr:`q_z`: Posterior Beta distributions for stick-breaking variables

        Returns:
            dict: Dictionary containing three loss components:
                - :obj:`'quality'`: Reconstruction loss :math:`\mathcal{L}_{\text{rec}}`
                  (see :func:`~tgp.utils.losses.weighted_bce_reconstruction_loss`)
                - :obj:`'kl'`: KL divergence loss :math:`\eta \cdot \mathcal{L}_{\text{KL}}` weighted by :attr:`eta`
                  (see :func:`~tgp.utils.losses.kl_loss`)
                - :obj:`'K_prior'`: Cluster connectivity prior loss :math:`\mathcal{L}_{\mathbf{K}}`
                  (see :func:`~tgp.utils.losses.cluster_connectivity_prior_loss`).
                  Set to :obj:`0.0` if :attr:`train_K=False`.

        Note:
            The total training loss is typically computed as:
            :math:`\mathcal{L}_{\text{total}} = \mathcal{L}_{\text{rec}} + \mathcal{L}_{\text{KL}} + \mathcal{L}_{\mathbf{K}}`
        """
        s, q_z = so.s, so.q_z
        rec_adj = self.get_rec_adj(s)

        # Reconstruction loss
        rec_loss = weighted_bce_reconstruction_loss(
            rec_adj,
            adj,
            mask,
            balance_links=self.balance_links,
            normalize_loss=self.rescale_loss,
            batch_reduction="mean",
        )

        # KL loss
        alpha_prior = self.get_buffer("alpha_prior")
        beta_prior = self.get_buffer("beta_prior")
        prior_dist = Beta(alpha_prior, beta_prior)
        kl_loss_value = kl_loss(
            q_z,
            prior_dist,
            mask=mask,
            node_axis=1,  # Nodes are on axis 1: (B, N, K-1)
            sum_axes=[2, 1],  # Sum over K-1 components (axis 2), then nodes (axis 1)
            normalize_loss=self.rescale_loss,
            batch_reduction="mean",
        )

        # K prior loss
        if self.train_K:
            K_prior_loss = cluster_connectivity_prior_loss(
                self.K,
                self.get_buffer("K_mu"),
                self.get_buffer("K_var"),
                normalize_loss=self.rescale_loss,
                mask=mask,
                batch_reduction="mean",
            )
        else:
            K_prior_loss = torch.tensor(0.0)

        # build the output dictionary
        return {
            "quality": rec_loss,
            "kl": self.eta * kl_loss_value,
            "K_prior": K_prior_loss,
        }

    def extra_repr_args(self) -> dict:
        return {
            "alpha_DP": self.alpha_DP,
            "k_prior_variance": self.K_var_val,
            "k_prior_mean": self.K_mu_val,
            "k_init_value": self.K_init_val,
            "eta": self.eta,
            "rescale_loss": self.rescale_loss,
            "balance_links": self.balance_links,
            "train_K": self.train_K,
        }

    def get_rec_adj(self, S):
        return S @ self.K @ S.transpose(-1, -2)
