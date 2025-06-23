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
    r"""The BN-Pool operator from the paper `"BN-Pool: a Bayesian Nonparametric Approach for Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana D., and Bianchi F.M., preprint, 2025).

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DPSelect` to perform variational inference of the stick-breaking process.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnect`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    This layer provides three auxiliary losses:

    + the reconstruction loss, i.e., the binary cross-entropy loss between the true and the reconstructed adjacency matrix,
    + the KL loss, i.e., the KL divergence between the prior and the posterior variational approximation of the assignments,
    + the K prior loss, i.e., the KL divergence between the prior and the cluster connectivity matrix.

    Args:
        in_channels (Union[int, List[int]]): The number of input channels or a list of input channels.
        k (int): The maximum number of clusters to be used in the pooling mechanism.
        alpha_DP (float, optional): Prior concentration parameter of the DP. Default is 1.0.
        K_var (float, optional): Variance of the cluster connectivity prior. Default is 1.0.
        K_mu (float, optional): Mean of the cluster connectivity prior. Default is 10.0.
        K_init (float, optional): Initial value for the cluster connectivity prior. Default is 1.0.
        eta (float, optional): Coefficient for the KL loss term. Default is 1.0.
        rescale_loss (bool, optional): Flag indicating whether to rescale the loss during training. Default is True.
        balance_links (bool, optional): Flag to enable balancing of incoming/outgoing links. Default is True.
        train_K (bool, optional): Specifies whether the cluster connectivity matrix is learnable. Default is True.
        act (str, optional): Activation function for the selector. Default is None.
        dropout (float, optional): Dropout rate to be used in the selector. Default is 0.0.
        adj_transpose (bool, optional): Flag whether to transpose adjacency matrices. Default is True.
        lift (LiftType, optional): Specifies the operation used in the lifting step. Default is "precomputed".
        s_inv_op (SinvType, optional): Specifies the sparse inverse operation in the selector.
            Default is "transpose".
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
        """Forward pass.

        Args:
            x (Tensor): The input feature matrix.
            adj (Optional[Tensor]): The adjacency matrix representing the graph's
                structure. Default is None.
            so (Optional[SelectOutput]): A selection output object used for lifting
                or pooling operations. Default is None.
            mask (Optional[Tensor]): A mask to apply during feature selection. Default
                is None.
            lifting (bool): Whether to perform a lifting operation instead of pooling.
                Default is False.
            **kwargs: Additional keyword arguments for specific implementations.

        Returns:
            PoolingOutput: An object containing the pooled features, coarsened
            adjacency matrix, selected outputs, and the loss value.
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
        """Computes the loss components for the model.

        This method calculates the reconstruction loss, the KL loss, and optionally
        the prior loss over the cluster connectivity matrix `K`. Reconstruction and KL losses are rescaled by the number of total nodes
        to ensure proper normalization when summing across graph structures. The final
        loss components are returned as a dictionary.

        Args:
            adj: Adjacency matrix to reconstruct with shape `B x N x N`, where `B`
                is the batch size and `N` is the number of nodes per graph.
            mask: Node mask of shape `B x N` for differentiating valid nodes, used
                to handle graphs with variable sizes within a batch.
            so: An object containing `s` (soft assignments of nodes to components)
                and `q_z` (latent distribution parameters for each node).

        Returns:
            dict: A dictionary containing the following loss terms:
                - 'quality': Mean reconstruction loss after summing over the nodes
                     and rescaling.
                - 'kl': Mean KL loss with rescaling, after adjusting by the scaling
                     factor `eta`.
                - 'K_prior': Mean prior loss for the number of clusters `K`. If
                     `train_K` is False, this will be zero.
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
            reduction="mean",
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
            reduction="mean",
        )

        # K prior loss
        if self.train_K:
            K_prior_loss = cluster_connectivity_prior_loss(
                self.K,
                self.get_buffer("K_mu"),
                self.get_buffer("K_var"),
                normalize_loss=self.rescale_loss,
                mask=mask,
                reduction="mean",
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
