from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta, kl_divergence

from tgp.connect import DenseConnect, postprocess_adj_pool
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DenseSelect, SelectOutput
from tgp.src import DenseSRCPooling, PoolingOutput
from tgp.utils.typing import LiftType, SinvType


class DPSelect(DenseSelect):
    """The DPSelect class provides a mechanism for performing dense Dirichlet Process-based selection.

    This class extends the DenseSelect class and is designed to compute the assignment matrix S by leveraging
    a truncated stick-breaking variational approximation of the DP posterior. To this end, an MLP is used to compute
    the alpha and beta parameters of the stick fractions' variational distribution; then, the final assignment matrix is obtained
    by simulating the stick breaking process.

    Args:
        in_channels: Input channels, defined as either an integer or a list
            of integers, which represent the number of channels.
        k: The maximum number of clusters.
        act: Name of the activation function to be used. Defaults to None if
            no activation is specified.
        dropout: Dropout probability applied during training for regularization.
            Takes values between 0.0 and 1.0. Defaults to 0.0.
        s_inv_op: An operation type to handle invariant operations for specific
            models. Accepts values of the SinvType enumeration. Defaults to
            "transpose".
    """

    is_dense = True

    def __init__(
        self,
        in_channels: Union[int, List[int]],
        k: int,
        act: str = None,
        dropout: float = 0.0,
        s_inv_op: SinvType = "transpose",
    ):
        # 2*max_key needs to compute both alphas and betas of the posterior
        super(DPSelect, self).__init__(
            in_channels=in_channels,
            k=2 * (k - 1),
            act=act,
            dropout=dropout,
            s_inv_op=s_inv_op,
        )
        self.k = k

    @staticmethod
    def _compute_pi_given_sticks(stick_fractions):
        """Computes the stick-breaking proportions (pi) for a given set of stick fractions.

        This function implements the stick-breaking by multiplying the stick fractions. The multiplications are done
        in the logarithmic space to avoid numerical errors.

        Args:
            stick_fractions (torch.Tensor): A tensor representing the stick fractions
                across dimensions [n_particles, batch, n_nodes, n_clusters-1]. Each
                value must be within the interval (0, 1).

        Returns:
            torch.Tensor: A tensor containing the cluster assignment probabilities [batch, n_nodes, n_clusters].
        """
        device = stick_fractions.device
        log_v = torch.concat(
            [
                torch.log(stick_fractions),
                torch.zeros(*stick_fractions.shape[:-1], 1, device=device),
            ],
            dim=-1,
        )
        log_one_minus_v = torch.concat(
            [
                torch.zeros(*stick_fractions.shape[:-1], 1, device=device),
                torch.log(1 - stick_fractions),
            ],
            dim=-1,
        )
        pi = torch.exp(
            log_v + torch.cumsum(log_one_minus_v, dim=-1)
        )  # has shape [n_particles, batch, n_nodes, n_clusters]
        return pi

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        """Applies the selection operator to the input data.

        This function takes input data to compute the parameters (q_v_alpha, q_v_beta) of the stick fractions'
        variational approximation. Then, a sample from the posterior is obtained by using the rsample method of the
        Beta distribution. This allows backpropagate through the sampling step. Finally, the assignment matrix S is
        obtained by simulating the stick breaking process.

        Args:
            x (Tensor): The input tensor to be processed. Typically expected to be
                either a 2D tensor (batch, features) or a 3D tensor with additional dimensions.
                If the input tensor has 2D, it is unsqueezed to 3D before further processing.
            mask (Optional[Tensor]): A tensor to mask portions of the generated output.
                The mask is expected to be applied along the last dimension of the output.
            **kwargs: Additional keyword arguments that can be passed to extend functionality
                or adapt the processing behavior as needed.

        Returns:
            SelectOutput: An object containing the processed output `s`, its inverse
                operation `s_inv_op`, the optional mask, and the sampled distribution `q_z`.
        """
        x = x.unsqueeze(0) if x.dim() == 2 else x

        out = torch.clamp(F.softplus(self.mlp(x)), min=1e-3, max=1e3)
        q_v_alpha, q_v_beta = torch.split(out, self.k - 1, dim=-1)
        q_z = Beta(q_v_alpha, q_v_beta)
        z = q_z.rsample()
        s = self._compute_pi_given_sticks(z)

        if mask is not None:
            s = s * mask.unsqueeze(-1)

        return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask, q_z=q_z)


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
        self._K_init = K_init
        self._alpha_DP = alpha_DP
        self._K_var = K_var
        self._K_mu = K_mu
        self._rescale_loss = rescale_loss
        self._balance_links = balance_links
        self._train_K = train_K
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

        self._pos_weight_cache = None

    def reset_parameters(self):
        super().reset_parameters()
        self.K.data = self._K_init * torch.eye(
            self.k, self.k, device=self.K.device
        ) - self._K_init * (1 - torch.eye(self.k, self.k, device=self.K.device))

    def _get_bce_weight(self, adj, mask=None):
        """Calculates the binary cross-entropy (BCE) weight for a given adjacency matrix to ensure balancing
        of positive and negative samples.

        This function generates weights for BCE loss calculations by computing the
        positive and negative edges contributions based on the given adjacency matrix
        and an optional mask. If enabled, it does caching to improve performance.

        Args:
            adj (torch.Tensor): The adjacency matrix representing the graph. It should
                be a tensor with `... x N x N` shape, where `N` is the number of nodes.
            mask (torch.Tensor | None): Optional tensor mask applied on the adjacency
                matrix for computation. It has the same dimensions as `adj`.

        Returns:
            torch.Tensor: Resulting tensor representing binary cross-entropy weights
                with the same dimensions as the adjacency matrix.
        """
        use_cache = self.preprocessing_cache is not None

        if use_cache and self._pos_weight_cache is not None:
            pos_weight = self._pos_weight_cache
        else:
            if mask is not None:
                N = mask.sum(-1).view(-1, 1, 1)  # has shape B x 1 x 1
            else:
                N = adj.shape[-1]
            n_edges = torch.clamp(adj.sum([-1, -2]), min=1).view(
                -1, 1, 1
            )  # this is a vector of size B x 1 x 1
            n_not_edges = torch.clamp(N**2 - n_edges, min=1).view(
                -1, 1, 1
            )  # this is a vector of size B x 1 x 1
            # the clamp is needed to avoid zero division when we have all edges
            pos_weight = (N**2 / n_edges) * adj + (N**2 / n_not_edges) * (1 - adj)

            if use_cache:
                self._pos_weight_cache = pos_weight
        return pos_weight

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
        rec_loss = self.dense_rec_loss(rec_adj, adj)  # has shape B x N x N
        kl_loss = self.eta * self.kl_loss(q_z)  # has shape B x N

        K_prior_loss = (
            self.K_prior_loss() if self._train_K else torch.tensor(0.0)
        )  # has shape 1
        # sum losses over nodes by considering the right number of nodes for each graph
        if mask is not None and not torch.all(mask):
            edge_mask = torch.einsum("bn,bm->bnm", mask, mask)  # has shape B x N x N
            rec_loss = rec_loss * edge_mask
            kl_loss = kl_loss * mask
        rec_loss = rec_loss.sum((-1, -2))  # has shape B
        kl_loss = kl_loss.sum(-1)  # has shape B

        # RESCALE THE LOSSES
        if self._rescale_loss:
            if mask is not None:
                N_2 = mask.sum(-1) ** 2
            else:
                N_2 = adj.shape[1] ** 2

            rec_loss = rec_loss / N_2
            kl_loss = kl_loss / N_2
            K_prior_loss = K_prior_loss / N_2

        # build the output dictionary
        return {
            "quality": rec_loss.mean(),
            "kl": self.eta * kl_loss.mean(),
            "K_prior": K_prior_loss.mean(),
        }

    def extra_repr_args(self) -> dict:
        return {
            "alpha_DP": self._alpha_DP,
            "k_prior_variance": self._K_var,
            "k_prior_mean": self._K_mu,
            "k_init_value": self._K_init,
            "eta": self.eta,
            "rescale_loss": self._rescale_loss,
            "balance_links": self._balance_links,
            "train_K": self._train_K,
        }

    def get_rec_adj(self, S):
        return S @ self.K @ S.transpose(-1, -2)

    def dense_rec_loss(self, rec_adj, adj):
        pos_weight = None
        if self._balance_links:
            pos_weight = self._get_bce_weight(adj)

        loss = F.binary_cross_entropy_with_logits(
            rec_adj, adj, weight=pos_weight, reduction="none"
        )

        return loss  # has shape B x N x N

    def kl_loss(self, q_z):
        p_z = Beta(self.get_buffer("alpha_prior"), self.get_buffer("beta_prior"))
        loss = kl_divergence(q_z, p_z).sum(-1)
        return loss  # has shape B x N

    def K_prior_loss(self):
        K_mu, K_var = self.get_buffer("K_mu"), self.get_buffer("K_var")
        K_prior_loss = (0.5 * (self.K - K_mu) ** 2 / K_var).sum()
        return K_prior_loss  # has shape 1
