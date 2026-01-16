from typing import List, Optional, Union

import torch
from torch import Tensor
from torch.distributions import Beta
from torch_geometric.typing import Adj

from tgp.connect import DenseConnectUnbatched
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import DPSelect, SelectOutput
from tgp.src import PoolingOutput, SRCPooling
from tgp.utils import (
    batched_negative_edge_sampling,
    connectivity_to_edge_index,
    negative_edge_sampling,
)
from tgp.utils.losses import (
    cluster_connectivity_prior_loss,
    kl_loss,
    sparse_bce_reconstruction_loss,
)
from tgp.utils.typing import LiftType, SinvType


class SparseBNPool(SRCPooling):
    r"""A sparse implementation of the BN-Pool operator from the paper `"BN-Pool: Bayesian Nonparametric Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana & Bianchi, 2025). See :class:`~tgp.poolers.BNPool` for more details.

    This sparse implementation implements the BN-Pool operator with a sparse adjacency matrix,
    computing the reconstruction loss on a subset of all possible edges to reduce
    computational complexity, making it more efficient for large sparse graphs.

    + The :math:`\texttt{select}` operator is implemented with :class:`~tgp.select.DPSelect` (with :obj:`batched_representation=False`) to perform variational inference of the stick-breaking process on sparse graphs.
    + The :math:`\texttt{reduce}` operator is implemented with :class:`~tgp.reduce.BaseReduce`.
    + The :math:`\texttt{connect}` operator is implemented with :class:`~tgp.connect.DenseConnectUnbatched`.
    + The :math:`\texttt{lift}` operator is implemented with :class:`~tgp.lift.BaseLift`.

    For a dense implementation that works with dense adjacency tensors, see :class:`~tgp.poolers.BNPool`.

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
        rescale_loss (bool, optional): If :obj:`True`, losses are normalized by the number of sampled edges
            :math:`E + E_{\text{neg}}` to ensure proper scaling across different graph sizes.
            (default: :obj:`True`)
        train_K (bool, optional): If :obj:`True`, the cluster connectivity matrix :math:`\mathbf{K}` is learnable.
            If :obj:`False`, :math:`\mathbf{K}` is fixed to its initial value.
            (default: :obj:`True`)
        act (str, optional): Activation function for the MLP in :class:`~tgp.select.DPSelect`.
            (default: :obj:`None`)
        dropout (float, optional): Dropout rate in the MLP of :class:`~tgp.select.DPSelect`.
            (default: :obj:`0.0`)
        remove_self_loops (bool, optional):
            If :obj:`True`, the self-loops will be removed from the pooled adjacency matrix.
            (default: :obj:`True`)
        degree_norm (bool, optional):
            If :obj:`True`, the pooled adjacency matrix will be symmetrically normalized.
            (default: :obj:`True`)
        edge_weight_norm (bool, optional):
            Whether to normalize the pooled edge weights by dividing by the maximum absolute value per graph.
            (default: :obj:`False`)
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
        train_K=True,  # hyperparameters of the selector
        act: str = None,
        dropout: float = 0.0,
        remove_self_loops: bool = True,
        degree_norm: bool = True,
        edge_weight_norm: bool = False,
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

        super(SparseBNPool, self).__init__(
            selector=DPSelect(
                in_channels,
                k,
                batched_representation=False,
                act=act,
                dropout=dropout,
                s_inv_op=s_inv_op,
            ),
            reducer=BaseReduce(),
            lifter=BaseLift(matrix_op=lift),
            connector=DenseConnectUnbatched(
                remove_self_loops=remove_self_loops,
                degree_norm=degree_norm,
                edge_weight_norm=edge_weight_norm,
            ),
        )
        self.adj_transpose = adj_transpose
        self.k = k
        self.K_init_val = K_init
        self.alpha_DP = alpha_DP
        self.K_var_val = K_var
        self.K_mu_val = K_mu
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
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        batch: Optional[Tensor] = None,
        batch_pooled: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                It can either be a :class:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape  :math:`[E]` or :math:`[E, 1]`
                containing the weights of the edges.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            batch_pooled (~torch.Tensor, optional): The batch vector for the pooled nodes.
                Required when lifting with dense :math:`[N, K]` SelectOutput on multi-graph
                batches. Pass `out.batch` from the pooling call. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            ~tgp.src.PoolingOutput: The output of the pooling operator.
        """
        if lifting:
            # Lift
            batch_orig = batch if batch is not None else so.batch
            x_lifted = self.lift(
                x_pool=x, so=so, batch=batch_orig, batch_pooled=batch_pooled
            )
            return x_lifted

        else:
            # Select
            so = self.select(x=x, batch=batch)

            loss = self.compute_loss(adj, batch, so)

            # Reduce
            x_pooled, batch_pooled = self.reduce(x=x, so=so, batch=batch)

            # Connect
            edge_index_pooled, edge_weight_pooled = self.connect(
                edge_index=adj,
                so=so,
                edge_weight=edge_weight,
                batch=batch,
                batch_pooled=batch_pooled,
            )

            out = PoolingOutput(
                x=x_pooled,
                edge_index=edge_index_pooled,
                edge_weight=edge_weight_pooled,
                batch=batch_pooled,
                so=so,
                loss=loss,
            )

            return out

    def compute_loss(self, adj, batch, so) -> dict:
        r"""Computes the loss components for BN-Pool training. See :class:`~tgp.poolers.BNPool` for more details.

        In this implementaion, the reconstruction loss is computed on a subset of all the possible edges to reduce the complexity.

        Args:
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                It can either be a :class:`~torch_sparse.SparseTensor` of (sparse) shape :math:`[N, N]`,
                where :math:`N` is the number of nodes in the batch or a :obj:`~torch.Tensor` of shape
                :math:`[2, E]`, where :math:`E` is the number of edges in the batch.
                If :obj:`lifting` is :obj:`False`, it cannot be :obj:`None`.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
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
        node_assignment, q_z = so.node_assignment, so.q_z

        if batch is not None:
            bs = int(batch.max()) + 1
        else:
            bs = 1

        # Reconstruction loss
        rec_loss, norm_const = self.get_sparse_rec_loss(node_assignment, adj, batch, bs)

        # KL loss
        alpha_prior = self.get_buffer("alpha_prior")
        beta_prior = self.get_buffer("beta_prior")
        prior_dist = Beta(alpha_prior, beta_prior)

        kl_loss_value = kl_loss(
            q_z,
            prior_dist,
            batch=batch,
            batch_size=bs,
            normalizing_const=norm_const,
            batch_reduction="mean",
        )

        # K prior loss
        if self.train_K:
            K_prior_loss = cluster_connectivity_prior_loss(
                self.K,
                self.get_buffer("K_mu"),
                self.get_buffer("K_var"),
                normalizing_const=norm_const,
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
            "train_K": self.train_K,
            "remove_self_loops": self.connector.remove_self_loops,
            "degree_norm": self.connector.degree_norm,
            "edge_weight_norm": self.connector.edge_weight_norm,
        }

    def get_sparse_rec_loss(self, node_assignment, adj, batch, bs):
        r"""Computes the sparse weighted binary cross-entropy reconstruction loss for BN-Pool."""
        edge_index, _ = connectivity_to_edge_index(adj)

        dev = edge_index.device
        if batch is None:
            neg_edge_index = negative_edge_sampling(edge_index, force_undirected=True)
        else:
            neg_edge_index = batched_negative_edge_sampling(
                edge_index, batch, force_undirected=True
            )

        E = edge_index.size(1)  # number of edges
        negE = neg_edge_index.size(1)  # numeber of negative samples
        all_edges = torch.cat(
            [edge_index, neg_edge_index], dim=1
        )  # has size 2 x (E+NegE]
        edges_batch_id = None

        if batch is not None:
            edges_batch_id = batch[all_edges[0]]

        link_prob_loigit = self.get_prob_link_logit(node_assignment, all_edges)

        pred_y = torch.cat(
            [torch.ones(E, device=dev), torch.zeros(negE, device=dev)], dim=0
        )

        return sparse_bce_reconstruction_loss(
            link_prob_loigit,
            pred_y,
            edges_batch_id=edges_batch_id,
            batch_size=bs,
        )

    def get_prob_link_logit(self, node_assignment, edges_list):
        left = node_assignment[edges_list[0]]  # E x K
        right = node_assignment[edges_list[1]]  # E x K
        aux = left @ self.K  # E x K
        return (aux * right).sum(-1)
        # return torch.einsum("ei, ej, ij -> e", left, right, self.K)
