from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from tgp.select import DenseSelect, SelectOutput
from tgp.utils.typing import SinvType


class DPSelect(DenseSelect):
    r"""The Dirichlet Process selection operator for the BN-Pool operator (:class:`~tgp.poolers.BNPool`)
    as proposed in the paper `"BN-Pool: a Bayesian Nonparametric Approach for Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana D., and Bianchi F.M., preprint, 2025).

    **Overview:**

    DPSelect implements a Bayesian nonparametric selection mechanism using a truncated stick-breaking
    representation of the Dirichlet Process. This allows the model to automatically learn both
    the number of clusters and their assignments through variational inference.

    **Mathematical Formulation:**

    The method uses a truncated stick-breaking process to model cluster assignments:

    .. math::
        \mathbf{v}_{ik} \sim \text{Beta}(\boldsymbol{\alpha}_{ik}, \boldsymbol{\beta}_{ik}), \quad k = 1, \ldots, K-1, \quad i = 1, \ldots, N

    where :math:`\mathbf{v}_{ik}` are the stick-breaking fractions. The assignment of node :math:`i` to cluster :math:`k` is computed as:

    .. math::
        \boldsymbol{\pi}_{ik} = \mathbf{v}_{ik} \prod_{j=1}^{k-1} (1 - \mathbf{v}_{ij}) \quad \text{for } k = 1, \ldots, K-1

    The variational parameters :math:`\boldsymbol{\alpha}_{ik}, \boldsymbol{\beta}_{ik}` are computed by an MLP from node features:

    .. math::
        [\boldsymbol{\alpha}_{i,1}, \ldots, \boldsymbol{\alpha}_{i,K-1}, \boldsymbol{\beta}_{i,1}, \ldots, \boldsymbol{\beta}_{i,K-1}] = \text{softplus}(\text{MLP}(\mathbf{x}_i))

    **Architecture:**

    1. **Feature Processing**: Node features :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}` are processed
       by an MLP to produce :math:`2(K-1)` outputs per node.

    2. **Parameter Extraction**: The MLP output is split into :math:`\alpha` and :math:`\beta` parameters
       for :math:`K-1` Beta distributions.

    3. **Sampling**: Stick-breaking fractions are obtained from the sampling procedure implemented in :class:`~torch.distributions.beta.Beta`:
       :math:`\mathbf{v}_{ik} = \text{Beta}(\boldsymbol{\alpha}_{ik}, \boldsymbol{\beta}_{ik}).rsample()`.

    4. **Cluster Assignment**: The assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
       is computed via the stick-breaking construction: :math:`\mathbf{S}_{[i,:]} = \mathbf{v}_{[i,:]}`.

    Args:
        in_channels (Union[int, List[int]]): Input feature dimensions. If an integer, specifies the input size.
            If a list, defines the full MLP architecture including hidden layers.
        k (int): Maximum number of clusters :math:`K`. The actual number of active clusters is learned
            through the stick-breaking process.
        act (str, optional): Activation function for the MLP hidden layers.
            See :class:`~torch_geometric.nn.models.mlp.MLP` for available options.
            (default: :obj:`None`)
        dropout (float, optional): Dropout probability in the MLP for regularization during training.
            Must be between 0.0 and 1.0.
            (default: :obj:`0.0`)
        s_inv_op (:class:`~tgp.utils.typing.SinvType`, optional): Method for computing the pseudo-inverse
            :math:`\mathbf{S}^{-1}` of the assignment matrix for lifting operations.

            - :obj:`"transpose"`: Use :math:`\mathbf{S}^{-1} = \mathbf{S}^{\top}`
            - :obj:`"inverse"`: Use Moore-Penrose pseudoinverse :math:`\mathbf{S}^{-1} = \mathbf{S}^{+}`

            (default: :obj:`"transpose"`)

    Note:
        This class extends :class:`~tgp.select.DenseSelect` but replaces the softmax assignment
        with the stick-breaking construction. The output includes both the assignment matrix
        :math:`\mathbf{S}` and the posterior distributions :math:`q(\mathbf{v}_k)` for computing KL divergence losses.

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
        r"""Applies the Dirichlet Process selection operator to compute cluster assignments.

        This method performs the following steps:

        1. Processes node features through an MLP to obtain Beta distribution parameters
        2. Samples stick-breaking fractions using the reparameterization trick
        3. Computes cluster assignment probabilities via stick-breaking construction
        4. Applies optional masking for variable-sized graphs

        Args:
            x (~torch.Tensor): Input node features of shape :math:`(B, N, F)` where:
                - :math:`B` is batch size
                - :math:`N` is number of nodes per graph
                - :math:`F` is feature dimension
                If input is 2D with shape :math:`(N, F)`, it will be unsqueezed to :math:`(1, N, F)`.
            mask (Optional[~torch.Tensor]): Boolean mask of shape :math:`(B, N)` indicating valid nodes.
                Applied element-wise to zero out assignments for invalid nodes.
                (default: :obj:`None`)
            **kwargs: Additional keyword arguments (unused, for compatibility with base class).

        Returns:
            :class:`~tgp.select.SelectOutput`: Selection output containing:
                - :attr:`s`: Assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}` with
                  cluster probabilities for each node
                - :attr:`s_inv_op`: Method used for computing :math:`\mathbf{S}^{-1}`
                - :attr:`mask`: Copy of the input mask (if provided)
                - :attr:`q_z`: Posterior Beta distributions :math:`q(\mathbf{v}_k)` for each stick-breaking fraction,
                  used for computing KL divergence losses

        Note:
            The sampling method :meth:`~torch.distributions.beta.Beta.rsample` implements a pathwise gradient estimator
            that allows to back-propagate the gradient from the samples to the distribution parameters,
            making the method end-to-end differentiable.
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
