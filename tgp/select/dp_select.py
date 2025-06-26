from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from tgp.select import DenseSelect, SelectOutput
from tgp.utils.typing import SinvType


class DPSelect(DenseSelect):
    r"""The Dirichlet Process selection operator for the :class:`~tgp.poolers.BNPool` operator,
    as proposed in the paper `"BN-Pool: Bayesian Nonparametric Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana & Bianchi, 2025).

    DPSelect implements a Bayesian nonparametric selection mechanism to automatically learn both
    the number of clusters and their assignments through variational inference. The method uses
    a truncated stick-breaking representation of the Dirichlet Process to model cluster assignments:

    .. math::
        v_{ik} \sim \text{Beta}(\alpha_{ik}, \beta_{ik}), \quad k = 1, \ldots, K-1, \quad i = 1, \ldots, N

    where :math:`v_{ik}` are the stick-breaking fractions.
    The assignment of node :math:`i` to cluster :math:`k` is computed as:

    .. math::
        \pi_{ik} = v_{ik} \prod_{j=1}^{k-1} (1 - v_{ij}) \quad \text{for } k = 1, \ldots, K-1

    The variational parameters :math:`\alpha_{ik}, \beta_{ik}` are computed by an MLP from node features:

    .. math::
        [\alpha_{i,1}, \ldots, \alpha_{i,K-1}, \beta_{i,1}, \ldots, \beta_{i,K-1}] = \text{softplus}(\text{MLP}(\mathbf{x}_i))

    The procedure can be summarized as follows:

    1. **Feature Processing**: Node features :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}` are processed
       by an MLP to produce :math:`2(K-1)` outputs per node.

    2. **Parameter Extraction**: The MLP output is split into :math:`\boldsymbol{\alpha}` and :math:`\boldsymbol{\beta}`
       parameters for the Beta distribution.

    3. **Sampling**: Stick-breaking fractions are obtained from the sampling procedure implemented in :class:`~torch.distributions.beta.Beta`:
       :math:`v_{ik} = \text{Beta}(\alpha_{ik}, \beta_{ik}).rsample()`.

    4. **Cluster Assignment**: The assignment matrix :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}`
       is computed via the stick-breaking construction: :math:`S_{bik} = \pi_{bik}`.

    Args:
        in_channels (int, list of int):
            Number of hidden units for each hidden layer in the
            :class:`~torch_geometric.nn.models.mlp.MLP` used to
            compute cluster assignments.
            The first integer must match the size of the node features.
        k (int):
            Maximum number of clusters :math:`K`. The actual number of active clusters is learned
            through the stick-breaking process.
        act (str or Callable, optional):
            Activation function in the hidden layers of the
            :class:`~torch_geometric.nn.models.mlp.MLP`.
        dropout (float, optional): Dropout probability in the
            :class:`~torch_geometric.nn.models.mlp.MLP`.
            (default: :obj:`0.0`)
        s_inv_op (~tgp.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.

    Note:
        This class extends :class:`~tgp.select.DenseSelect` but replaces the softmax assignment with the
        stick-breaking construction. The :class:`~tgp.select.SelectOutput` returned includes both the assignment matrix
        :math:`\mathbf{S}` and the posterior distributions :math:`q(v_{ik})` for computing KL divergence losses.
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

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the node assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}` is
                being created within this method.
            mask (~torch.Tensor, optional): Mask matrix
                :math:`\mathbf{M} \in {\{ 0, 1 \}}^{B \times N}` indicating
                the valid nodes for each graph. (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
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
