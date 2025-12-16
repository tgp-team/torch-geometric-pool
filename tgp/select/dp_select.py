from typing import List, Optional, Union

import torch
import torch.nn.functional as F
import torch_sparse
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
        out_size = stick_fractions.size()
        device = stick_fractions.device

        pi = torch.zeros(out_size[:-1] + (out_size[-1] + 1,), device=device)
        pi[..., :-1] = torch.log(stick_fractions)
        pi[..., 1:] += torch.cumsum(torch.log(1 - stick_fractions), dim=-1)
        return torch.exp(pi)

    def _inner_forward(self, x):
        out = torch.clamp(F.softplus(self.mlp(x)), min=1e-3, max=1e3)
        q_v_alpha, q_v_beta = torch.split(out, self.k - 1, dim=-1)
        q_z = Beta(q_v_alpha, q_v_beta)
        z = q_z.rsample()
        s = self._compute_pi_given_sticks(z)
        return s, q_z

    def forward(
        self, x: Tensor, mask: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        r"""Applies the Dirichlet Process selection operator to compute cluster assignments. This select operator works
        with the dense graph representation, i.e., the input tensor :obj:`"x"` is expected to
        be of shape :math:`\mathbb{R}^{B \times N \times F}`.

        Therefore, the shape of the assignment matrix :math:`\mathbf{S}` is :math:`\mathbb{R}^{B \times N \times K}`.

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
        s, q_z = self._inner_forward(x)

        if mask is not None:
            # we are in the dense case
            s = s * mask.unsqueeze(-1)

        return SelectOutput(s=s, s_inv_op=self.s_inv_op, mask=mask, q_z=q_z)


class DPSelectSparse(DPSelect):
    is_dense = False

    def forward(
        self, x: Tensor, batch: Optional[Tensor] = None, **kwargs
    ) -> SelectOutput:
        r"""Applies the Dirichlet Process selection operator to compute cluster assignments. This select operator works
        with a sparse graphs representation, i.e., the input tensor :obj:`"x"` is expected to
        be of shape :math:`\mathbb{R}^{N \times F}`.

        Therefore, the shape of the assignment matrix :math:`\mathbf{S}` is :math:`\mathbb{R}^{N \times (B \times K)}`.

        Args:
            x (~torch.Tensor): Node feature tensor
                :math:`\mathbf{X} \in \mathbb{R}^{B \times N \times F}`, with
                batch-size :math:`B`, (maximum) number of nodes :math:`N` for
                each graph, and feature dimension :math:`F`.
                Note that the node assignment matrix
                :math:`\mathbf{S} \in \mathbb{R}^{B \times N \times K}` is
                being created within this method.
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        s, q_z = self._inner_forward(x)

        if batch is None:
            batch = torch.zeros(
                x.size(0), dtype=torch.long, device=x.device
            )  # only one batch

        # we are in the sparse case, shas shape NxK
        dev = x.device
        row = torch.arange(x.size(0), device=dev).view(-1, 1).repeat(1, self.k).view(-1)
        col = (self.k * batch.view(-1, 1) + torch.arange(self.k, device=dev)).view(-1)
        # s_block = torch.sparse_coo_tensor(values=s.view(-1), indices=torch.stack([row, col], dim=0), is_coalesced=True)
        s_block = torch_sparse.SparseTensor(row=row, col=col, value=s.view(-1))
        return SelectOutput(
            s=s_block, s_inv_op=self.s_inv_op, q_z=q_z, node_assignment=s
        )
