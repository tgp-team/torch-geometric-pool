from typing import List, Optional, Union

import torch
import torch.nn.functional as F
from torch import Tensor
from torch.distributions import Beta

from tgp.select import DenseSelect, SelectOutput
from tgp.utils.typing import SinvType


class DPSelect(DenseSelect):
    r"""The select operator for the BN-Pool operator (:class:`~tgp.poolers.BNPool`)
    as proposed in the paper `"BN-Pool: a Bayesian Nonparametric Approach for Graph Pooling" <https://arxiv.org/abs/2501.09821>`_
    (Castellana D., and Bianchi F.M., preprint, 2025).

    It provides a mechanism for performing dense Dirichlet Process-based selection.
    This class extends the DenseSelect class and is designed to compute the assignment matrix S by leveraging
    a truncated stick-breaking variational approximation of the DP posterior. To this end, an MLP is used to compute
    the alpha and beta parameters of the stick fractions' variational distribution; then, the final assignment matrix is
    obtained by simulating the stick breaking process.

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
