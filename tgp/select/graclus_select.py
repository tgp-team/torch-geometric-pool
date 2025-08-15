from typing import Optional

import torch
from torch import Tensor
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.imports import check_torch_cluster_available, torch_cluster
from tgp.select import Select, SelectOutput
from tgp.utils import check_and_filter_edge_weights, connectivity_to_edge_index
from tgp.utils.typing import SinvType


class GraclusSelect(Select):
    r"""The :math:`\texttt{select}` operator inspired by the paper `"Weighted Graph Cuts without
    Eigenvectors: A Multilevel Approach" <https://ieeexplore.ieee.org/document/4302760>`_
    (Dhillon et al., TPAMI 2007).

    It implements a greedy clustering algorithm for picking an unmarked vertex and matching
    it with one of its unmarked neighbors (that maximizes its edge weight).

    Args:
        s_inv_op (~tgp.utils.typing.SinvType, optional):
            The operation used to compute :math:`\mathbf{S}_\text{inv}` from the select matrix
            :math:`\mathbf{S}`. :math:`\mathbf{S}_\text{inv}` is stored in the :obj:`"s_inv"` attribute of
            the :class:`~tgp.select.SelectOutput`. It can be one of:

            - :obj:`"transpose"` (default): Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^\top`,
              the transpose of :math:`\mathbf{S}`.
            - :obj:`"inverse"`: Computes :math:`\mathbf{S}_\text{inv}` as :math:`\mathbf{S}^+`,
              the Moore-Penrose pseudoinverse of :math:`\mathbf{S}`.
    """

    def __init__(self, s_inv_op: SinvType = "transpose"):
        super().__init__()

        self.s_inv_op = s_inv_op

    def forward(
        self,
        edge_index: Tensor,
        edge_weight: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        r"""Forward pass.

        Args:
            edge_index (~torch.Tensor, optional):
                The edge indices. Is a tensor of of shape  :math:`[2, E]`,
                where :math:`E` is the number of edges in the batch.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional):
                A vector of shape  :math:`[E]` or :math:`[E, 1]` containing the weights of the edges.
                (default: :obj:`None`)
            num_nodes (int, optional):
                The total number of nodes of the graphs in the batch.
                (default: :obj:`None`)

        Returns:
            :class:`~tgp.select.SelectOutput`: The output of :math:`\texttt{select}` operator.
        """
        check_torch_cluster_available()
        if isinstance(edge_index, SparseTensor):
            edge_index, edge_weight = connectivity_to_edge_index(
                edge_index, edge_weight
            )
        edge_weight = check_and_filter_edge_weights(edge_weight)
        num_nodes = maybe_num_nodes(edge_index, num_nodes)

        assignment = torch_cluster.graclus_cluster(
            edge_index[0], edge_index[1], edge_weight, num_nodes
        )
        # relabel nodes
        ids, assignment = torch.unique(assignment, sorted=True, return_inverse=True)
        num_supernodes = ids.size(0)

        so = SelectOutput(
            node_index=torch.arange(num_nodes, device=assignment.device),
            num_nodes=num_nodes,
            cluster_index=assignment,
            num_supernodes=num_supernodes,
            s_inv_op=self.s_inv_op,
        )

        return so

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s_inv_op={self.s_inv_op})"
