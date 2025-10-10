from typing import Optional

from torch import Tensor
from torch_geometric.typing import Adj

from tgp.connect import SparseConnect
from tgp.lift import BaseLift
from tgp.reduce import BaseReduce
from tgp.select import IdentitySelect, SelectOutput
from tgp.src import BasePrecoarseningMixin, PoolingOutput, SRCPooling


class NoPool(BasePrecoarseningMixin, SRCPooling):
    r"""Identity pooling operator that performs no actual pooling.
    This pooler creates a consistent SelectOutput and PoolingOutput structure
    but doesn't perform any actual pooling - each node maps to itself and
    all features and edges are preserved unchanged.
    """

    def __init__(
        self,
    ):
        super().__init__(
            selector=IdentitySelect(),
            reducer=BaseReduce(reduce_op="sum"),
            lifter=BaseLift(matrix_op="precomputed", reduce_op="sum"),
            connector=SparseConnect(reduce_op="sum", remove_self_loops=False),
        )

    def forward(
        self,
        x: Tensor,
        adj: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        so: Optional[SelectOutput] = None,
        batch: Optional[Tensor] = None,
        lifting: bool = False,
        **kwargs,
    ) -> PoolingOutput:
        r"""Forward pass.

        Args:
            x (~torch.Tensor): The node feature matrix of shape :math:`[N, F]`,
                where :math:`N` is the number of nodes in the batch and
                :math:`F` is the number of node features.
            adj (~torch_geometric.typing.Adj, optional): The connectivity matrix.
                (default: :obj:`None`)
            edge_weight (~torch.Tensor, optional): A vector of shape
                :math:`[E]` containing the weights of the edges.
                (default: :obj:`None`)
            so (~tgp.select.SelectOutput, optional): The output of the :math:`\texttt{select}` operator.
                (default: :obj:`None`)
            batch (~torch.Tensor, optional): The batch vector
                :math:`\mathbf{b} \in {\{ 0, \ldots, B-1\}}^N`, which indicates
                to which graph in the batch each node belongs. (default: :obj:`None`)
            lifting (bool, optional): If set to :obj:`True`, the :math:`\texttt{lift}` operation is performed.
                (default: :obj:`False`)

        Returns:
            PoolingOutput or Tensor: The output of the pooling operator.
        """
        if lifting:
            # Lift - for identity pooling, this just returns the input
            x_lifted = self.lift(x_pool=x, so=so)
            return x_lifted
        else:
            # Select - create identity mapping
            so = self.select(x=x, edge_index=adj)

            # Reduce - pass features unchanged
            x_pooled, batch_pooled = x, batch

            # Connect - pass edges unchanged
            edge_index_pooled, edge_weight_pooled = adj, edge_weight

            out = PoolingOutput(
                x=x_pooled,
                edge_index=edge_index_pooled,
                edge_weight=edge_weight_pooled,
                batch=batch_pooled,
                so=so,
            )
            return out

    def precoarsening(
        self,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        *,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **select_kwargs,
    ) -> PoolingOutput:
        """Precoarsening for NoPool - returns identity mapping with features."""
        so = self.select(
            edge_index=edge_index,
            edge_weight=edge_weight,
            batch=batch,
            num_nodes=num_nodes,
            **select_kwargs,
        )
        batch_pooled = batch
        edge_index_pooled, edge_weight_pooled = edge_index, edge_weight

        return PoolingOutput(
            edge_index=edge_index_pooled,
            edge_weight=edge_weight_pooled,
            batch=batch_pooled,
            so=so,
        )
