from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.select import Select, SelectOutput


class IdentitySelect(Select):
    """Identity select operator that maps each node to itself (no pooling)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        *,
        x: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        **kwargs,
    ) -> SelectOutput:
        """Create identity mapping where each node maps to itself."""
        if x is None and edge_index is None:
            raise ValueError("x and edge_index cannot both be None")

        if x is not None:
            num_nodes = x.size(0)
            device = x.device
        else:
            num_nodes = maybe_num_nodes(edge_index)
            if isinstance(edge_index, SparseTensor):
                device = edge_index.device()
            else:
                device = edge_index.device
        # Create identity matrix: each node maps to itself

        node_index = torch.arange(num_nodes, device=device)
        cluster_index = torch.arange(num_nodes, device=device)

        return SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            cluster_index=cluster_index,
            num_supernodes=num_nodes,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
