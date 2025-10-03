from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_sparse import SparseTensor

from tgp.select import SelectOutput


class IdentitySelect(torch.nn.Module):
    """Identity select operator that maps each node to itself (no pooling)."""

    def __init__(self):
        super().__init__()
        self.s_inv_op = "transpose"

    def reset_parameters(self):
        """Reset parameters (no-op for identity)."""
        pass

    def forward(
        self,
        x: Optional[Tensor] = None,
        edge_index: Optional[Adj] = None,
        edge_weight: Optional[Tensor] = None,
        batch: Optional[Tensor] = None,
        num_nodes: Optional[int] = None,
        **kwargs,
    ) -> SelectOutput:
        """Create identity mapping where each node maps to itself."""
        if num_nodes is None:
            if x is not None:
                num_nodes = x.size(0)
            elif edge_index is not None:
                if isinstance(edge_index, SparseTensor):
                    num_nodes = edge_index.size(0)
                else:
                    num_nodes = edge_index.max().item() + 1
            else:
                raise ValueError("Cannot determine num_nodes from inputs")

        # Create identity matrix: each node maps to itself
        # For sparse representation, create cluster_index = node_index = [0, 1, 2, ..., num_nodes-1]
        node_index = torch.arange(num_nodes, device=x.device if x is not None else None)
        cluster_index = torch.arange(
            num_nodes, device=x.device if x is not None else None
        )

        return SelectOutput(
            node_index=node_index,
            num_nodes=num_nodes,
            cluster_index=cluster_index,
            num_supernodes=num_nodes,
            s_inv_op=self.s_inv_op,
        )

    def __repr__(self) -> str:
        return f"{self.__class__.__name__}(s_inv_op={self.s_inv_op})"
