from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj
from torch_geometric.utils.num_nodes import maybe_num_nodes
from torch_sparse import SparseTensor

from tgp.select import Select, SelectOutput


def get_device(
    x: Optional[Tensor] = None, edge_index: Optional[Adj] = None
) -> torch.device:
    if edge_index is not None:
        if isinstance(edge_index, SparseTensor):
            return edge_index.device()
        else:
            return edge_index.device
    elif x is not None:
        return x.device
    else:
        raise ValueError("No device found")


class IdentitySelect(Select):
    """Identity select operator that maps each node to itself (no pooling)."""

    def __init__(self):
        super().__init__()

    def forward(
        self,
        *,
        edge_index: Optional[Adj] = None,
        num_nodes: Optional[int] = None,
        device: Optional[torch.device] = None,
        **kwargs,
    ) -> SelectOutput:
        """Create identity mapping where each node maps to itself.

        Args:
            edge_index (Optional[Adj]): The edge index of the graph.
            num_nodes (Optional[int]): The number of nodes in the graph.
                If not provided, it will be inferred from the edge_index.
            device (Optional[torch.device]): The device to use for the output tensors.
                If not provided, it will be inferred from the edge_index.

        Returns:
            SelectOutput: The output of the identity select operator.
        """
        num_nodes = maybe_num_nodes(edge_index, num_nodes)
        device = device if device is not None else get_device(edge_index=edge_index)

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
