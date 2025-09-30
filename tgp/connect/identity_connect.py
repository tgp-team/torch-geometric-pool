from typing import Optional

import torch
from torch import Tensor
from torch_geometric.typing import Adj

from tgp.select import SelectOutput


class IdentityConnect(torch.nn.Module):
    """Identity connect operator that preserves original edges."""
    
    def __init__(self):
        super().__init__()
    
    def reset_parameters(self):
        """Reset parameters (no-op for identity)."""
        pass
    
    def forward(
        self,
        edge_index: Adj,
        edge_weight: Optional[Tensor] = None,
        so: SelectOutput = None,
        **kwargs,
    ) -> tuple[Adj, Optional[Tensor]]:
        """Pass edges unchanged."""
        return edge_index, edge_weight
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
