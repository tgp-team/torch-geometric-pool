from typing import Optional

import torch
from torch import Tensor
from torch import nn

from tgp.select import SelectOutput


class IdentityReduce(nn.Module):
    """Identity reduce operator that passes features unchanged."""
    
    def __init__(self):
        super().__init__()
    
    def reset_parameters(self):
        """Reset parameters (no-op for identity)."""
        pass
    
    def forward(
        self,
        x: Tensor,
        so: SelectOutput,
        batch: Optional[Tensor] = None,
        **kwargs,
    ) -> tuple[Tensor, Optional[Tensor]]:
        """Pass features unchanged and preserve batch information."""
        batch_pool = batch
        return x, batch_pool
    
    @staticmethod
    def reduce_batch(so: SelectOutput, batch: Optional[Tensor]) -> Optional[Tensor]:
        """Reduce batch for identity mapping (no change)."""
        return batch
    
    def __repr__(self) -> str:
        return f"{self.__class__.__name__}()"
