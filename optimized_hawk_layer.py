"""
Optimized Hawk layer that's a drop-in replacement for the baseline Hawk
in the inference server.
"""

import torch
import torch.nn as nn
from hawk import HawkOptimized

class OptimizedHawkLayer(nn.Module):
    """Drop-in replacement for baseline Hawk that uses optimized implementation"""
    
    def __init__(self, hidden_size: int):
        super().__init__()
        self.hidden_size = hidden_size
        self.hawk = HawkOptimized(hidden_size, conv_kernel_size=4)
        
    def init_state(self, batch_size: int, device: torch.device):
        """Initialize state matching baseline interface"""
        return self.hawk.init_state(batch_size, device)
    
    def forward(self, x: torch.Tensor, state) -> tuple[torch.Tensor, list]:
        """Forward pass matching baseline interface"""
        return self.hawk(x, state)


def create_optimized_hawk_layer(hidden_size: int):
    """Factory function that creates optimized Hawk layer"""
    return OptimizedHawkLayer(hidden_size)
