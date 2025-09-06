"""
Optimized xLSTM layer that's a drop-in replacement for the baseline XLSTM
in the inference server. This maintains the same interface but uses our 
optimized implementation internally.
"""

import torch
import torch.nn as nn
from xlstm import OptimizedxLSTM, create_optimized_xlstm

class OptimizedXLSTMLayer(nn.Module):    
    def __init__(self, hidden_size: int, mlstm_num_heads: int = 8, slstm_num_heads: int = 4):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlstm_num_heads = mlstm_num_heads
        self.slstm_num_heads = slstm_num_heads
        
        self.mlstm_block = None
        self.slstm_block = None
        self._init_optimized_blocks()
        
    def _init_optimized_blocks(self):
        from xlstm import MLSTMBlock, SLSTMBlock
        
        self.mlstm_block = MLSTMBlock(
            hidden_size=self.hidden_size,
            num_heads=self.mlstm_num_heads,
            conv_kernel_size=4,
            qkv_proj_block_size=4,
            expand_factor=2,
            gate_soft_cap=30.0,
        )
        
        self.slstm_block = SLSTMBlock(
            hidden_size=self.hidden_size, 
            num_heads=self.slstm_num_heads,
            conv_kernel_size=4,
            gate_soft_cap=30.0,
        )
    
    def init_state(self, batch_size: int, device: torch.device):
        slstm_state = self.slstm_block.init_state(batch_size, device)
        mlstm_state = self.mlstm_block.init_state(batch_size, device)
        return (slstm_state, mlstm_state)
    
    def forward(self, x: torch.Tensor, state) -> tuple[torch.Tensor, tuple]:
        slstm_state, mlstm_state = state
        
        x, new_slstm_state = self.slstm_block(x, slstm_state)
        x, new_mlstm_state = self.mlstm_block(x, mlstm_state)
        
        return x, (new_slstm_state, new_mlstm_state)


def create_optimized_xlstm_layer(hidden_size: int, num_heads: int = 8):
    return OptimizedXLSTMLayer(
        hidden_size=hidden_size,
        mlstm_num_heads=num_heads,
        slstm_num_heads=num_heads // 2,
    )
