import torch
import torch.nn as nn
import torch.nn.functional as F
import math

def get_model_device(model):
    return next(iter(model.parameters())).device

class RMSNorm(nn.Module):
    def __init__(self, hidden_size: int, eps: float = 1e-6):
        super().__init__()
        self.eps = eps
        self.weight = nn.Parameter(torch.ones(hidden_size))

    def forward(self, x):
        variance = x.pow(2).mean(-1, keepdim=True)
        x = x * torch.rsqrt(variance + self.eps)
        return self.weight * x

class MLSTMCell(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 8, gate_soft_cap: float = 30.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.eps = 1e-6
        self.gate_soft_cap = gate_soft_cap

        self.igate_proj = nn.Linear(3 * hidden_size, num_heads, bias=True)
        self.fgate_proj = nn.Linear(3 * hidden_size, num_heads, bias=True)
        self.outnorm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_size)
        
        nn.init.constant_(self.igate_proj.bias, -10.0)

    def forward(self, q: torch.Tensor, k: torch.Tensor, v: torch.Tensor, state):
        batch_size, hidden_size = q.shape
        cell_state, norm_state, max_state = state

        qkv_cat = torch.cat([q, k, v], dim=-1)
        igate_preact = self.igate_proj(qkv_cat)
        fgate_preact = self.fgate_proj(qkv_cat)
        
        igate_preact = self.gate_soft_cap * torch.tanh(igate_preact / self.gate_soft_cap)
        fgate_preact = self.gate_soft_cap * torch.tanh(fgate_preact / self.gate_soft_cap)

        q = q.view(batch_size, self.num_heads, self.head_size)
        k = k.view(batch_size, self.num_heads, self.head_size)
        v = v.view(batch_size, self.num_heads, self.head_size)

        k = k / math.sqrt(self.head_size)

        log_f = torch.nn.functional.logsigmoid(fgate_preact)
        max_new = torch.maximum(igate_preact, max_state + log_f)

        i_gate = torch.exp(igate_preact - max_new)
        f_gate = torch.exp(log_f + max_state - max_new)

        k_expanded = k.unsqueeze(-1)
        v_expanded = v.unsqueeze(-2)
        kv_outer = k_expanded * v_expanded

        cell_new = f_gate[:, :, None, None] * cell_state + i_gate[:, :, None, None] * kv_outer
        norm_new = f_gate[:, :, None] * norm_state + i_gate[:, :, None] * k

        q_bmm = q.reshape(-1, 1, self.head_size)
        cell_bmm = cell_new.reshape(-1, self.head_size, self.head_size)
        
        numerator_bmm = torch.bmm(q_bmm, cell_bmm)
        numerator = numerator_bmm.view(batch_size, self.num_heads, self.head_size)

        qn_dotproduct = (q * norm_new).sum(dim=-1)
        max_val = torch.exp(-max_new)
        denominator = torch.maximum(qn_dotproduct.abs(), max_val) + self.eps
        
        out = numerator / denominator[:, :, None]
        out = self.outnorm(out.view(batch_size, self.hidden_size))

        return out, (cell_new, norm_new, max_new)

    def init_state(self, batch_size: int, device: torch.device):
        dtype_accum = torch.float32
        return (
            torch.zeros(
                batch_size,
                self.num_heads,
                self.head_size,
                self.head_size,
                dtype=dtype_accum,
                device=device,
            ),
            torch.zeros(batch_size, self.num_heads, self.head_size, dtype=dtype_accum, device=device),
            torch.zeros(batch_size, self.num_heads, dtype=dtype_accum, device=device),
        )

class CausalConv1d(nn.Module):
    def __init__(self, hidden_size, kernel_size):
        super().__init__()
        self.hidden_size = hidden_size
        self.kernel_size = kernel_size
        self.conv = nn.Conv1d(
            hidden_size, hidden_size, kernel_size, groups=hidden_size, bias=True
        )
        # Cache weights without using them yet - will be computed on first forward
        self._w_rev = None

    def init_state(self, batch_size: int, device: torch.device | None = None):
        if device is None:
            device = get_model_device(self)
        return torch.zeros(
            batch_size, self.hidden_size, self.kernel_size - 1, device=device
        )

    def forward(self, x: torch.Tensor, state: torch.Tensor):
        # Use original approach for now - the main optimizations are elsewhere
        x_with_state = torch.cat([state, x[:, :, None]], dim=-1)
        out = self.conv(x_with_state)
        new_state = x_with_state[:, :, 1:]
        return out.squeeze(-1), new_state

class BlockLinear(nn.Module):
    def __init__(self, num_blocks: int, hidden_size: int, bias: bool = True):
        super().__init__()
        self.num_blocks = num_blocks
        self.block_size = hidden_size // num_blocks
        self.hidden_size = hidden_size
        self.weight = nn.Parameter(
            torch.empty(num_blocks, self.block_size, self.block_size)
        )
        if bias:
            self.bias = nn.Parameter(torch.empty(self.hidden_size))
        else:
            self.bias = None
        
        nn.init.xavier_uniform_(self.weight)

    def forward(self, x):
        batch_size = x.shape[0]
        assert x.shape[1] == self.hidden_size
        
        x = x.contiguous().reshape(batch_size, self.num_blocks, self.block_size)
        out = torch.einsum("bnh,nkh->bnk", x, self.weight)
        out = out.reshape(batch_size, self.hidden_size)
        
        if self.bias is not None:
            out.add_(self.bias)
        return out

class MLSTMBlock(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        num_heads: int = 8,
        conv_kernel_size: int = 4,
        qkv_proj_block_size: int = 4,
        expand_factor: int = 2,
        gate_soft_cap: float = 30.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.inner_size = expand_factor * hidden_size

        self.norm = RMSNorm(hidden_size)

        self.x_proj = nn.Linear(hidden_size, self.inner_size, bias=False)
        self.gate_proj = nn.Linear(hidden_size, self.inner_size, bias=False)

        # Use simpler fused projection instead of grouped conv
        self.qkv_proj = nn.Linear(self.inner_size, 3 * self.inner_size, bias=False)

        self.conv1d = CausalConv1d(self.inner_size, kernel_size=conv_kernel_size)

        self.mlstm_cell = MLSTMCell(self.inner_size, num_heads, gate_soft_cap=gate_soft_cap)
        self.proj_down = nn.Linear(self.inner_size, hidden_size, bias=False)
        self.learnable_skip = nn.Parameter(torch.ones(self.inner_size))

    def forward(self, x: torch.Tensor, state):
        conv_state, recurrent_state = state

        skip = x

        x = self.norm(x)
        x_mlstm = self.x_proj(x)
        x_gate = self.gate_proj(x)

        x_conv, new_conv_state = self.conv1d(x_mlstm, conv_state)
        x_mlstm_conv = F.silu(x_conv)

        # Fused QKV projection
        qkv = self.qkv_proj(x_mlstm_conv)
        q, k, v = qkv.chunk(3, dim=1)

        mlstm_out, new_recurrent_state = self.mlstm_cell(q, k, v, recurrent_state)

        mlstm_out_skip = mlstm_out + (self.learnable_skip * x_mlstm_conv)
        h_state = mlstm_out_skip * F.silu(x_gate)
        y = self.proj_down(h_state)

        return y + skip, (new_conv_state, new_recurrent_state)

    def init_state(self, batch_size: int, device: torch.device):
        return (
            self.conv1d.init_state(batch_size, device),
            self.mlstm_cell.init_state(batch_size, device),
        )

class SLSTMCell(nn.Module):
    def __init__(self, hidden_size: int, num_heads: int = 4, gate_soft_cap: float = 30.0):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads
        self.head_size = hidden_size // num_heads
        self.eps = 1e-6
        self.gate_soft_cap = gate_soft_cap

    def forward(
        self,
        i: torch.Tensor,
        f: torch.Tensor,
        z: torch.Tensor,
        o: torch.Tensor,
        state,
    ):
        cell_state, norm_state, max_state = state

        i = self.gate_soft_cap * torch.tanh(i / self.gate_soft_cap)
        f = self.gate_soft_cap * torch.tanh(f / self.gate_soft_cap)
        z = self.gate_soft_cap * torch.tanh(z / self.gate_soft_cap)
        o = self.gate_soft_cap * torch.tanh(o / self.gate_soft_cap)

        log_f_plus_m = max_state + torch.nn.functional.logsigmoid(f)
        max_new = torch.maximum(i, log_f_plus_m)

        o_gate = torch.sigmoid(o)
        i_gate = torch.exp(i - max_new)
        f_gate = torch.exp(log_f_plus_m - max_new)

        cell_new = f_gate * cell_state + i_gate * torch.tanh(z)
        norm_new = f_gate * norm_state + i_gate
        y_new = o_gate * cell_new / (norm_new + self.eps)

        return y_new, (cell_new, norm_new, max_new)

    def init_state(self, batch_size: int, device: torch.device):
        dtype_accum = torch.float32
        return (
            torch.zeros(batch_size, self.hidden_size, dtype=dtype_accum, device=device),
            torch.zeros(batch_size, self.hidden_size, dtype=dtype_accum, device=device),
            torch.full((batch_size, self.hidden_size), float("-inf"), dtype=dtype_accum, device=device),
        )

class SLSTMBlock(nn.Module):
    def __init__(
        self, 
        hidden_size: int, 
        num_heads: int = 4, 
        conv_kernel_size: int = 4,
        gate_soft_cap: float = 30.0,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_heads = num_heads

        self.norm = RMSNorm(hidden_size)
        self.conv1d = CausalConv1d(hidden_size, kernel_size=conv_kernel_size)
        
        self.in_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=False)
        self.state_gates = nn.Linear(hidden_size, 4 * hidden_size, bias=True)
        
        with torch.no_grad():
            self.state_gates.bias[0::4] = -10.0

        self.slstm_cell = SLSTMCell(hidden_size, num_heads, gate_soft_cap=gate_soft_cap)
        self.group_norm = nn.GroupNorm(num_groups=num_heads, num_channels=hidden_size)

    def _forward_impl(self, x, state):
        conv_state, recurrent_state, slstm_state = state

        skip = x
        x = self.norm(x)

        x_conv, new_conv_state = self.conv1d(x, conv_state)
        x_conv_act = F.silu(x_conv)

        g_in = self.in_gates(x_conv_act)
        g_st = self.state_gates(recurrent_state)
        i, f, z, o = (g_in + g_st).chunk(4, dim=1)

        new_recurrent_state, new_slstm_state = self.slstm_cell(i, f, z, o, slstm_state)
        
        slstm_out = self.group_norm(new_recurrent_state)

        return slstm_out + skip, (new_conv_state, new_recurrent_state, new_slstm_state)
    
    def forward(self, x: torch.Tensor, state):
        return self._forward_impl(x, state)

    def init_state(self, batch_size: int, device: torch.device):
        return (
            self.conv1d.init_state(batch_size, device),
            torch.zeros(batch_size, self.hidden_size, device=device),
            self.slstm_cell.init_state(batch_size, device),
        )

class OptimizedxLSTM(nn.Module):
    def __init__(
        self,
        vocab_size: int,
        hidden_size: int,
        num_layers: int = 6,
        num_heads: int = 8,
        mlstm_layers: list = None,
        conv_kernel_size: int = 4,
        qkv_proj_block_size: int = 4,
        expand_factor: int = 2,
        use_gradient_checkpointing: bool = False,
        gate_soft_cap: float = 30.0,
        dropout: float = 0.1,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.num_layers = num_layers
        self.use_gradient_checkpointing = use_gradient_checkpointing
        
        if mlstm_layers is None:
            mlstm_layers = [i % 2 == 0 for i in range(num_layers)]
        self.mlstm_layers = mlstm_layers
        
        self.embedding = nn.Embedding(vocab_size, hidden_size)
        self.dropout = nn.Dropout(dropout)
        
        self.layers = nn.ModuleList()
        for i in range(num_layers):
            if mlstm_layers[i]:
                layer = MLSTMBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads,
                    conv_kernel_size=conv_kernel_size,
                    qkv_proj_block_size=qkv_proj_block_size,
                    expand_factor=expand_factor,
                    gate_soft_cap=gate_soft_cap,
                )
            else:
                layer = SLSTMBlock(
                    hidden_size=hidden_size,
                    num_heads=num_heads // 2,
                    conv_kernel_size=conv_kernel_size,
                    gate_soft_cap=gate_soft_cap,
                )
            self.layers.append(layer)
            
        self.final_norm = RMSNorm(hidden_size)
        self.lm_head = nn.Linear(hidden_size, vocab_size, bias=False)
        
        self.apply(self._init_weights)
        
    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.xavier_uniform_(module.weight)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, std=0.02)
        elif isinstance(module, RMSNorm):
            nn.init.ones_(module.weight)
    
    def init_states(self, batch_size: int, device: torch.device):
        states = []
        for i, layer in enumerate(self.layers):
            states.append(layer.init_state(batch_size, device))
        return states
    
    def forward(self, input_ids: torch.Tensor, states=None, return_states=False):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        if input_ids.dim() == 1:
            input_ids = input_ids.unsqueeze(-1)
        seq_len = input_ids.shape[1]
        
        if states is None:
            states = self.init_states(batch_size, device)
        
        x = self.embedding(input_ids)
        x = self.dropout(x)
        
        # Process sequence token by token
        for t in range(seq_len):
            x_t = x[:, t]
            new_layer_states = []
            
            for i, layer in enumerate(self.layers):
                x_t, new_state = layer(x_t, states[i])
                new_layer_states.append(new_state)
            
            # Update states after processing all layers
            states = new_layer_states
        
        x_out = self.final_norm(x_t)
        logits = self.lm_head(x_out)
        
        if return_states:
            return logits, states
        return logits

    @torch.no_grad()
    def generate(
        self, 
        input_ids: torch.Tensor,
        max_length: int = 100,
        temperature: float = 1.0,
        top_k: int = None,
        top_p: float = None,
    ):
        batch_size = input_ids.shape[0]
        device = input_ids.device
        
        states = self.init_states(batch_size, device)
        
        for i in range(input_ids.shape[1]):
            token = input_ids[:, i]
            logits, states = self.forward(token, states, return_states=True)
        
        generated = input_ids.clone()
        
        for _ in range(max_length - input_ids.shape[1]):
            logits, states = self.forward(generated[:, -1], states, return_states=True)
            
            logits = logits / temperature
            
            if top_k is not None:
                v, _ = torch.topk(logits, top_k)
                logits[logits < v[:, [-1]]] = float('-inf')
            
            if top_p is not None:
                sorted_logits, sorted_indices = torch.sort(logits, descending=True)
                cumulative_probs = torch.cumsum(F.softmax(sorted_logits, dim=-1), dim=-1)
                sorted_indices_to_remove = cumulative_probs > top_p
                sorted_indices_to_remove[:, 1:] = sorted_indices_to_remove[:, :-1].clone()
                sorted_indices_to_remove[:, 0] = 0
                logits[sorted_indices_to_remove.scatter(1, sorted_indices, sorted_indices_to_remove)] = float('-inf')
            
            probs = F.softmax(logits, dim=-1)
            next_token = torch.multinomial(probs, num_samples=1)
            
            generated = torch.cat([generated, next_token], dim=1)
        
        return generated

def create_optimized_xlstm(
    vocab_size: int,
    hidden_size: int = 512,
    num_layers: int = 6,
    compile_model: bool = True,
    use_mixed_precision: bool = True,
    **kwargs
):
    if torch.cuda.is_available():
        torch.backends.cuda.matmul.allow_tf32 = True
        torch.backends.cudnn.allow_tf32 = True
    
    model = OptimizedxLSTM(
        vocab_size=vocab_size,
        hidden_size=hidden_size,
        num_layers=num_layers,
        **kwargs
    )
    
    if compile_model and hasattr(torch, 'compile'):
        model = torch.compile(
            model, 
            mode="reduce-overhead", 
            fullgraph=True, 
            dynamic=False
        )
    
    return model

# Create a compatible XLSTM class that matches the original interface
class XLSTM(nn.Module):
    def __init__(
        self,
        hidden_size: int,
        mlstm_num_heads: int = 8,
        slstm_num_heads: int = 4,
    ):
        super().__init__()
        self.hidden_size = hidden_size
        self.mlstm_num_heads = mlstm_num_heads
        self.slstm_num_heads = slstm_num_heads

        # Use the optimized blocks
        self.slstm = SLSTMBlock(
            hidden_size=hidden_size,
            num_heads=slstm_num_heads,
            conv_kernel_size=4,
            gate_soft_cap=30.0,
        )
        self.mlstm = MLSTMBlock(
            hidden_size=hidden_size,
            num_heads=mlstm_num_heads,
            conv_kernel_size=4,
            qkv_proj_block_size=4,
            expand_factor=2,
            gate_soft_cap=30.0,
        )
        self.final_norm = RMSNorm(hidden_size)  # Use optimized RMSNorm

    def forward(self, x: torch.Tensor, state):
        slstm_state, mlstm_state = state
        x, new_slstm_state = self.slstm(x, slstm_state)
        x, new_mlstm_state = self.mlstm(x, mlstm_state)
        x = self.final_norm(x)
        return x, (new_slstm_state, new_mlstm_state)

    def init_state(self, batch_size: int, device: torch.device):
        slstm_state = self.slstm.init_state(batch_size, device)
        mlstm_state = self.mlstm.init_state(batch_size, device)
        return (slstm_state, mlstm_state)
