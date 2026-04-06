# refer to: 
# 1. [karpathy/nano-llama31](https://github.com/karpathy/nano-llama31/blob/master/llama31.py)
# 2. [GeeeekExplorer/nano-vllm](https://github.com/GeeeekExplorer/nano-vllm/blob/main/nanovllm/layers/rotary_embedding.py)
# for simpler implementation, remove scaling function
# 业界主流实现, 旋转对使用半截配对, 即 (x_0, x_{d/2}), (x_1, x_{d/2+1}), ...

import torch

def precompute_freqs_cis(head_dim: int, max_len: int, theta: float = 10000.0) -> torch.Tensor:
    freqs = 1.0 / (theta ** (torch.arange(0, head_dim, 2).float() / head_dim)) # shape (head_dim/2,)
    t = torch.arange(max_len, device=freqs.device).float() # shape (max_len,)
    freqs = torch.outer(t, freqs) # equal to einsum('i,j->ij', t, freqs), shape (max_len, head_dim/2)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64, equal to torch.complex(torch.cos(freqs), torch.sin(freqs))
    cos_sin = torch.cat([freqs_cis.real, freqs_cis.imag], dim=-1) # [cos, sin] shape (max_len, head_dim/2, 2)
    return cos_sin
    
def apply_rotary_emb(x:torch.Tensor, cos_sin:torch.Tensor):
    x1, x2 = torch.chunk(x, 2, dim=-1)
    cos, sin = torch.chunk(cos_sin, 2, dim=-1)
    x_out = torch.stack([x1 * cos - x2 * sin, 
                         x1 * sin + x2 * cos], dim=-1)
    return x_out.reshape(*x.shape).type_as(x)

class RotaryPositionalEncoding(torch.nn.Module):
    def __init__(self, d_model:int, max_len:int, theta=10000.0):
        """
        the d_model here in the refered code is head_dim here(head_dim = d_model//n_heads)
        but I have used d_model here as the head_dim
        """
        super(RotaryPositionalEncoding, self).__init__()
        assert d_model % 2 == 0, 'd_model should be even'
        self.d_model = d_model
        self.register_buffer('cos_sin', precompute_freqs_cis(d_model, max_len, theta))

    def forward(self, x:torch.Tensor):
        # x is (**, seq_len, d_model)
        return apply_rotary_emb(x, self.cos_sin[:x.size(-2)])
