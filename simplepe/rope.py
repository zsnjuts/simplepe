import torch
import torch.nn as nn

# refer to: https://github.com/karpathy/nano-llama31/blob/master/llama31.py
# for simpler implementation, remove scaling function
# 论文标准实现, 旋转对使用相邻配对, 即 (x0, x1), (x2, x3), ...

def precompute_freqs_cis(dim: int, end: int, theta: float = 10000.0) -> torch.Tensor:
    """提前生成公式中的 $\cos(m \theta_i)$ 和 $\sin(m * \theta_i)$
    其中 $\theta_i = 10000^{-2i/D}$, i 为维度的组号, D 为维度数量
    :param dim token 的 hidden vector 长度
    :param end token 序列长度
    :param theta $\theta_i$ 公式中的底数 10000
    """
    # 1. 计算每一组维度的基础频率 theta_i
    # 公式: theta_i = 10000^(-2i/D), i 从 0 到 D/2
    # 这里 torch.arange(0, dim, 2) 后面的 [: (dim // 2)] 是为了处理 dim 为奇数的情况
    #   dim 为偶数时, 前者得到的数组长度为 dim//2, 无需截断;
    #   dim 为奇数时, 前者得到的数组长度为 dim//2 + 1, 需要截断到 dim//2
    # freqs 长度为 dim // 2
    freqs = 1.0 / (theta ** (torch.arange(0, dim, 2)[: (dim // 2)].float() / dim))
    # 2. 生成序列位置 m (0, 1, ..., end-1)
    t = torch.arange(end, device=freqs.device, dtype=torch.float32)
    # 3. 计算外积，得到所有位置、所有维度的转角 m * theta_i
    # 结果 freqs 形状 (end, dim//2)
    # freqs[m, i] = m * theta_i
    freqs = torch.outer(t, freqs)
    # 4. 利用极坐标构造复数: 1 * exp(i * freqs) = cos(freqs) + i*sin(freqs)
    freqs_cis = torch.polar(torch.ones_like(freqs), freqs)  # complex64
    # 5. 为了方便 forward 计算，将复数的实部和虚部在最后一个维度拼接
    # 结果形状: (max_len, dim//2, 2) -> 最后一维 0 是 cos, 1 是 sin
    freqs_cis_real = torch.stack([freqs_cis.real, freqs_cis.imag], dim=-1)
    return freqs_cis_real

class RotaryPositionalEncoding(nn.Module):
    def __init__(self, d_model:int, max_len:int, theta=10000.0):
        """
        the d_model here in the refered code is head_dim here(head_dim = d_model//n_heads)
        but I have used d_model here as the head_dim
        """
        super(RotaryPositionalEncoding, self).__init__()
        # 每一个位置编码是两两成对旋转的，所以维度必须是偶数
        assert d_model % 2 == 0, 'd_model should be even'
        self.d_model = d_model
        # 将预计算好的 cos/sin 矩阵注册为 buffer（不参与梯度更新）
        self.register_buffer('freqs_cis', precompute_freqs_cis(d_model, max_len, theta))

    def forward(self, x:torch.Tensor):
        # x is (**, seq_len, d_model)

        # 1. 重塑形状，将最后一个维度两两分组
        # (..., seq_len, d_model/2, 2) -> 这里的 2 代表复数的实部和虚部 [x0, x1]
        xshaped = x.reshape(*x.shape[:-1], -1, 2) # -> (**, seq_len, d_model/2, 2)
        # 2. 截取对应序列长度的位置频率，并确保设备一致
        # freqs_cis is (seq_len, d_model/2, 2)
        freqs_cis = self.freqs_cis[:x.size(-2)].to(x.device)
        # 3. 核心计算：复数乘法展开
        # 假设 x = x0 + i*x1, freqs_cis = cos + i*sin
        # 实部 (new_x0) = x0*cos - x1*sin
        # 虚部 (new_x1) = x0*sin + x1*cos
        x_out = torch.stack([
            xshaped[..., 0] * freqs_cis[..., 0] - xshaped[..., 1] * freqs_cis[..., 1],
            xshaped[..., 0] * freqs_cis[..., 1] + xshaped[..., 1] * freqs_cis[..., 0]
        ], dim=-1)
        # 4. 还原形状回 (batch, seq_len, d_model) 并保证数据类型一致
        return x_out.reshape(*x.shape).type_as(x)
