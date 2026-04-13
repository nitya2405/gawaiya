from __future__ import annotations

import torch
from torch import nn


class RotaryEmbedding(nn.Module):
    def __init__(self, dim: int, *, theta: float = 10000.0) -> None:
        super().__init__()
        if dim % 2 != 0:
            raise ValueError("RoPE dim must be even")
        inv_freq = 1.0 / (theta ** (torch.arange(0, dim, 2).float() / dim))
        self.register_buffer("inv_freq", inv_freq, persistent=False)

    def get_sin_cos(self, seq_len: int, *, device: torch.device, dtype: torch.dtype, offset: int = 0):
        t = torch.arange(offset, offset + seq_len, device=device, dtype=self.inv_freq.dtype)
        freqs = torch.einsum("i,j->ij", t, self.inv_freq)  # [T, dim/2]
        sin = freqs.sin().to(dtype=dtype)
        cos = freqs.cos().to(dtype=dtype)
        return sin, cos


def apply_rope(x: torch.Tensor, sin: torch.Tensor, cos: torch.Tensor) -> torch.Tensor:
    """
    Apply rotary position embedding to a tensor x shaped [..., T, D].
    sin/cos are [T, D/2].
    """
    t = x.shape[-2]
    d = x.shape[-1]
    x_ = x.view(*x.shape[:-1], d // 2, 2)
    x1 = x_[..., 0]
    x2 = x_[..., 1]

    sin = sin[:t].unsqueeze(-2)  # [..., T, 1, D/2] after broadcasting
    cos = cos[:t].unsqueeze(-2)

    # Broadcast sin/cos to match x1/x2: [..., T, D/2]
    sin = sin.squeeze(-2)
    cos = cos.squeeze(-2)

    y1 = x1 * cos - x2 * sin
    y2 = x1 * sin + x2 * cos
    y = torch.stack((y1, y2), dim=-1).flatten(-2)
    return y

