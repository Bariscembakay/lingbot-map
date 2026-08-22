"""Attention primitives for the memory modules.

Separate from `lingbot_map.layers.attention` because that class fuses q, k and v
from a single input; the write and read are both cross-attention between two
different token sets.

QK-norm is on by default to match the aggregator's frame/global blocks
(`AggregatorBase.__init__`, `qk_norm=True`). It matters more here than there:
the write is unrolled over hundreds of recurrent steps, so a slow drift in the
logit scale compounds.
"""
from __future__ import annotations

import torch
import torch.nn as nn
import torch.nn.functional as F


class CrossAttention(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, qkv_bias: bool = True,
                 qk_norm: bool = True):
        super().__init__()
        assert dim % num_heads == 0, f"{dim} not divisible by {num_heads} heads"
        self.num_heads = num_heads
        self.head_dim = dim // num_heads
        self.q = nn.Linear(dim, dim, bias=qkv_bias)
        self.kv = nn.Linear(dim, 2 * dim, bias=qkv_bias)
        self.proj = nn.Linear(dim, dim)
        norm = nn.LayerNorm if qk_norm else nn.Identity
        self.q_norm = norm(self.head_dim)
        self.k_norm = norm(self.head_dim)

    def _split(self, x: torch.Tensor) -> torch.Tensor:
        B, N, _ = x.shape
        return x.view(B, N, self.num_heads, self.head_dim).transpose(1, 2)

    def forward(self, x: torch.Tensor, ctx: torch.Tensor) -> torch.Tensor:
        """x: [B, Nq, D] queries. ctx: [B, Nk, D] keys/values."""
        B, Nq, D = x.shape
        q = self.q_norm(self._split(self.q(x)))
        k, v = self.kv(ctx).chunk(2, dim=-1)
        k = self.k_norm(self._split(k))
        v = self._split(v)
        out = F.scaled_dot_product_attention(q, k, v)
        return self.proj(out.transpose(1, 2).reshape(B, Nq, D))


class SelfAttention(CrossAttention):
    def forward(self, x: torch.Tensor, ctx: torch.Tensor | None = None) -> torch.Tensor:
        return super().forward(x, x if ctx is None else ctx)


class Mlp(nn.Module):
    def __init__(self, dim: int, ratio: float = 4.0):
        super().__init__()
        hidden = int(dim * ratio)
        self.fc1 = nn.Linear(dim, hidden)
        self.fc2 = nn.Linear(hidden, dim)

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.fc2(F.gelu(self.fc1(x)))


class LayerScale(nn.Module):
    """Zero-init per-channel residual gate.

    With `init=0` a freshly built module is the identity, so the state passes
    through untouched at step 0 and every later change is attributable.
    """

    def __init__(self, dim: int, init: float = 0.0):
        super().__init__()
        self.gamma = nn.Parameter(torch.full((dim,), init))

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return x * self.gamma
