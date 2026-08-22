"""The write path: one recurrent step per frame.

`S <- WriteTransformer(Q=S, KV=frame tokens)`, applied independently per tap
(V3 partitioned). The four streams never interact, so this is one module
instantiated four times.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttention, LayerScale, Mlp, SelfAttention


class WriteBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0,
                 qk_norm: bool = True, residual_gate: bool = False):
        super().__init__()
        self.norm_s_cross = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads, qk_norm=qk_norm)
        self.norm_s_self = nn.LayerNorm(dim)
        self.self_attn = SelfAttention(dim, num_heads, qk_norm=qk_norm)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)

        def gate():
            return LayerScale(dim) if residual_gate else nn.Identity()

        self.ls_cross, self.ls_self, self.ls_mlp = gate(), gate(), gate()

    def forward(self, s: torch.Tensor, kv: torch.Tensor) -> torch.Tensor:
        s = s + self.ls_cross(self.cross(self.norm_s_cross(s), self.norm_kv(kv)))
        s = s + self.ls_self(self.self_attn(self.norm_s_self(s)))
        s = s + self.ls_mlp(self.mlp(self.norm_mlp(s)))
        return s


class WriteTransformer(nn.Module):
    """Pre-norm residual blocks, plus a LayerNorm on the state itself.

    The output norm is not cosmetic. Each block adds `f(LN(s))`, whose magnitude
    is independent of ||s|| , so over a long clip those residuals accumulate:
    measured without it, a 128-frame clip drove ||S|| from 14.5 to 40,206 (x2775)
    with no training at all. Normalising the state once per step bounds the
    recurrence regardless of clip length or layer count.
    """

    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 16,
                 mlp_ratio: float = 4.0, qk_norm: bool = True,
                 residual_gate: bool = False,
                 position_mode: str = "none",
                 state_norm: bool = True):
        super().__init__()
        if position_mode not in ("none", "pose", "xyz"):
            raise ValueError(f"unknown position_mode {position_mode!r}")
        # "none" is arm (a): the write sees only the frame's own tokens, so pose
        # is present but entangled inside the camera token. "pose"/"xyz" are the
        # arms that decode it explicitly -- wired now because retrofitting the
        # data path later is much worse than carrying a dead branch.
        self.position_mode = position_mode
        self.blocks = nn.ModuleList([
            WriteBlock(dim, num_heads, mlp_ratio, qk_norm, residual_gate)
            for _ in range(num_layers)
        ])
        self.out_norm = nn.LayerNorm(dim) if state_norm else nn.Identity()

    def forward(self, s: torch.Tensor, kv: torch.Tensor,
                position: torch.Tensor | None = None) -> torch.Tensor:
        """s: [B, N, D] state slots. kv: [B, P, D] one frame's tokens."""
        if self.position_mode != "none":
            if position is None:
                raise ValueError(f"position_mode={self.position_mode} needs `position`")
            kv = kv + position
        for blk in self.blocks:
            s = blk(s, kv)
        return self.out_norm(s)
