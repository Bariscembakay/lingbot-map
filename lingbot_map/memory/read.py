"""The read path: refine the aggregator's last-layer tokens using the state.

Queries are the current frame's own tokens (Loss 1) or raymap tokens for a past
camera (Loss 2) -- the module is the same, only the query differs.

The residual gates are **always present and zero-initialised**, not optional:
they are what makes the untrained model bit-identical to published lingbot-map,
so every point of improvement is attributable. Cross-attention and MLP are gated
separately so the state path's contribution can be measured rather than inferred.

The residual is **relative**, not absolute. Measured on a real clip, the
aggregator's last-layer tokens have per-token norm ~463 while an untrained
cross-attention branch outputs ~1 -- a ratio of 0.0023, so an absolute residual
would need the gate to reach ~4.4 before it moved the token by even 1%, which at
lr 1e-4 is the entire training run. Rescaling the branch to the query's own RMS
makes the gate dimensionless: `gate = 0.01` is a 1% correction, and `gate = 0` is
still exactly the identity. The frozen head LayerNorms its input anyway, so only
the direction of the token matters downstream.
"""
from __future__ import annotations

import torch
import torch.nn as nn

from .attention import CrossAttention, LayerScale, Mlp


class ReadBlock(nn.Module):
    def __init__(self, dim: int, num_heads: int = 16, mlp_ratio: float = 4.0,
                 qk_norm: bool = True, gate_init: float = 0.0,
                 relative_residual: bool = True, query_gate_init: float = 1.0):
        super().__init__()
        self.relative_residual = relative_residual
        self.branch_norm = nn.LayerNorm(dim, elementwise_affine=False)
        self.norm_q = nn.LayerNorm(dim)
        self.norm_kv = nn.LayerNorm(dim)
        self.cross = CrossAttention(dim, num_heads, qk_norm=qk_norm)
        self.norm_mlp = nn.LayerNorm(dim)
        self.mlp = Mlp(dim, mlp_ratio)
        self.gate_cross = LayerScale(dim, gate_init)
        self.gate_mlp = LayerScale(dim, gate_init)
        self.gate_cross_q = LayerScale(dim, query_gate_init)
        self.gate_mlp_q = LayerScale(dim, query_gate_init)

    def _scaled(self, branch: torch.Tensor, q: torch.Tensor) -> torch.Tensor:
        if not self.relative_residual:
            return branch
        # The scale is a unit conversion, not a signal, so it is detached.
        rms = q.detach().pow(2).mean(-1, keepdim=True).sqrt()
        return self.branch_norm(branch) * rms

    def forward(self, q: torch.Tensor, state: torch.Tensor,
                query_path: bool = False) -> torch.Tensor:
        g_cross = self.gate_cross_q if query_path else self.gate_cross
        g_mlp = self.gate_mlp_q if query_path else self.gate_mlp
        b = self.cross(self.norm_q(q), self.norm_kv(state))
        q = q + g_cross(self._scaled(b, q))
        b = self.mlp(self.norm_mlp(q))
        q = q + g_mlp(self._scaled(b, q))
        return q


class ReadTransformer(nn.Module):
    def __init__(self, dim: int, num_layers: int = 2, num_heads: int = 16,
                 mlp_ratio: float = 4.0, qk_norm: bool = True,
                 gate_init: float = 0.0, relative_residual: bool = True,
                 query_gate_init: float = 1.0):
        super().__init__()
        self.blocks = nn.ModuleList([
            ReadBlock(dim, num_heads, mlp_ratio, qk_norm, gate_init,
                      relative_residual, query_gate_init)
            for _ in range(num_layers)
        ])

    def forward(self, query: torch.Tensor, state: torch.Tensor,
                query_path: bool = False) -> torch.Tensor:
        """query: [B, Q, D] tokens or rays. state: [B, N, D]."""
        for blk in self.blocks:
            query = blk(query, state, query_path)
        return query

    def gate_norms(self) -> dict:
        out = {}
        for i, b in enumerate(self.blocks):
            out[f"read{i}/gate_cross"] = b.gate_cross.gamma.abs().mean().item()
            out[f"read{i}/gate_mlp"] = b.gate_mlp.gamma.abs().mean().item()
            out[f"read{i}/gate_cross_q"] = b.gate_cross_q.gamma.abs().mean().item()
            out[f"read{i}/gate_mlp_q"] = b.gate_mlp_q.gamma.abs().mean().item()
        return out
