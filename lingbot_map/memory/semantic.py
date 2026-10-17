"""Semantic recall: query the memory state at 3D points for Chorus features.

Coordinates: queries are 3D points in the STREAM frame (first streamed camera,
canonical units), the frame the geometry head predicts in. Targets are looked
up per clip in the Chorus voxel table, which lives in ScanNet++ world metres:
a clip-frame point p is at C0 @ (s * p) there (scripts/memory/chorus_clip_align.py).
That conversion is only a label lookup; the model never sees metric coordinates.
"""
from __future__ import annotations

import json
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from scipy.spatial import cKDTree
from torch.utils.checkpoint import checkpoint

CHORUS_ROOT = Path("/data/chorus-scannetpp-v2-vox5cm")
CHORUS_CKPT = "/group/compact-3dmem/checkpoints/Chorus/lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc.pth"
UPCAST_DIM = 1232


class FourierPE(nn.Module):
    """sin/cos of xyz at octave frequencies. Canonical units are ~1.9 m, so the
    default bands span periods of 16 units (a whole floor) down to 1/32 unit
    (~6 cm, about one 5 cm Chorus voxel)."""

    def __init__(self, kmin: int = -4, kmax: int = 5):
        super().__init__()
        self.register_buffer("freqs", 2.0 * torch.pi * 2.0 ** torch.arange(kmin, kmax + 1).float())
        self.dim = 3 + 6 * len(self.freqs)

    def forward(self, xyz: torch.Tensor) -> torch.Tensor:
        a = xyz[..., None] * self.freqs                     # [..., 3, F]
        return torch.cat([xyz, a.sin().flatten(-2), a.cos().flatten(-2)], dim=-1)


class QueryBlock(nn.Module):
    """Cross-attention to the state + MLP. No attention between queries, so a
    point's answer depends only on the state and its own position."""

    def __init__(self, dim: int, heads: int):
        super().__init__()
        self.ln_q, self.ln_kv, self.ln2 = nn.LayerNorm(dim), nn.LayerNorm(dim), nn.LayerNorm(dim)
        self.attn = nn.MultiheadAttention(dim, heads, batch_first=True)
        self.mlp = nn.Sequential(nn.Linear(dim, 4 * dim), nn.GELU(), nn.Linear(4 * dim, dim))

    def forward(self, x: torch.Tensor, state: torch.Tensor) -> torch.Tensor:
        kv = self.ln_kv(state)
        x = x + self.attn(self.ln_q(x), kv, kv, need_weights=False)[0]
        return x + self.mlp(self.ln2(x))


class VoxelQuery(nn.Module):
    """Learnable token + FourierPE(xyz) -> QueryBlocks over the state -> feature.

    Predicts the per-channel STANDARDISED target (feat_mean / feat_std buffers,
    set once from the training scenes and saved with the model, so validation
    and later evaluation normalise identically)."""

    def __init__(self, dim: int = 768, out_dim: int = UPCAST_DIM, depth: int = 2, heads: int = 12):
        super().__init__()
        self.pe = FourierPE()
        self.inp = nn.Linear(self.pe.dim, dim)
        self.token = nn.Parameter(torch.randn(dim) * 0.02)
        self.blocks = nn.ModuleList([QueryBlock(dim, heads) for _ in range(depth)])
        self.norm = nn.LayerNorm(dim)
        self.out = nn.Linear(dim, out_dim)
        self.register_buffer("feat_mean", torch.zeros(out_dim))
        self.register_buffer("feat_std", torch.ones(out_dim))

    def forward(self, state: torch.Tensor, xyz: torch.Tensor) -> torch.Tensor:
        """state [B, S, dim], xyz [B, R, 3] (stream frame) -> [B, R, out_dim]."""
        x = self.token + self.inp(self.pe(xyz.float()))
        for blk in self.blocks:
            if self.training and torch.is_grad_enabled():
                x = checkpoint(blk, x, state, use_reentrant=False)
            else:
                x = blk(x, state)
        return self.out(self.norm(x))

    def standardise(self, feat: torch.Tensor) -> torch.Tensor:
        return (feat.float() - self.feat_mean) / self.feat_std

    def raw(self, pred: torch.Tensor) -> torch.Tensor:
        return pred.float() * self.feat_std + self.feat_mean


def semantic_loss(pred: torch.Tensor, target: torch.Tensor, valid: torch.Tensor):
    """Chorus's matching loss, (1 - cos) + SmoothL1, on standardised features.
    pred/target [B, R, D], valid [B, R] -> (loss, mean cosine)."""
    pred, target = pred.float(), target.float()
    cos = F.cosine_similarity(pred, target, dim=-1)
    l1 = F.smooth_l1_loss(pred, target, reduction="none").mean(-1)
    v = valid.float()
    n = v.sum().clamp_min(1.0)
    return ((1.0 - cos + l1) * v).sum() / n, float((cos * v).sum() / n)


class ChorusTargets:
    """One clip's Chorus voxel table and the clip -> ScanNet++ world transform."""

    # 10 cm (2 voxels): keeps ~92% of GT-lifted points on 13 clips (7.5 cm: 89%);
    # adjacent voxels agree to cosine ~0.99, so a neighbour is a sound label.
    def __init__(self, clip_name: str, align: dict, root: Path = CHORUS_ROOT, max_dist: float = 0.10):
        a = align[clip_name]
        d = root / a["scene"]
        self.xyz = np.load(d / "voxel_xyz.npy")
        self.feat = torch.from_numpy(np.load(d / "upcast.npy"))       # fp16, host RAM
        self.tree = cKDTree(self.xyz)
        self.C0 = np.asarray(a["C0"], np.float64)
        self.scale = float(a["scale"])
        self.max_dist = max_dist

    def lookup(self, p_clip: np.ndarray):
        """p_clip [M, 3] clip-frame canonical points -> (feat [M, D] fp32, valid [M])."""
        X = (p_clip * self.scale) @ self.C0[:3, :3].T + self.C0[:3, 3]
        dist, idx = self.tree.query(X, distance_upper_bound=self.max_dist)
        ok = np.isfinite(dist)
        feat = torch.zeros(len(X), self.feat.shape[1])
        if ok.any():
            feat[torch.from_numpy(ok)] = self.feat[idx[ok]].float()
        return feat, torch.from_numpy(ok)


def load_align(path: Path | str = CHORUS_ROOT / "clip_align.json") -> dict:
    return json.loads(Path(path).read_text())


def sample_pixels(rng, depth: torch.Tensor, n: int):
    """n random pixels with valid GT depth from one [H, W] map -> (v, u) int arrays.
    Rejection-samples (valid fractions are high) instead of scanning all pixels."""
    h, w = depth.shape
    vs, us = [], []
    for _ in range(8):
        v = rng.integers(0, h, 4 * n)
        u = rng.integers(0, w, 4 * n)
        ok = depth[v, u].numpy() > 0
        vs.append(v[ok]); us.append(u[ok])
        if sum(len(x) for x in vs) >= n:
            break
    v, u = np.concatenate(vs)[:n], np.concatenate(us)[:n]
    return v, u


def lift_pixels(depth, K, c2w, v, u):
    """Pixels (v, u) of one frame -> clip-frame points [M, 3] (numpy, canonical)."""
    z = depth[v, u].numpy().astype(np.float64)
    K = K.numpy().astype(np.float64); c2w = c2w.numpy().astype(np.float64)
    pc = np.stack([(u - K[0, 2]) / K[0, 0] * z, (v - K[1, 2]) / K[1, 1] * z, z], 1)
    return pc @ c2w[:3, :3].T + c2w[:3, 3]


def to_stream(p_clip: np.ndarray, c2w_first) -> np.ndarray:
    """Clip-frame points -> stream frame (first streamed camera), same as the
    geometry targets' world frame."""
    T = c2w_first.numpy().astype(np.float64)
    return (p_clip - T[:3, 3]) @ T[:3, :3]


def load_chorus_projectors(ckpt: str = CHORUS_CKPT) -> dict:
    """{'lang': 1232->1152, 'dino': 1232->1024} Chorus projectors without the final
    L2 normalisation, rebuilt from the checkpoint (Chorus's own loader needs CUDA)."""
    sd = torch.load(ckpt, map_location="cpu", weights_only=False)["state_dict"]
    out = {}
    for name, d in (("lang", 1152), ("dino", 1024)):
        p = f"module.projectors.{name}."
        m = nn.Sequential(nn.LayerNorm(UPCAST_DIM), nn.Linear(UPCAST_DIM, 4928), nn.GELU(), nn.Linear(4928, d))
        m.load_state_dict({"0.weight": sd[p + "input_norm.weight"], "0.bias": sd[p + "input_norm.bias"],
                           "1.weight": sd[p + "mlp_head.0.weight"], "1.bias": sd[p + "mlp_head.0.bias"],
                           "3.weight": sd[p + "mlp_head.2.weight"], "3.bias": sd[p + "mlp_head.2.bias"]})
        out[name] = m.eval().requires_grad_(False)
    return out
