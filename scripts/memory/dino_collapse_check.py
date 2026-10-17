#!/usr/bin/env python3
"""Is lingbot's fine-tuned DINO collapsed, or just anisotropic?

Same frames through official DINOv2 and lingbot's trained copy (patch tokens):
  - global mean-patch cosine, raw and after removing the mean over all frames
  - patch-token effective rank (participation ratio of the covariance spectrum)
    and the variance share of the top principal component
  - patch correspondence: in view pairs that overlap > 0.5 (GT depth + poses),
    how often a patch's nearest neighbour (cosine) in the other view is its GT
    correspondence (within 1 patch)
Usage: python dino_collapse_check.py <clip_dir> --other <clip_dir>
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[2]))
from dino_global_scene import MEAN, STD, decode, load_dino  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

P = 14


@torch.no_grad()
def patch_tokens(model, rgb, device, bs=16):
    m, s = torch.tensor(MEAN, device=device)[:, None, None], torch.tensor(STD, device=device)[:, None, None]
    out = []
    for i in range(0, len(rgb), bs):
        x = torch.from_numpy(rgb[i:i + bs]).to(device).permute(0, 3, 1, 2).float() / 255
        out.append(model.forward_features((x - m) / s)["x_norm_patchtokens"].float())
    return torch.cat(out)                                                   # [n, hp*wp, C]


def global_stats(a, b):
    """a, b: mean-patch vectors of two scenes [n, C]. Raw and centred cosine."""
    res = {}
    for name, mu in (("raw", 0), ("centred", torch.cat([a, b]).mean(0))):
        x, y = F.normalize(a - mu, dim=-1), F.normalize(b - mu, dim=-1)
        w = x @ x.T
        res[name] = dict(within=float(w[torch.triu_indices(len(x), len(x), 1).unbind()].mean()),
                         other=float((x @ y.T).mean()))
    return res


def spectrum(tok, n=20000, seed=0):
    x = tok.reshape(-1, tok.shape[-1])
    x = x[torch.randperm(len(x), generator=torch.Generator().manual_seed(seed))[:n]].double()
    lam = torch.linalg.eigvalsh(torch.cov(x.T)).clamp(min=0).flip(0)
    rnd = F.normalize(x[:2000].float(), dim=-1)
    return dict(participation_ratio=float(lam.sum() ** 2 / (lam ** 2).sum()),
                top1_var_share=float(lam[0] / lam.sum()),
                raw_patch_pair_cos=float((rnd[:1000] * rnd[1000:]).sum(-1).mean()))


def correspondences(c, frames, hp, wp, device, min_overlap=0.5, max_pairs=60, seed=0):
    """GT patch correspondences for overlapping pairs: list of (i, j, src_idx, dst_ij)."""
    d = torch.from_numpy(np.asarray(c.gt_depth[frames], np.float32)).to(device)
    K = torch.from_numpy(np.asarray(c.gt_intrinsics[frames], np.float32)).to(device)
    T = torch.from_numpy(np.asarray(c.gt_c2w[frames], np.float32)).to(device)
    H, W = d.shape[1:]
    pv, pu = torch.meshgrid(torch.arange(hp, device=device), torch.arange(wp, device=device), indexing="ij")
    u, v = (pu.reshape(-1) * P + P // 2).float(), (pv.reshape(-1) * P + P // 2).float()
    w2c = torch.linalg.inv(T)
    pairs = []
    for i in range(len(frames)):
        z = d[i, v.long(), u.long()]
        pc = torch.stack([(u - K[i, 0, 2]) / K[i, 0, 0] * z, (v - K[i, 1, 2]) / K[i, 1, 1] * z, z], 1)
        pw = pc @ T[i, :3, :3].T + T[i, :3, 3]
        for j in range(len(frames)):
            if j == i:
                continue
            q = pw @ w2c[j, :3, :3].T + w2c[j, :3, 3]
            zj = q[:, 2].clamp(min=1e-6)
            uj, vj = K[j, 0, 0] * q[:, 0] / zj + K[j, 0, 2], K[j, 1, 1] * q[:, 1] / zj + K[j, 1, 2]
            inb = (z > 0) & (q[:, 2] > 0) & (uj >= 0) & (uj < W) & (vj >= 0) & (vj < H)
            ui, vi = uj.long().clamp(0, W - 1), vj.long().clamp(0, H - 1)
            ok = inb & ((q[:, 2] - d[j, vi, ui]).abs() < 0.05 * d[j, vi, ui])
            if ok.float().mean() > min_overlap:
                pairs.append((i, j, ok.nonzero()[:, 0], torch.stack([vj[ok] / P, uj[ok] / P], 1)))
    rng = np.random.default_rng(seed)
    return [pairs[k] for k in rng.choice(len(pairs), min(max_pairs, len(pairs)), replace=False)]


def match_acc(tok, pairs, hp, wp):
    acc = []
    for i, j, src, dst in pairs:
        a, b = F.normalize(tok[i, src], dim=-1), F.normalize(tok[j], dim=-1)
        nn = (a @ b.T).argmax(1)
        pr = torch.stack([nn // wp, nn % wp], 1).float()
        acc.append(float(((pr - dst.floor()).abs().max(1).values <= 1).float().mean()))
    return float(np.mean(acc))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("--other", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    c, co = ClipCache(args.clip), ClipCache(args.other)
    fa, fb = list(range(0, len(c), 5)), list(range(0, len(co), 5))
    ra, rb = decode(args.clip, c, fa), decode(args.other, co, fb)
    hp, wp = c.meta.height // P, c.meta.width // P
    pairs = correspondences(c, fa, hp, wp, dev)
    res = {"pairs_used": len(pairs), "frames_per_scene": len(fa)}
    for which in ("official", "lingbot"):
        model = load_dino(dev, which)
        ta, tb = patch_tokens(model, ra, dev), patch_tokens(model, rb, dev)
        res[which] = dict(globalcos=global_stats(ta.mean(1), tb.mean(1)), spectrum=spectrum(ta.cpu()),
                          patch_match_acc=match_acc(ta, pairs, hp, wp))
        del model, ta, tb
        torch.cuda.empty_cache()
    args.out.write_text(json.dumps(res, indent=1))
    print(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
