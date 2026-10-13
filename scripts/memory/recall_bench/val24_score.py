#!/usr/bin/env python3
"""val24 recall comparison, CPU phase: score every dumped method identically.

GT: each val clip's self-frame pointmaps at the queried cameras, canonical
units, on the cache grid (train_state Clip.probe_inputs, as evaluate() uses).
Predictions on another grid (CUT3R/TTT3R: 512x384 from the full 4:3 frame,
while the cache grid is a 518x378 crop) are resampled PER RAY: each GT pixel's
ray K_gt^-1 [u,v,1] is projected with the predictor's own intrinsics and the
pointmap sampled bilinearly there -- a plain image resize would compare
different rays. Metric predictions are divided by the clip's gt_scale.

Per clip, pooled over all valid pixels of the queried views (= valm_self):
  raw       L21 as predicted (our models are trained in these units)
  aligned   L21 after one median depth-ratio scale per clip and method
  recent / old   raw L21 on lags <= 4 / lags >= T//2
Usage: python val24_score.py <dump_root> [label ...]
"""
from __future__ import annotations

import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path.home() / "lingbot-map"
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/memory"))
from train_state import Clip  # noqa: E402


def resample(pred, K_pred, K_gt, h, w):
    """pred [Hp, Wp, 3] on K_pred's grid -> [h, w, 3] on K_gt's grid, plus inside-mask."""
    Hp, Wp = pred.shape[:2]
    v, u = np.meshgrid(np.arange(h), np.arange(w), indexing="ij")
    r = np.stack([u, v, np.ones_like(u)], -1).reshape(-1, 3).astype(np.float64) @ np.linalg.inv(K_gt).T
    q = r @ K_pred.T
    x, y = q[:, 0] / q[:, 2], q[:, 1] / q[:, 2]
    inside = (x >= 0) & (x <= Wp - 1) & (y >= 0) & (y <= Hp - 1)
    x0 = np.clip(np.floor(x).astype(int), 0, Wp - 2); y0 = np.clip(np.floor(y).astype(int), 0, Hp - 2)
    wx = np.clip(x - x0, 0, 1)[:, None]; wy = np.clip(y - y0, 0, 1)[:, None]
    p = (pred[y0, x0] * (1 - wx) * (1 - wy) + pred[y0, x0 + 1] * wx * (1 - wy)
         + pred[y0 + 1, x0] * (1 - wx) * wy + pred[y0 + 1, x0 + 1] * wx * wy)
    return p.reshape(h, w, 3), inside.reshape(h, w)


def score_clip(d, clip_dir):
    c = Clip(Path(clip_dir), 2, "cpu", 160, "23", store="cpu")
    qs = [int(q) for q in d["qs"]]
    _, xs, _, valid = c.probe_inputs(qs, "cut3r", anchor=0)
    xs, valid = xs.numpy(), valid.numpy()
    pred = d["pred_self"].astype(np.float64)
    if str(d["unit"]) == "metres":
        pred = pred / float(d["gt_scale"])
    if pred.shape[1:3] != xs.shape[1:3]:
        out = np.zeros_like(xs, dtype=np.float64)
        for i in range(len(qs)):
            out[i], inside = resample(pred[i], d["K_pred"][i].astype(np.float64),
                                      c.K[qs[i]].numpy().astype(np.float64), *xs.shape[1:3])
            valid[i] &= inside
        pred = out
    T = min(96, len(c)) - 1
    lag = np.array([T - q for q in qs])
    err = np.linalg.norm(pred - xs, axis=-1)
    z, zg = pred[..., 2], xs[..., 2]
    m = valid & (z > 1e-6)
    s = float(np.median(zg[m]) / np.median(z[m]))
    err_al = np.linalg.norm(pred * s - xs, axis=-1)
    rec, old = valid & (lag <= 4)[:, None, None], valid & (lag >= T // 2)[:, None, None]
    return dict(raw=float(err[valid].mean()), aligned=float(err_al[valid].mean()), scale=s,
                recent=float(err[rec].mean()), old=float(err[old].mean()))


def main():
    root = Path(sys.argv[1])
    labels = sys.argv[2:] or sorted(p.name for p in root.iterdir() if p.is_dir())
    clips = {Path(p).name: p for p in (REPO / ".agents/scratch/memory_eval/val24.txt").read_text().split()}
    res = {}
    for lab in labels:
        res[lab] = {}
        for f in sorted((root / lab).glob("*.npz")):
            res[lab][f.stem] = score_clip(np.load(f), clips[f.stem])
    (root / "val24_scores.json").write_text(json.dumps(res, indent=1))
    print(f"{'method':34s} {'n':>3s} {'raw':>7s} {'aligned':>8s} {'recent':>7s} {'old':>7s} {'scale med':>9s}")
    for lab, r in res.items():
        if not r:
            continue
        a = {k: np.mean([v[k] for v in r.values()]) for k in ("raw", "aligned", "recent", "old")}
        print(f"{lab:34s} {len(r):3d} {a['raw']:7.4f} {a['aligned']:8.4f} {a['recent']:7.4f} {a['old']:7.4f} "
              f"{np.median([v['scale'] for v in r.values()]):9.3f}")


if __name__ == "__main__":
    main()
