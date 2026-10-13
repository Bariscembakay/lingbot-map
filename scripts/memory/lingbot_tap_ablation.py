#!/usr/bin/env python3
"""Direct effect of each aggregator tap on lingbot's depth and camera heads.

The heads read disjoint inputs: the DPT depth head the PATCH tokens of taps
4/11/17/23, the causal camera head only the CAMERA token (index 0) of tap 23.
Each ablation mean-replaces one input (per channel, over this clip's frames and
patches) and reruns the heads on the cached taps -- the aggregator never runs,
so these are effects at the head input, not through later aggregator layers.
A tap is cat([frame_attn, global_attn]); each half is ablated on its own too.

Depth: AbsRel and delta<1.25 vs GT after per-frame median scaling.
Pose: ATE RMSE after a Sim(3) fit to GT camera centres, in metres (x gt_scale),
and mean relative rotation error between consecutive frames (deg).
Baseline must reproduce the cache's own depth.npy / pose_enc.npy.
Usage: python lingbot_tap_ablation.py --out results.json <clip_dir> ...
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402
from lingbot_map.memory.losses import pose_enc_to_c2w  # noqa: E402

TAPS = [4, 11, 17, 23]


def build_model(device):
    from lingbot_map.models.gct_stream import GCTStream
    # Same construction as build_cache.py, which produced the cached taps.
    model = GCTStream(img_size=518, patch_size=14, enable_3d_rope=True, max_frame_num=1024,
                      kv_cache_sliding_window=64, kv_cache_scale_frames=8,
                      kv_cache_cross_frame_special=True, kv_cache_include_scale_frames=True,
                      use_sdpa=False, camera_num_iterations=4)
    ck = torch.load(REPO / "ckpt/lingbot-map.pt", map_location="cpu", weights_only=False)
    model.load_state_dict(ck.get("model", ck), strict=False)
    assert not model.pred_normalization
    return model.to(device).eval()


class CachedAggregator(torch.nn.Module):
    """Stands in for the aggregator: hands out the (ablated) cached tokens of
    the next frames in stream order."""

    def __init__(self, toks, patch_start_idx):
        super().__init__()
        self.toks, self.psi, self.ptr = toks, patch_start_idx, 0

    def forward(self, images, **_):
        s = images.shape[1]
        out = [t[None, self.ptr:self.ptr + s] for t in self.toks]
        self.ptr += s
        return out, self.psi


def ablate(toks, means, spec, psi):
    """spec: ('drop'|'keep', tap or None, 'all'|'frame'|'global', 'patch'|'cam')."""
    mode, tap, half, part = spec
    E = toks[0].shape[-1]
    ch = {"all": slice(0, E), "frame": slice(0, E // 2), "global": slice(E // 2, E)}[half]
    tk = slice(psi, None) if part == "patch" else slice(0, 1)
    out = [t.clone() for t in toks]
    targets = [TAPS.index(tap)] if mode == "drop" else [i for i in range(4) if TAPS[i] != tap]
    for i in targets:
        m = means[i][0] if part == "patch" else means[i][1]
        out[i][:, tk, ch] = m[ch]
    return out


def depth_metrics(pred, gt):
    a, d = [], []
    for p, g in zip(pred, gt):
        m = g > 0
        s = np.median(g[m]) / np.median(p[m])
        r = s * p[m] / g[m]
        a.append(np.mean(np.abs(r - 1))); d.append(np.mean(np.maximum(r, 1 / r) < 1.25))
    return float(np.mean(a)), float(np.mean(d))


def sim3_ate(P, G):
    mp, mg = P.mean(0), G.mean(0)
    X, Y = P - mp, G - mg
    U, S, Vt = np.linalg.svd(Y.T @ X / len(P))
    D = np.eye(3); D[2, 2] = np.sign(np.linalg.det(U @ Vt))
    R = U @ D @ Vt
    s = np.trace(np.diag(S) @ D) / X.var(0).sum()
    return float(np.sqrt(np.mean(np.sum((s * X @ R.T - Y) ** 2, 1))))


def rpe_rot(Rp, Rg):
    e = []
    for i in range(len(Rp) - 1):
        dp, dg = Rp[i].T @ Rp[i + 1], Rg[i].T @ Rg[i + 1]
        c = (np.trace(dg.T @ dp) - 1) / 2
        e.append(np.degrees(np.arccos(np.clip(c, -1, 1))))
    return float(np.mean(e))


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--frames", type=int, default=96)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model = build_model(dev)
    real_agg = model.aggregator
    heads = {k: getattr(model, k) for k in ("depth_head", "point_head", "local_point_head")}

    specs = {"base": None}
    for t in TAPS:
        for h in ("all", "frame", "global"):
            specs[f"drop{t}_{h}"] = ("drop", t, h, "patch")
        specs[f"only{t}"] = ("keep", t, "all", "patch")
    for h in ("all", "frame", "global"):
        specs[f"cam23_{h}"] = ("drop", 23, h, "cam")
    res = json.loads(args.out.read_text()) if args.out.exists() else {}
    for cp in args.clips:
        if cp.name in res:
            continue
        c = ClipCache(cp)
        assert list(c.meta.tap_layers) == TAPS, (cp, c.meta.tap_layers)
        n, psi = args.frames, c.meta.patch_start_idx
        H, W = c.meta.height, c.meta.width
        raw = torch.from_numpy(np.ascontiguousarray(c.taps[:n])).to(dev)      # [n, 4, P, E] fp16
        toks = [raw[:, i].float() for i in range(4)]
        del raw
        means = [(t[:, psi:].mean((0, 1)), t[:, 0].mean(0)) for t in toks]
        gt_d = np.asarray(c.gt_depth[:n], np.float32)
        gt_c2w = np.asarray(c.gt_c2w[:n], np.float64)
        scale = float(c.meta.gt_scale)
        imgs = torch.zeros(1, n, 3, H, W)
        out = {}
        for name, spec in specs.items():
            tk = toks if spec is None else ablate(toks, means, spec, psi)
            r = {}
            if spec is None or spec[3] == "patch":
                with torch.no_grad():
                    d, _ = heads["depth_head"]([t[None] for t in tk], images=imgs.to(dev), patch_start_idx=psi)
                d = d[0, ..., 0].float().cpu().numpy()
                r["absrel"], r["delta1"] = depth_metrics(d, gt_d)
                if spec is None:
                    ref = np.asarray(c.depth[:n], np.float32)
                    r["repro_depth_relerr"] = float(np.mean(np.abs(d - ref) / ref))
            if spec is None or spec[3] == "cam" or spec == ("drop", 23, "all", "patch"):
                model.aggregator = CachedAggregator(tk, psi)
                for k in heads:
                    setattr(model, k, None)
                with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                    pred = model.inference_streaming(imgs, num_scale_frames=8, keyframe_interval=1,
                                                     output_device=torch.device("cpu"))
                model.aggregator = real_agg
                for k, v in heads.items():
                    setattr(model, k, v)
                model.clean_kv_cache()
                pe = pred["pose_enc"][0].float()
                c2w = pose_enc_to_c2w(pe, (H, W)).numpy().astype(np.float64)
                r["ate_m"] = sim3_ate(c2w[:, :3, 3], gt_c2w[:, :3, 3]) * scale
                r["rpe_rot_deg"] = rpe_rot(c2w[:, :3, :3], gt_c2w[:, :3, :3])
                if spec is None:
                    r["repro_pose_maxdiff"] = float((pe - torch.from_numpy(np.asarray(c.pose_enc[:n]))).abs().max())
            out[name] = r
        res[cp.name] = out
        args.out.write_text(json.dumps(res, indent=1))
        print(cp.name, json.dumps(out["base"]), flush=True)


if __name__ == "__main__":
    main()
