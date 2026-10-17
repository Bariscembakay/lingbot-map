#!/usr/bin/env python3
"""lingbot with a lagged re-query: each frame's pose is re-predicted just before it leaves the
64-frame window.

lingbot streams as usual (every frame appended, sliding window 64 + 8 scale frames, SDPA, as the
NRGBD eval cache was built). Right after frame i is appended, frame k = i - 63 (the oldest
non-anchor frame, evicted when frame i+1 arrives) is run again through the whole pipeline (DINO,
aggregator, camera head) with cache appends disabled, so it attends to the anchors, the evicted
frames' special tokens and frames k..i, and its new pose replaces the causal one. At a prefix
length t, frames t-63..t-1 never reach that point, so they are re-queried at time t-1.
Temporal RoPE position of a re-query:
  now   the next free position (total_frames_processed, as for a new frame)
  orig  the frame's own index k
Scale frames 0..7 keep their poses. The plain stream of the same run is the lingbot baseline.
Usage: eval_lag_requery.py --out results.json [--scenes a,b] [--lengths 64,128,256,320,500]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from scripts.memory.build_cache import benchmark_resize, build_model  # noqa: E402

INPUTS = Path("/data/lingbot-poserefine-v1/nrgbd/specials")
CACHE = Path("/group/compact-3dmem/campaigns/spatial_memory/eval_nrgbd_cache")
A, LAG = 8, 63


def load_rgb(scene, frame_ids, root):
    W, H = benchmark_resize(640, 480, 518, 14, 255000)
    return np.stack([cv2.resize(np.array(Image.open(root / scene / "images" / f"img{f}.png").convert("RGB")),
                                (W, H), interpolation=cv2.INTER_LANCZOS4) for f in frame_ids])


@torch.no_grad()
def run(model, images, lengths):
    """images [S,3,H,W] -> plain pose_enc [S,9]; {mode: {t: pose_enc [t,9]}} with re-queries."""
    dev = next(model.parameters()).device
    S = images.shape[0]
    agg = model.aggregator

    def fwd(x, nfb):
        return model.forward(x[None].to(dev), num_frame_for_scale=A, num_frame_per_block=nfb,
                             causal_inference=True)["pose_enc"][0].float().cpu()

    def requery(k, mode):
        model._set_skip_append(True)
        saved = agg.total_frames_processed
        if mode == "orig":
            agg.total_frames_processed = k
        p = fwd(images[k:k + 1], 1)[0]
        agg.total_frames_processed = saved
        model._set_skip_append(False)
        return p

    modes = ("now", "orig")
    model.clean_kv_cache()
    plain = [fwd(images[:A], A)]
    lag = {m: {} for m in modes}            # frame -> pose re-queried at k + LAG
    tail = {m: {} for m in modes}           # t -> {frame: pose re-queried at t - 1}
    for i in range(A, S):
        plain.append(fwd(images[i:i + 1], 1))
        for m in modes:
            k = i - LAG
            if k >= A:
                lag[m][k] = requery(k, m)
            if i + 1 in lengths:
                tail[m][i + 1] = {f: requery(f, m) for f in range(max(A, i + 1 - LAG), i + 1)}
    plain = torch.cat(plain)
    out = {}
    for m in modes:
        out[m] = {}
        for t in lengths:
            p = plain[:t].clone()
            for f in range(A, t):
                p[f] = lag[m][f] if f + LAG <= t - 1 else tail[m][t][f]
            out[m][t] = p
    return plain, out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scenes", default="")
    ap.add_argument("--lengths", default="64,128,256,320,500")
    ap.add_argument("--root", type=Path, default=Path("/data/NRGBD"))
    ap.add_argument("--ckpt", default=str(REPO / "ckpt/lingbot-map.pt"))
    args = ap.parse_args()
    lengths = [int(x) for x in args.lengths.split(",")]
    margs = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=1024, kv_cache_sliding_window=64,
                            num_scale_frames=A, use_sdpa=True, camera_num_iterations=4, ckpt=args.ckpt)
    model = build_model(margs, torch.device("cuda"))
    files = sorted(INPUTS.glob("*.npz"))
    if args.scenes:
        files = [f for f in files if f.stem.rsplit("_500f", 1)[0] in args.scenes.split(",")]
    per_clip = {}
    for f in files:
        z = np.load(f)
        meta = json.loads((CACHE / f.stem / "meta.json").read_text())
        hw, scale = tuple(int(x) for x in z["hw"]), float(z["gt_scale"])
        gt = torch.from_numpy(z["gt_c2w"])
        images = torch.from_numpy(np.ascontiguousarray(
            load_rgb(meta["scene"], meta["frame_ids"], args.root).transpose(0, 3, 1, 2))).float().div_(255)
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            plain, req = run(model, images, lengths)
        cached = torch.from_numpy(z["pose_enc"]).float()
        print(f"[{f.stem}] plain vs cached lingbot pose_enc max abs diff {(plain - cached).abs().max():.2e}", flush=True)
        for t in lengths:
            r = {"lingbot_cached": tp.traj_metrics(tp.pose_enc_to_c2w(cached[:t], hw), gt[:t], scale),
                 "lingbot": tp.traj_metrics(tp.pose_enc_to_c2w(plain[:t], hw), gt[:t], scale)}
            for m, d in req.items():
                r[f"requery_{m}"] = tp.traj_metrics(tp.pose_enc_to_c2w(d[t], hw), gt[:t], scale)
            per_clip[f"{f.stem}@{t}"] = r
            print(f"  t={t}: " + " ".join(f"{k} {v['ate']*100:.2f}" for k, v in r.items()), flush=True)
    summary = {}
    for t in lengths:
        rows = [r for k, r in per_clip.items() if k.endswith(f"@{t}")]
        summary[t] = {k: {m: float(np.mean([r[k][m] for r in rows])) for m in tp.METRICS} for k in rows[0]}
        print(f"[mean t={t}, {len(rows)} scenes] ATE cm " +
              " ".join(f"{k} {v['ate']*100:.2f}" for k, v in summary[t].items()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "per_clip": per_clip}, indent=1))


if __name__ == "__main__":
    main()
