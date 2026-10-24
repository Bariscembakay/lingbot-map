#!/usr/bin/env python3
"""inference_chunked with keyframe_interval > 1 against frame-by-frame inference_streaming on real clips:
max |diff| of the tap-23 special tokens and pose_enc, the ATE of both to GT, and frames/s of each.
Usage: check_chunked_keyframes.py --dataset kitti360 --n-clips 2 --frames 700 [--chunk 64] [--attn flex]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO))
import train_pose_refine as tp  # noqa: E402
from clip_sources import source  # noqa: E402
from scripts.memory.build_cache import build_model  # noqa: E402

PLANS = Path("/group/compact-3dmem/campaigns/spatial_memory/_data_prep/plans_full")


@torch.no_grad()
def run(model, x, kf, mode, chunk, attn):
    agg = model.aggregator
    caps = []
    h = agg.register_forward_hook(lambda _m, _i, o: caps.append(torch.stack(o[0], 2)[:, :, :, :agg.patch_start_idx].float()))
    torch.cuda.synchronize()
    t0 = time.time()
    try:
        with torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if mode == "stream":
                pose = model.inference_streaming(x, num_scale_frames=8, keyframe_interval=kf, output_device=None)["pose_enc"]
            else:
                pose = model.inference_chunked(x, chunk=chunk, num_scale_frames=8, attn=attn, keyframe_interval=kf,
                                               max_frames=max(4096, kf * 512))["pose_enc"]
    finally:
        h.remove()
    torch.cuda.synchronize()
    S = x.shape[1]
    return torch.cat(caps, 1)[:, :S, tp.TAP23], pose.float()[:, :S], S / (time.time() - t0)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", default="kitti360")
    ap.add_argument("--n-clips", type=int, default=2)
    ap.add_argument("--frames", type=int, default=700)
    ap.add_argument("--kf", type=int, default=0, help="0 = lingbot's rule ceil(T/320)")
    ap.add_argument("--chunk", type=int, default=64)
    ap.add_argument("--attn", default="flex")
    a = ap.parse_args()
    dev = torch.device("cuda")
    margs = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=8,
                            use_sdpa=True, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(margs, dev)
    model.depth_head = None
    src = source(a.dataset)
    clips = [c for c in json.loads((PLANS / f"{a.dataset}.json").read_text())["clips"] if len(c["frames"]) >= a.frames][:a.n_clips]
    for c in clips:
        ids = c["frames"][:a.frames]
        im, _ = src.frames(c["seq"], ids)
        x = torch.from_numpy(np.ascontiguousarray(im.transpose(0, 3, 1, 2))).float().div_(255)[None]
        kf = a.kf or (1 if len(ids) <= 320 else -(-len(ids) // 320))
        G = src.poses(c["seq"])[ids]
        G = np.linalg.inv(G[0])[None] @ G
        res = {}
        for mode in ("stream", "chunk"):
            spec, pose, fps = run(model, x, kf, mode, a.chunk, a.attn)
            P = tp.pose_enc_to_c2w(pose[0].cpu(), im.shape[1:3]).double()
            ate = float(tp.traj_metrics(P, torch.from_numpy(G), 1.0)["ate"])
            res[mode] = (spec, pose, fps, ate)
        (s0, p0, f0, a0), (s1, p1, f1, a1) = res["stream"], res["chunk"]
        ds, dp = (s1 - s0).abs(), (p1 - p0).abs()
        print(f"[check] {c['seq']} {len(ids)} frames {tuple(im.shape[1:3])} kf {kf}: spec23 max|d| {ds.max():.4g} "
              f"(|x| max {s0.abs().max():.3g}, mean|d| {ds.mean():.3g}); pose max|d| {dp.max():.4g}, mean {dp.mean():.3g}; "
              f"ATE stream {a0:.4g} chunk {a1:.4g}; fps stream {f0:.2f} chunk {f1:.2f} ({f1 / f0:.1f}x)", flush=True)
        # bf16 rounding vs a real context bug: a wrong mask shows up first at keyframes vs non-keyframes
        per = ds.amax(dim=(0, 2, 3))
        nk = torch.tensor([(i - 8) % kf != 0 for i in range(len(ids))])
        print(f"[check]   per-frame spec max|d|: keyframes {per[~nk].max():.4g}, non-keyframes {per[nk].max() if nk.any() else 0:.4g}, "
              f"first 72 {per[:72].max():.4g}, last 100 {per[-100:].max():.4g}", flush=True)


if __name__ == "__main__":
    main()
