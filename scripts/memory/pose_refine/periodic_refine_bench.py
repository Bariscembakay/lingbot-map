#!/usr/bin/env python3
"""Periodic refinement while lingbot streams, on benchmark scenes cached by eval_refiner_bench.py.

Lingbot's backbone never reads its camera head's output, so its per-frame tokens (the cache) do not
depend on what the head does; only the head is replayed here. Every K frames the token refiner
re-predicts the camera tokens of either all frames so far ("all") or the last W frames ("last"):
  feedback  (A): the refined camera tokens replace the originals in lingbot's camera-head memory (its
                 KV cache is rebuilt by replaying the head over them), so every later frame's pose is
                 predicted against the corrected past;
  output    (B): lingbot's head is left alone; only the output poses are replaced.
Two trajectories are scored per run: "online" (each frame's pose as predicted when it streamed in,
with feedback in A, plain lingbot in B) and "final" (each frame's most recent refined pose; frames
never refined keep their online pose). Lingbot's own head replayed on the original tokens is scored
too, as a check that the replay reproduces the cached poses.
Usage: periodic_refine_bench.py --bench kitti_kf1 --run RUN_DIR --scenes 00,01 [--every 512,128]
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_refiner_bench as eb  # noqa: E402
import train_bidir_refiner as br  # noqa: E402
import train_pose_refine as tp  # noqa: E402

A = br.A


def refined_cam(model, spec):
    """BidirRefiner.forward up to the camera tokens it hands the head: [t,6,D] -> [t,D]."""
    t = spec.shape[0]
    proj = model.inp if model.inp is not None else (lambda z: z)
    x = (proj(spec[None].float()) + model.type_emb[:6]).reshape(1, t * 6, -1)
    pos = torch.arange(t, device=spec.device).repeat_interleave(6)[None]
    for blk in model.blocks:
        x = blk(x, pos)
    cam = x.reshape(1, t, 6, -1)[:, :, 0]
    if model.out is not None:
        cam = model.out(cam)
    if model.anchors_fixed:
        cam = torch.cat([spec[None, :A, 0].to(cam.dtype), cam[:, A:]], 1)
    return cam[0].float()


class StreamHead:
    """lingbot's causal camera head fed one frame's camera token at a time, as GCTStream does."""

    def __init__(self, head, sliding_window):
        self.head, self.sw = head, sliding_window

    def reset(self):
        self.head.clean_kv_cache()

    def step(self, cam):          # cam [n, D]: the 8 scale frames together, then one frame per call
        n = cam.shape[0]
        tok = cam[None, :, None, :]                    # head reads aggregated_tokens_list[-1][:, :, 0]
        with torch.amp.autocast("cuda", enabled=False):
            out = self.head([tok.float()], causal_inference=True, num_frame_for_scale=A,
                            num_frame_per_block=n, sliding_window_size=self.sw)
        return out[-1][0]                              # [n, 9]

    def replay(self, cams):       # rebuild the memory over frames 0..len(cams)-1
        self.reset()
        self.step(cams[:A])
        for i in range(A, len(cams)):
            self.step(cams[i:i + 1])


@torch.no_grad()
def run_scene(f, model, sh, every, mode, window, feedback, dev):
    z = np.load(f)
    spec = torch.from_numpy(np.ascontiguousarray(z["specials"][:, tp.TAP23])).to(dev)
    T = len(spec)
    cams = spec[:, 0].float().clone()                 # what the head's memory holds for each past frame
    online = torch.zeros(T, 9, device=dev)
    final = torch.full((T, 9), float("nan"), device=dev)
    sh.reset()
    online[:A] = sh.step(cams[:A])
    for t in range(A + 1, T + 1):
        online[t - 1] = sh.step(cams[t - 1:t])[0]
        if every and t % every == 0 and t < T:
            a = 0 if mode == "all" else max(0, t - window)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                pose = model(spec[None, a:t], online[None, a:t])[-1][0].float()
                if feedback:
                    cr = refined_cam(model, spec[a:t])
            final[a:t] = pose
            if feedback:
                cams[a:t] = cr
                sh.replay(cams[:t])
    final = torch.where(torch.isnan(final), online, final)
    return z, online, final


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--bench", required=True, help="cache dir name, e.g. kitti_kf1")
    ap.add_argument("--run", required=True, help="token-refiner run dir (best.pt)")
    ap.add_argument("--scenes")
    ap.add_argument("--every", default="512,128")
    ap.add_argument("--window", type=int, default=1024)
    args = ap.parse_args()
    dev = torch.device("cuda")
    model, step = eb.load_refiner(args.run, dev)
    assert isinstance(model, br.BidirRefiner), "feedback needs the token refiner (camera tokens through the head)"
    # lingbot's own streaming camera head (its kv-cache settings), not the refiner's dense-mask copy
    from methods.lingbot_map import LingbotMapMethod
    import yaml
    cfg = yaml.safe_load((eb.REPO / "benchmark/configs/methods/lingbot_map_sdpa.yaml").read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["checkpoint"] = str(eb.REPO / "ckpt/lingbot-map.pt")
    lb = LingbotMapMethod(**kw).model
    sh = StreamHead(lb.camera_head, lb.sliding_window_size if getattr(lb, "enable_camera_sliding_window", False) else -1)
    files = sorted((eb.CACHE / args.bench).glob("*.npz"))
    if args.scenes:
        files = [f for f in files if f.stem in set(args.scenes.split(","))]
    arm_base = Path(args.run).name
    configs = [("replay", 0, "all", False)] + [(f"{fb}_{m}_k{k}", int(k), m, fb == "feedback")
                                               for k in args.every.split(",") for m in ("all", "last")
                                               for fb in ("feedback", "output")]
    for f in files:
        for name, k, mode, fb in configs:
            t0 = time.time()
            z, online, final = run_scene(f, model, sh, k, mode, args.window, fb, dev)
            hw = tuple(int(x) for x in z["hw"])
            gt_dir = str(z["gt_dir"])
            res = {}
            for kind, p in (("online", online), ("final", final)):
                if name == "replay" and kind == "final":
                    continue
                arm = f"periodic_{name}_{kind}__{arm_base}" if name != "replay" else "lingbot_headreplay"
                c2w = tp.pose_enc_to_c2w(p, hw).double().cpu().numpy()
                res[kind] = eb.score(c2w, gt_dir, eb.OUT / arm / args.bench / f.stem / "pred", False)["ate"]
            if name == "replay":
                diff = float((online - torch.from_numpy(z["pose_enc"]).to(dev)).abs().max())
                res["max_abs_diff_vs_cached_pose"] = diff
            print(f"[periodic] {args.bench}/{f.stem} {name}: " + ", ".join(f"{k} {v:.4g}" for k, v in res.items())
                  + f" ({time.time() - t0:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
