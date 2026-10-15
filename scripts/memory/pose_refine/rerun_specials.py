#!/usr/bin/env python3
"""Specials-only lingbot rerun for tap-23-only (v5) caches.

The v5 caches keep tap 23 only, so the refiner's 4-tap specials have to be
recomputed. Re-streams lingbot over the cache's exact frame_ids and keeps just
the 6 special tokens of taps 4/11/17/23 per frame (~30 MB/clip instead of the
~5 GB full 4-tap cache). GT comes from the existing cache; its tap-23 specials
and pose_enc double as a reproduction check, stored in the npz as `check_*`.
Output format matches build_inputs.py.
Usage: rerun_specials.py --clips list.txt --out DIR [--use-sdpa]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "scripts/memory"))
from build_cache import TAP_LAYERS, build_model, decode_frames  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--scannetpp-root", default="/data/ScanNetpp")
    ap.add_argument("--ckpt", default=str(REPO / "ckpt/lingbot-map.pt"))
    ap.add_argument("--use-sdpa", action="store_true")
    args = ap.parse_args()
    dev = torch.device("cuda")
    out = args.out / "specials"
    out.mkdir(parents=True, exist_ok=True)
    model = None
    for line in args.clips.read_text().split():
        p = Path(line)
        f = out / f"{p.name}.npz"
        if f.exists():
            try:
                with np.load(f) as z:
                    z["check_pose_maxabs"]
                continue
            except Exception:   # truncated by a walltime kill: redo it
                print(f"[redo] {p.name}: unreadable npz", flush=True)
        c = ClipCache(p)
        m = c.meta
        if model is None:
            ns = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=1024,
                                 kv_cache_sliding_window=m.kv_cache_sliding_window,
                                 num_scale_frames=m.scale_frames, use_sdpa=args.use_sdpa,
                                 camera_num_iterations=4, ckpt=args.ckpt)
            model = build_model(ns, dev)
        psi = m.patch_start_idx
        video = Path(args.scannetpp_root) / "data" / m.scene / "iphone" / "rgb.mkv"
        rgb = decode_frames(video, list(m.frame_ids), m.width, m.height)
        spec = np.zeros((m.num_frames, 4, psi, m.embed_dim), np.float16)
        n = {"i": 0}

        def capture(_mod, _inp, output):
            tok = torch.stack(output[0], 2)[0][:, :, :psi]      # [S, 4, psi, E]
            s = tok.shape[0]
            spec[n["i"]:n["i"] + s] = tok.float().cpu().numpy().astype(np.float16)
            n["i"] += s

        h = model.aggregator.register_forward_hook(capture)
        images = torch.from_numpy(np.ascontiguousarray(rgb.transpose(0, 3, 1, 2))).float().div_(255.0)
        try:
            with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.inference_streaming(images, num_scale_frames=m.scale_frames,
                                                 keyframe_interval=m.keyframe_interval,
                                                 output_device=torch.device("cpu"))
        finally:
            h.remove()
        assert n["i"] == m.num_frames, (n["i"], m.num_frames)
        pose = pred["pose_enc"][0].float().numpy()
        old = np.asarray(c.taps[:, m.tap_layers.index(23), :psi], np.float32)
        new = spec[:, TAP_LAYERS.index(23)].astype(np.float32)
        cos = float(np.mean(np.sum(old * new, -1) / (np.linalg.norm(old, axis=-1)
                                                      * np.linalg.norm(new, axis=-1) + 1e-6)))
        dpose = float(np.abs(pose - c.pose_enc).max())
        tmp = f.with_suffix(".tmp.npz")
        np.savez(tmp, specials=spec, pose_enc=pose.astype(np.float32),
                 gt_c2w=c.gt_c2w.astype(np.float32), K=c.gt_intrinsics.astype(np.float32),
                 hw=np.int32([m.height, m.width]), gt_scale=np.float64(m.gt_scale), source=str(p),
                 check_tap23_special_cos=np.float32(cos), check_pose_maxabs=np.float32(dpose))
        tmp.rename(f)
        print(f"[rerun] {p.name}: tap23-specials cos {cos:.5f}, pose_enc max|d| {dpose:.4f}", flush=True)


if __name__ == "__main__":
    main()
