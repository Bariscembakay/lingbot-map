#!/usr/bin/env python3
"""Build a v4 tap cache for one 7-Scenes sequence clip.

Mirrors scripts/memory/build_cache.py (same build_model, same tap-capture hook,
same fp16 absmax guard, same benchmark_resize) with 7-Scenes I/O and GT:

* RGB from frame-NNNNNN.color.png (640x480), lanczos-resized to the benchmark
  grid (518x378 -> patch grid 37x27, the training grid).
* GT poses from frame-NNNNNN.pose.txt: metric C2W, re-expressed relative to
  frame 0. Kept METRIC -- gt_scale = 1.0 (unlike ScanNet++'s canonical scale).
* GT depth from frame-NNNNNN.depth.png (DEPTH-camera frame, mm) registered
  into the COLOR camera before use, then nearest-resized. The registration is
  the existing helper benchmark/datasets/seven_scenes.py
  SevenScenesDataset._register_depth_to_color, copied verbatim below (its
  import chain needs plyfile, which the cut3r env does not carry) -- including
  the empirically-fixed translation sign.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
from PIL import Image

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))

from lingbot_map.memory.cache_format import (  # noqa: E402
    CONF, DEPTH, FORMAT_VERSION, FP16_MAX, GT_C2W, GT_DEPTH,
    GT_INTRINSICS, META, POSE, REVISIT, TAPS, ClipMeta,
)
from scripts.memory.build_cache import (  # noqa: E402
    TAP_LAYERS, benchmark_resize, build_model, git_state,
)

# benchmark/datasets/seven_scenes.py constants (zinsmatt/7-Scenes-Calibration).
COLOR_K = {"fx": 525.0, "fy": 525.0, "cx": 320.0, "cy": 240.0}
DEPTH_K = {"fx": 598.84, "fy": 587.62, "cx": 320.0, "cy": 240.0}
DEPTH_TO_COLOR_T = np.array([0.023449, 0.006177, 0.010525], np.float64)
NATIVE_W, NATIVE_H = 640, 480



NATIVE_W, NATIVE_H = 640, 480
K_NRGBD = {"fx": 554.2562584220408, "fy": 554.2562584220408, "cx": 320.0, "cy": 240.0}


def load_poses_glcv(p):
    lines = [l for l in Path(p).read_text().splitlines() if l.strip()]
    n = len(lines) // 4
    P = np.array([[list(map(float, lines[4 * i + j].split())) for j in range(4)]
                  for i in range(n)], np.float64)
    P[:, :, 1:3] *= -1.0        # OpenGL c2w -> OpenCV c2w (CUT3R's own line)
    return P


def load_depth_m(p):
    d = cv2.imread(str(p), cv2.IMREAD_UNCHANGED)
    d = np.nan_to_num(d.astype(np.float32)) / 1000.0
    d[(d > 10) | (d < 1e-3)] = 0
    return d


def verify_pose_target(gt_c2w, pred_c2w):
    def consecutive(c2w):
        return np.einsum("nij,njk->nik", np.linalg.inv(c2w[:-1]), c2w[1:])
    ra, rb = consecutive(gt_c2w)[:, :3, :3], consecutive(pred_c2w)[:, :3, :3]
    cos = np.clip((np.trace(np.einsum("nij,nkj->nik", ra, rb),
                            axis1=1, axis2=2) - 1) / 2, -1, 1)
    resid = float(np.degrees(np.arccos(cos)).mean())
    return resid < 1.5, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/NRGBD")
    ap.add_argument("--scene", default="breakfast_room")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-len", type=int, default=96)
    ap.add_argument("--stride", type=int, default=5)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt/lingbot-map.pt")
    ap.add_argument("--device", default="cuda")
    ap.add_argument("--image-size", type=int, default=518)
    ap.add_argument("--patch-size", type=int, default=14)
    ap.add_argument("--target-width", type=int, default=518)
    ap.add_argument("--align", type=int, default=14)
    ap.add_argument("--area-budget", type=int, default=255000)
    ap.add_argument("--num-scale-frames", type=int, default=8)
    ap.add_argument("--kv-cache-sliding-window", type=int, default=64)
    ap.add_argument("--keyframe-interval", type=int, default=1)
    ap.add_argument("--max-frame-num", type=int, default=1024)
    ap.add_argument("--camera-num-iterations", type=int, default=4)
    ap.add_argument("--use-sdpa", action="store_true")
    args = ap.parse_args()

    scene_dir = Path(args.root) / args.scene
    frame_ids = [args.start + i * args.stride for i in range(args.clip_len)]
    for f in frame_ids:
        assert (scene_dir / "images" / f"img{f}.png").exists(), f

    W, H = benchmark_resize(NATIVE_W, NATIVE_H, args.target_width, args.align,
                            args.area_budget)
    patch_h, patch_w = H // args.patch_size, W // args.patch_size
    print(f"{args.scene}: {NATIVE_W}x{NATIVE_H} -> {W}x{H}, grid "
          f"{patch_w}x{patch_h}, frames {frame_ids[0]}..{frame_ids[-1]} "
          f"stride {args.stride}")

    rgb = np.stack([
        cv2.resize(np.array(Image.open(scene_dir / "images" / f"img{f}.png"
                                        ).convert("RGB")),
                   (W, H), interpolation=cv2.INTER_LANCZOS4)
        for f in frame_ids])

    import torch
    out = Path(args.out); out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)
    print("Building model...")
    model = build_model(args, device)
    P = patch_w * patch_h + model.aggregator.patch_start_idx
    E = model.aggregator.embed_dim * 2
    taps_mm = np.memmap(out / TAPS, dtype=np.float16, mode="w+",
                        shape=(args.clip_len, len(TAP_LAYERS), P, E))
    written = {"n": 0, "absmax": 0.0}

    def capture(_m, _i, output):
        tokens = output[0]
        stacked = torch.stack(tokens, dim=2)[0]
        written["absmax"] = max(written["absmax"], float(stacked.abs().amax()))
        i = written["n"]
        taps_mm[i:i + stacked.shape[0]] = stacked.to(torch.float16).cpu().numpy()
        written["n"] = i + stacked.shape[0]

    handle = model.aggregator.register_forward_hook(capture)
    images = torch.from_numpy(np.ascontiguousarray(
        rgb.transpose(0, 3, 1, 2))).float().div_(255.0)
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference_streaming(
                images, num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"))
    finally:
        handle.remove()
    assert written["n"] == args.clip_len
    assert written["absmax"] <= FP16_MAX
    taps_mm.flush()

    depth = pred["depth"][0, ..., 0].float().numpy()
    conf = pred["depth_conf"][0].float().numpy()
    pose_enc = pred["pose_enc"][0].float().numpy()
    np.memmap(out / DEPTH, dtype=np.float16, mode="w+",
              shape=depth.shape)[:] = depth.astype(np.float16)
    np.memmap(out / CONF, dtype=np.float16, mode="w+",
              shape=conf.shape)[:] = conf.astype(np.float16)
    np.save(out / POSE, pose_enc)

    poses = load_poses_glcv(scene_dir / "poses.txt")[frame_ids]
    c2w_rel = np.linalg.inv(poses[0])[None] @ poses
    from lingbot_map.memory.losses import pose_enc_to_c2w
    pred_c2w = pose_enc_to_c2w(torch.from_numpy(pose_enc), (H, W)).numpy()
    pred_rel = np.linalg.inv(pred_c2w[0])[None] @ pred_c2w
    trusted, resid = verify_pose_target(c2w_rel, pred_rel)
    print(f"  pose check: residual {resid:.3f} deg -> trusted={trusted}")

    gt_depth_m = np.stack([
        cv2.resize(load_depth_m(scene_dir / "depth" / f"depth{f}.png"),
                   (W, H), interpolation=cv2.INTER_NEAREST)
        for f in frame_ids]).astype(np.float32)
    K = np.array([[K_NRGBD["fx"], 0, K_NRGBD["cx"]],
                  [0, K_NRGBD["fy"], K_NRGBD["cy"]], [0, 0, 1]], np.float64)
    K[0] *= W / NATIVE_W
    K[1] *= H / NATIVE_H
    Ks = np.repeat(K[None], args.clip_len, 0).astype(np.float32)

    # Canonical scale (paper 3.2): mean anchor-cloud distance -> the model
    # sees this scene in the SAME units it was trained on, unlike the metric
    # 7-Scenes cache.
    from lingbot_map.memory.gt import canonical_scale
    s = canonical_scale(gt_depth_m, Ks.astype(np.float64), c2w_rel,
                        args.num_scale_frames)
    print(f"  canonical scale s = {s:.4f} m/unit")
    gt_depth = gt_depth_m / s
    c2w_canon = c2w_rel.copy(); c2w_canon[:, :3, 3] /= s

    np.memmap(out / GT_DEPTH, dtype=np.float16, mode="w+",
              shape=gt_depth.shape)[:] = gt_depth.astype(np.float16)
    np.save(out / GT_C2W, c2w_canon.astype(np.float32))
    np.save(out / GT_INTRINSICS, Ks)
    np.save(out / REVISIT, np.zeros(args.clip_len, np.float32))

    ratios = []
    for i in range(args.clip_len):
        m = gt_depth[i] > 0
        ratios.append(float(np.median(gt_depth[i][m]) / np.median(depth[i][m])))
    print(f"  gt/pred median-depth ratio (canonical units): mean "
          f"{np.mean(ratios):.3f} [{np.min(ratios):.3f},{np.max(ratios):.3f}]")

    origin_norm = np.linalg.norm(pose_enc[:, :3], axis=-1)
    git_sha, dirty = git_state(REPO)
    meta = ClipMeta(
        format_version=FORMAT_VERSION, scene=args.scene, clip_index=0,
        frame_ids=frame_ids, stride=args.stride, num_frames=args.clip_len,
        height=H, width=W, patch_h=patch_h, patch_w=patch_w, num_tokens=P,
        patch_start_idx=int(model.aggregator.patch_start_idx),
        tap_layers=TAP_LAYERS, embed_dim=E, tap_dtype="float16",
        scale_frames=args.num_scale_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        keyframe_interval=args.keyframe_interval,
        model_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        git_sha=git_sha, git_dirty=dirty,
        gt_scale=float(s), gt_convention="opencv_c2w",
        gt_pose_trusted=bool(trusted), gt_pose_residual_deg=resid,
        stats={"tap_absmax": written["absmax"],
               "origin_norm_pct": {str(p): float(np.percentile(origin_norm, p))
                                   for p in (0, 50, 100)},
               "gt_invalid_fraction": float((gt_depth <= 0).mean()),
               "gt_vs_pred_median_ratio_canonical": float(np.mean(ratios))})
    (out / META).write_text(meta.to_json())
    print(f"  wrote {out}")


if __name__ == "__main__":
    main()
