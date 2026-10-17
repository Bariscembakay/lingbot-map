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


def load_depth_m(depth_file: Path) -> np.ndarray:
    """SevenScenesDataset._load_depth: 16-bit mm PNG -> float32 metres, 0=invalid."""
    d = np.array(Image.open(depth_file), dtype=np.float32)
    assert d.shape == (NATIVE_H, NATIVE_W), d.shape
    d[d == 65535] = 0
    d = np.nan_to_num(d, nan=0.0) / 1000.0
    d[d > 10.0] = 0
    d[d < 1e-3] = 0
    return d


def register_depth_to_color(depth: np.ndarray) -> np.ndarray:
    """Verbatim SevenScenesDataset._register_depth_to_color (sign fixed
    empirically there; do not flip)."""
    H, W = depth.shape
    v, u = np.nonzero(depth > 0)
    if v.size == 0:
        return np.zeros_like(depth)
    d = depth[v, u].astype(np.float64)

    x = (u - DEPTH_K["cx"]) / DEPTH_K["fx"] * d
    y = (v - DEPTH_K["cy"]) / DEPTH_K["fy"] * d
    x, y, z = (x - DEPTH_TO_COLOR_T[0],
               y - DEPTH_TO_COLOR_T[1],
               d - DEPTH_TO_COLOR_T[2])

    ok = z > 1e-6
    x, y, z = x[ok], y[ok], z[ok]
    uc = np.rint(COLOR_K["fx"] * x / z + COLOR_K["cx"]).astype(np.int32)
    vc = np.rint(COLOR_K["fy"] * y / z + COLOR_K["cy"]).astype(np.int32)

    inb = (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
    uc, vc, z = uc[inb], vc[inb], z[inb]

    order = np.argsort(-z)
    out = np.zeros((H, W), np.float32)
    out[vc[order], uc[order]] = z[order].astype(np.float32)
    return out


def verify_pose_target(gt_c2w: np.ndarray, pred_c2w: np.ndarray):
    """Verbatim lingbot_map/memory/gt.py (that module imports lz4, which the
    cut3r env lacks): consecutive relative rotations, world-frame/scale
    invariant, so a residual is a real disagreement."""
    def geodesic_deg(ra, rb):
        m = np.einsum("nij,njk->nik", np.transpose(ra, (0, 2, 1)), rb)
        cos = np.clip((np.trace(m, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def consecutive(c2w):
        a, b = c2w[:-1, :3, :3], c2w[1:, :3, :3]
        return np.einsum("nij,njk->nik", np.transpose(a, (0, 2, 1)), b)

    resid = float(geodesic_deg(consecutive(gt_c2w), consecutive(pred_c2w)).mean())
    return resid <= 1.5, resid


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/7-Scenes")
    ap.add_argument("--scene", default="chess/seq-03")
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-len", type=int, default=32)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--start", type=int, default=0)
    ap.add_argument("--ckpt", default="ckpt/lingbot-map.pt")
    ap.add_argument("--device", default="cuda")
    # build_model config: defaults mirror scripts/memory/build_cache.py, which
    # mirrors benchmark/configs/methods/lingbot_map.yaml.
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

    scene_name, seq = args.scene.split("/")
    split_file = Path(args.root) / scene_name / "TestSplit.txt"
    test_seqs = [f"seq-{int(s.strip().replace('sequence', '')):02d}"
                 for s in split_file.read_text().split() if s.strip()]
    assert seq in test_seqs, f"{seq} not in {scene_name} TestSplit {test_seqs}"

    seq_dir = Path(args.root) / args.scene
    frame_ids = [args.start + i * args.stride for i in range(args.clip_len)]
    for f in frame_ids:
        assert (seq_dir / f"frame-{f:06d}.color.png").exists(), f

    W, H = benchmark_resize(NATIVE_W, NATIVE_H, args.target_width, args.align,
                            args.area_budget)
    patch_h, patch_w = H // args.patch_size, W // args.patch_size
    print(f"{args.scene}: {NATIVE_W}x{NATIVE_H} -> {W}x{H}, patch grid "
          f"{patch_w}x{patch_h}={patch_w * patch_h}, frames "
          f"{frame_ids[0]}..{frame_ids[-1]} stride {args.stride}")

    rgb = np.stack([
        cv2.resize(np.array(Image.open(
            seq_dir / f"frame-{f:06d}.color.png").convert("RGB")),
            (W, H), interpolation=cv2.INTER_LANCZOS4)
        for f in frame_ids])
    print(f"  decoded {rgb.shape}")

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Building model...")
    model = build_model(args, device)
    P = patch_w * patch_h + model.aggregator.patch_start_idx
    E = model.aggregator.embed_dim * 2

    taps_mm = np.memmap(out / TAPS, dtype=np.float16, mode="w+",
                        shape=(args.clip_len, len(TAP_LAYERS), P, E))
    written = {"n": 0, "absmax": 0.0}

    def capture(_module, _inputs, output):
        tokens = output[0]
        assert len(tokens) == len(TAP_LAYERS)
        stacked = torch.stack(tokens, dim=2)[0]          # [S, 4, P, E]
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
                images,
                num_scale_frames=args.num_scale_frames,
                keyframe_interval=args.keyframe_interval,
                output_device=torch.device("cpu"),
            )
    finally:
        handle.remove()

    assert written["n"] == args.clip_len, written
    if written["absmax"] > FP16_MAX:
        raise SystemExit(f"tap absmax {written['absmax']:.1f} > fp16 range")
    taps_mm.flush()

    depth = pred["depth"][0, ..., 0].float().numpy()
    conf = pred["depth_conf"][0].float().numpy()
    pose_enc = pred["pose_enc"][0].float().numpy()
    assert depth.shape == (args.clip_len, H, W), depth.shape

    np.memmap(out / DEPTH, dtype=np.float16, mode="w+",
              shape=depth.shape)[:] = depth.astype(np.float16)
    np.memmap(out / CONF, dtype=np.float16, mode="w+",
              shape=conf.shape)[:] = conf.astype(np.float16)
    np.save(out / POSE, pose_enc)

    # ---- GT: metric, relative to frame 0, registered-to-color depth.
    c2w_raw = np.stack([
        np.loadtxt(seq_dir / f"frame-{f:06d}.pose.txt").reshape(4, 4)
        for f in frame_ids]).astype(np.float64)
    c2w_rel = np.linalg.inv(c2w_raw[0])[None] @ c2w_raw

    from lingbot_map.memory.losses import pose_enc_to_c2w
    pred_c2w = pose_enc_to_c2w(torch.from_numpy(pose_enc), (H, W)).numpy()
    pred_rel = np.linalg.inv(pred_c2w[0])[None] @ pred_c2w
    trusted, resid = verify_pose_target(c2w_rel, pred_rel)
    print(f"  pose check: residual {resid:.3f} deg -> trusted={trusted}")

    gt_depth = np.stack([
        cv2.resize(register_depth_to_color(
            load_depth_m(seq_dir / f"frame-{f:06d}.depth.png")),
            (W, H), interpolation=cv2.INTER_NEAREST)
        for f in frame_ids]).astype(np.float32)

    np.memmap(out / GT_DEPTH, dtype=np.float16, mode="w+",
              shape=gt_depth.shape)[:] = gt_depth.astype(np.float16)
    K = np.array([[COLOR_K["fx"], 0, COLOR_K["cx"]],
                  [0, COLOR_K["fy"], COLOR_K["cy"]],
                  [0, 0, 1]], np.float64)
    K[0] *= W / NATIVE_W
    K[1] *= H / NATIVE_H
    np.save(out / GT_C2W, c2w_rel.astype(np.float32))
    np.save(out / GT_INTRINSICS,
            np.repeat(K[None], args.clip_len, 0).astype(np.float32))
    np.save(out / REVISIT, np.zeros(args.clip_len, np.float32))

    # ---- Sanity: valid fraction, depth range, GT-vs-predicted median ratio.
    valid_frac = (gt_depth > 0).reshape(args.clip_len, -1).mean(1)
    ratios = []
    for i in range(args.clip_len):
        m = gt_depth[i] > 0
        ratios.append(float(np.median(gt_depth[i][m]) / np.median(depth[i][m])))
    print("  per-frame valid GT fraction / gt-depth median / gt-vs-pred ratio:")
    for i in range(args.clip_len):
        m = gt_depth[i] > 0
        print(f"    f{frame_ids[i]:4d}  valid={valid_frac[i]:.3f}  "
              f"z_med={np.median(gt_depth[i][m]):.3f} m  "
              f"gt/pred={ratios[i]:.3f}")
    v = gt_depth[gt_depth > 0]
    print(f"  valid frac mean {valid_frac.mean():.3f} min {valid_frac.min():.3f}; "
          f"gt depth range [{v.min():.3f}, {v.max():.3f}] m, "
          f"pct 1/50/99 = {np.percentile(v, [1, 50, 99]).round(3)}")
    print(f"  gt/pred median-depth ratio: mean {np.mean(ratios):.3f} "
          f"min {np.min(ratios):.3f} max {np.max(ratios):.3f}")

    origin_norm = np.linalg.norm(pose_enc[:, :3], axis=-1)
    git_sha, dirty = git_state(REPO)
    meta = ClipMeta(
        format_version=FORMAT_VERSION,
        scene=args.scene,
        clip_index=0,
        frame_ids=frame_ids,
        stride=args.stride,
        num_frames=args.clip_len,
        height=H, width=W,
        patch_h=patch_h, patch_w=patch_w,
        num_tokens=P,
        patch_start_idx=int(model.aggregator.patch_start_idx),
        tap_layers=TAP_LAYERS,
        embed_dim=E,
        tap_dtype="float16",
        scale_frames=args.num_scale_frames,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        keyframe_interval=args.keyframe_interval,
        model_sha256=hashlib.sha256(Path(args.ckpt).read_bytes()).hexdigest(),
        git_sha=git_sha, git_dirty=dirty,
        gt_scale=1.0,                      # 7-Scenes GT kept metric
        gt_convention="opencv_c2w",
        gt_pose_trusted=bool(trusted),
        gt_pose_residual_deg=resid,
        stats={
            "tap_absmax": written["absmax"],
            "origin_norm_pct": {str(p): float(np.percentile(origin_norm, p))
                                for p in (0, 1, 25, 50, 75, 99, 100)},
            "depth_pct": {str(p): float(np.percentile(depth, p))
                          for p in (1, 25, 50, 75, 99)},
            "conf_pct": {str(p): float(np.percentile(conf, p))
                         for p in (1, 50, 99)},
            "gt_invalid_fraction": float((gt_depth <= 0).mean()),
            "gt_convention_scores_deg": None,
            "gt_pose_residual_deg": resid,
            "gt_pose_trusted": bool(trusted),
            "gt_depth_pct_valid": {str(p): float(np.percentile(v, p))
                                   for p in (1, 25, 50, 75, 99)},
            "gt_vs_pred_median_ratio": {"mean": float(np.mean(ratios)),
                                        "min": float(np.min(ratios)),
                                        "max": float(np.max(ratios))},
            "depth_registration": "reprojected depth-cam -> color-cam "
                                  "(seven_scenes.py convention, minus-T)",
            "rgb_resize": "cv2 INTER_LANCZOS4 (build_cache.py uses ffmpeg lanczos)",
        },
    )
    (out / META).write_text(meta.to_json())

    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"  wrote {out}  {total / 1e9:.2f} GB")


if __name__ == "__main__":
    main()
