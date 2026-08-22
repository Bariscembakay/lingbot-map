#!/usr/bin/env python3
"""Stage 1: precompute the frozen teacher cache for one ScanNet++ clip.

Runs lingbot-map once and stores, per frame, everything the spatial memory needs:
the 4 tap tensors (write inputs), depth + confidence (labels), and pose_enc (read
queries). Training then never runs the aggregator again -- the frozen forward
costs ~7 TFLOPs/frame and would otherwise re-execute once per clip pass.

Resize replicates the benchmark's two steps exactly (width->518 with the height
floored to a multiple of 14, then the area_budget cap), because those are what
produced the published numbers. For 4:3 the second step is a no-op; it only
bites near-square aspect ratios.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import subprocess
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory.cache_format import (  # noqa: E402
    CONF, DEPTH, FORMAT_VERSION, FP16_MAX, GT_C2W, GT_DEPTH, META, POSE,
    REVISIT, TAPS, ClipMeta,
)

TAP_LAYERS = [4, 11, 17, 23]


def probe_video(path: Path) -> tuple[int, int, int]:
    out = subprocess.run(
        ["ffprobe", "-v", "error", "-select_streams", "v:0",
         "-show_entries", "stream=width,height,nb_frames",
         "-of", "json", str(path)],
        capture_output=True, text=True, check=True,
    ).stdout
    s = json.loads(out)["streams"][0]
    return int(s["width"]), int(s["height"]), int(s["nb_frames"])


def benchmark_resize(w: int, h: int, target_w: int, align: int,
                     area_budget: int | None) -> tuple[int, int]:
    h1 = (int(target_w * h / w) // align) * align
    w1 = target_w
    if area_budget is None:
        return w1, h1
    s = min((area_budget / (w1 * h1)) ** 0.5, 1.0)
    return (max((int(w1 * s) // align) * align, align),
            max((int(h1 * s) // align) * align, align))


def decode_frames(path: Path, frame_ids: list[int], w: int, h: int) -> np.ndarray:
    """Decode exactly `frame_ids` from the video, scaled to (w, h)."""
    start, stride = frame_ids[0], frame_ids[1] - frame_ids[0]
    end = frame_ids[-1]
    sel = (rf"select='between(n\,{start}\,{end})"
           rf"*not(mod(n-{start}\,{stride}))'")
    cmd = [
        "ffmpeg", "-v", "error", "-i", str(path),
        "-vf", f"{sel},scale={w}:{h}:flags=lanczos",
        # select drops frames; without passthrough timing ffmpeg re-duplicates
        # them to keep a constant frame rate and the count comes back wrong.
        "-fps_mode", "passthrough",
        "-pix_fmt", "rgb24", "-f", "rawvideo", "-",
    ]
    nbytes = len(frame_ids) * h * w * 3
    proc = subprocess.run(cmd, capture_output=True, check=True)
    if len(proc.stdout) != nbytes:
        raise RuntimeError(
            f"decoded {len(proc.stdout)} bytes, expected {nbytes} "
            f"({len(proc.stdout) / (h * w * 3):.1f} frames vs {len(frame_ids)})"
        )
    return np.frombuffer(proc.stdout, np.uint8).reshape(len(frame_ids), h, w, 3)


def git_state(repo: Path) -> tuple[str, bool]:
    def run(*a):
        return subprocess.run(["git", "-C", str(repo), *a],
                              capture_output=True, text=True).stdout.strip()
    return run("rev-parse", "HEAD"), bool(run("status", "--porcelain"))


def build_model(args, device):
    import torch
    from lingbot_map.models.gct_stream import GCTStream
    model = GCTStream(
        img_size=args.image_size,
        patch_size=args.patch_size,
        enable_3d_rope=True,
        max_frame_num=args.max_frame_num,
        kv_cache_sliding_window=args.kv_cache_sliding_window,
        kv_cache_scale_frames=args.num_scale_frames,
        kv_cache_cross_frame_special=True,
        kv_cache_include_scale_frames=True,
        use_sdpa=args.use_sdpa,
        camera_num_iterations=args.camera_num_iterations,
    )
    ckpt = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    missing, unexpected = model.load_state_dict(ckpt.get("model", ckpt), strict=False)
    print(f"  missing={len(missing)} unexpected={len(unexpected)}")
    # Labels must be the raw head output: _normalize_predictions would rescale
    # depth after the fact and no per-frame decode of the cached taps could
    # reproduce it.
    assert not model.pred_normalization, "pred_normalization must be off"
    model = model.to(device).eval()
    model.aggregator = model.aggregator.to(dtype=torch.bfloat16)
    return model


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scannetpp-root", default="/data/ScanNetpp")
    ap.add_argument("--scene", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--clip-index", type=int, default=0)
    ap.add_argument("--clip-len", type=int, default=512)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--ckpt", default="ckpt/lingbot-map.pt")
    ap.add_argument("--device", default="cuda")
    # Defaults mirror benchmark/configs/methods/lingbot_map.yaml so the teacher
    # is the published configuration, not a variant of it.
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
    ap.add_argument("--decode-only", action="store_true",
                    help="stop after decoding; no GPU needed")
    ap.add_argument("--gt-convention", default="auto",
                    help="camera axis convention, or 'auto' to detect against the "
                         "model's own trajectory")
    args = ap.parse_args()

    repo = Path(__file__).resolve().parents[2]
    video = Path(args.scannetpp_root) / "data" / args.scene / "iphone" / "rgb.mkv"
    if not video.exists():
        raise SystemExit(f"missing {video}")

    vw, vh, nframes = probe_video(video)
    W, H = benchmark_resize(vw, vh, args.target_width, args.align, args.area_budget)
    patch_h, patch_w = H // args.patch_size, W // args.patch_size

    start = args.clip_index * args.clip_len * args.stride
    frame_ids = [start + i * args.stride for i in range(args.clip_len)]
    if frame_ids[-1] >= nframes:
        raise SystemExit(
            f"{args.scene}: clip {args.clip_index} needs frame {frame_ids[-1]} "
            f"but the video has {nframes}"
        )

    print(f"{args.scene} clip {args.clip_index}: {vw}x{vh}x{nframes} -> {W}x{H}, "
          f"{patch_w}x{patch_h}={patch_w * patch_h} patches, "
          f"frames {frame_ids[0]}..{frame_ids[-1]} stride {args.stride}")

    rgb = decode_frames(video, frame_ids, W, H)
    print(f"  decoded {rgb.shape}")
    if args.decode_only:
        return

    import torch

    out = Path(args.out)
    out.mkdir(parents=True, exist_ok=True)
    device = torch.device(args.device)

    print("Building model...")
    model = build_model(args, device)
    P = patch_w * patch_h + model.aggregator.patch_start_idx
    E = model.aggregator.embed_dim * 2  # taps are cat([frame_inter, global_inter])

    taps_mm = np.memmap(out / TAPS, dtype=np.float16, mode="w+",
                        shape=(args.clip_len, len(TAP_LAYERS), P, E))
    written = {"n": 0, "absmax": 0.0}

    def capture(_module, _inputs, output):
        tokens = output[0]
        assert len(tokens) == len(TAP_LAYERS), f"{len(tokens)} taps, expected {len(TAP_LAYERS)}"
        stacked = torch.stack(tokens, dim=2)[0]          # [S, 4, P, E]
        written["absmax"] = max(written["absmax"], float(stacked.abs().amax()))
        i = written["n"]
        s = stacked.shape[0]
        taps_mm[i:i + s] = stacked.to(torch.float16).cpu().numpy()
        written["n"] = i + s

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

    assert written["n"] == args.clip_len, f"captured {written['n']} of {args.clip_len} frames"
    if written["absmax"] > FP16_MAX:
        raise SystemExit(
            f"tap magnitude {written['absmax']:.1f} exceeds fp16 range {FP16_MAX}; "
            "store raw bfloat16 bits instead"
        )
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

    # Loss 1 needs GT, and GT is metric while the model is canonical. `gt.prepare`
    # applies the paper's anchor-frame normalisation and detects the camera axis
    # convention against the model's own trajectory rather than assuming one.
    from lingbot_map.memory import gt as gtmod
    from lingbot_map.memory.losses import pose_enc_to_c2w
    pred_c2w = pose_enc_to_c2w(
        torch.from_numpy(pose_enc), (H, W)
    ).numpy()
    g = gtmod.prepare(
        args.scannetpp_root, args.scene, frame_ids, H, W,
        anchor_frames=args.num_scale_frames,
        window=args.kv_cache_sliding_window,
        pred_c2w=pred_c2w,
        convention=args.gt_convention,
    )
    np.memmap(out / GT_DEPTH, dtype=np.float16, mode="w+",
              shape=g["gt_depth"].shape)[:] = g["gt_depth"].astype(np.float16)
    np.save(out / GT_C2W, g["gt_c2w"])
    np.save(out / REVISIT, g["revisit"])

    # The sinusoidal band for ray origins has to cover the range that actually
    # occurs; the model's canonical scale fixes the unit but not the extent.
    origin_norm = np.linalg.norm(pose_enc[:, :3], axis=-1)
    git_sha, dirty = git_state(repo)
    meta = ClipMeta(
        format_version=FORMAT_VERSION,
        scene=args.scene,
        clip_index=args.clip_index,
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
        gt_scale=g["scale"], gt_convention=g["convention"],
        gt_pose_trusted=g["pose_trusted"],
        gt_pose_residual_deg=g["pose_residual_deg"],
        stats={
            "tap_absmax": written["absmax"],
            "origin_norm_pct": {
                str(p): float(np.percentile(origin_norm, p))
                for p in (0, 1, 25, 50, 75, 99, 100)
            },
            "depth_pct": {
                str(p): float(np.percentile(depth, p))
                for p in (1, 25, 50, 75, 99)
            },
            "conf_pct": {
                str(p): float(np.percentile(conf, p))
                for p in (1, 50, 99)
            },
            "gt_invalid_fraction": g["invalid_fraction"],
            "gt_convention_scores_deg": g["convention_scores_deg"],
            "gt_pose_residual_deg": g["pose_residual_deg"],
            "gt_pose_trusted": g["pose_trusted"],
            "revisit_pct": {
                str(p): float(np.percentile(g["revisit"], p))
                for p in (10, 25, 50, 75, 90, 100)
            },
            "revisit_frac_over_10pct": float((g["revisit"] > 0.10).mean()),
            "revisit_frac_over_25pct": float((g["revisit"] > 0.25).mean()),
        },
    )
    (out / META).write_text(meta.to_json())

    total = sum(f.stat().st_size for f in out.iterdir())
    print(f"  wrote {out}  {total / 1e9:.2f} GB  "
          f"({total / args.clip_len / 1e6:.1f} MB/frame)")
    print(json.dumps(meta.stats, indent=2))


if __name__ == "__main__":
    main()
