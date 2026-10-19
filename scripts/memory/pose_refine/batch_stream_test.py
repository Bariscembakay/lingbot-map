#!/usr/bin/env python3
"""Can the cache builder stream B clips at once? Checks that batched inference_streaming reproduces the
batch-1 captures (specials at the 4 taps, pose_enc, lingbot-DINO CLS) per clip, including a clip padded
at the end by repeating its last frame (streaming is causal, so padding must not change earlier frames),
and measures clip-frames/s per batch size. Same model settings as build_clip_cache.py.
Usage: batch_stream_test.py --plan plan.json [--frames 256] [--batches 1 2 4 8]
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

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
from clip_sources import source  # noqa: E402
from scripts.memory.build_cache import build_model  # noqa: E402

A = 8


def run(model, x):
    """x [B, S, 3, H, W] on CPU -> per-clip captures"""
    agg = model.aggregator
    psi = agg.patch_start_idx
    acc = {"spec": [], "cls": []}
    B = x.shape[0]
    hs = [agg.register_forward_hook(lambda _m, _i, o: acc["spec"].append(torch.stack(o[0], 2)[:, :, :, :psi].float())),
          agg.patch_embed.register_forward_hook(lambda _m, _i, o: acc["cls"].append(o["x_norm_clstoken"].float().reshape(B, -1, 1024)))]
    torch.cuda.synchronize()
    t = time.time()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference_streaming(x, num_scale_frames=A, keyframe_interval=1, output_device=None)
        torch.cuda.synchronize()
    finally:
        for h in hs:
            h.remove()
    dt = time.time() - t
    return {"spec": torch.cat(acc["spec"], 1), "cls": torch.cat(acc["cls"], 1), "pose": pred["pose_enc"].float()}, dt


def diff(a, b):
    a, b = a.flatten(1), b.flatten(1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1).min().item()
    rel = ((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-6)).max().item()
    return f"min-cos {cos:.6f} max-rel-err {rel:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=256)
    ap.add_argument("--batches", type=int, nargs="+", default=[1, 2, 4, 8])
    ap.add_argument("--pad", type=int, default=64, help="frames of end padding on the last clip of the largest batch")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    src = source(plan["summary"]["dataset"])
    nb = max(args.batches)
    clips = [c for c in plan["clips"] if len(c["frames"]) >= args.frames]
    imgs, hw = [], None
    for c in clips:
        im, _ = src.frames(c["seq"], c["frames"][:args.frames])
        if hw is None or im.shape[1:3] == hw:
            hw = im.shape[1:3]
            imgs.append(im)
        if len(imgs) == nb:
            break
    x = torch.from_numpy(np.ascontiguousarray(np.stack(imgs).transpose(0, 1, 4, 2, 3))).float().div_(255)
    print(f"{plan['summary']['dataset']}: {nb} clips x {args.frames} frames at {tuple(hw)}, {torch.cuda.get_device_name()}", flush=True)
    m = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=A,
                        use_sdpa=True, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(m, torch.device("cuda"))
    model.depth_head = None
    run(model, x[:1, :32])   # warm-up

    ref = [run(model, x[i:i + 1])[0] for i in range(nb)]
    again, _ = run(model, x[:1])
    print(f"[repeat B=1, clip 0] specials {diff(again['spec'], ref[0]['spec'])} | pose {diff(again['pose'], ref[0]['pose'])}", flush=True)

    for B in args.batches:
        xb = x[:B].clone()
        if B == nb and B > 1 and args.pad:
            xb[-1, -args.pad:] = xb[-1, -args.pad - 1]
        out, dt = run(model, xb)
        mem = torch.cuda.max_memory_allocated() / 2**30
        print(f"[B={B}] {B * args.frames / dt:.1f} clip-frames/s ({args.frames / dt:.2f} fps per clip), peak {mem:.1f} GiB", flush=True)
        for i in range(B):
            n = args.frames - args.pad if (B == nb and B > 1 and args.pad and i == B - 1) else args.frames
            r = ref[i]
            print(f"   clip {i}{' (padded, first %d frames)' % n if n < args.frames else ''}: "
                  f"specials {diff(out['spec'][i:i + 1, :n], r['spec'][:, :n])} | "
                  f"pose {diff(out['pose'][i:i + 1, :n], r['pose'][:, :n])} | "
                  f"cls {diff(out['cls'][i:i + 1, :n], r['cls'][:, :n])}", flush=True)
        torch.cuda.reset_peak_memory_stats()


if __name__ == "__main__":
    main()
