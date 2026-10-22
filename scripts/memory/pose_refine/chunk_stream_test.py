#!/usr/bin/env python3
"""Does GCTStream.inference_chunked reproduce frame-by-frame inference_streaming? Compares the cache
captures (specials at the 4 taps, pose_enc, lingbot-DINO CLS) of the first clips of a plan, and
times both. Same model settings as build_clip_cache.py.
Usage: chunk_stream_test.py --plan plan.json [--frames 320] [--chunks 4 16 32] [--clips 2]
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


def run(model, x, chunk, attn="sdpa"):
    agg = model.aggregator
    psi = agg.patch_start_idx
    acc = {"spec": [], "cls": []}
    hs = [agg.register_forward_hook(lambda _m, _i, o: acc["spec"].append(torch.stack(o[0], 2)[0][:, :, :psi].float())),
          agg.patch_embed.register_forward_hook(lambda _m, _i, o: acc["cls"].append(o["x_norm_clstoken"].float()))]
    torch.cuda.synchronize()
    torch.cuda.reset_peak_memory_stats()
    t = time.time()
    try:
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            if chunk:
                pred = model.inference_chunked(x, chunk=chunk, num_scale_frames=A, attn=attn)
            else:
                pred = model.inference_streaming(x, num_scale_frames=A, keyframe_interval=1, output_device=None)
        torch.cuda.synchronize()
    finally:
        for h in hs:
            h.remove()
    dt = time.time() - t
    return {"spec": torch.cat(acc["spec"]), "cls": torch.cat(acc["cls"]), "pose": pred["pose_enc"][0].float()}, dt


def diff(a, b):
    a, b = a.flatten(1), b.flatten(1)
    cos = torch.nn.functional.cosine_similarity(a, b, dim=1).min().item()
    rel = ((a - b).norm(dim=1) / b.norm(dim=1).clamp(min=1e-6)).max().item()
    return f"min-cos {cos:.6f} max-rel-err {rel:.2e}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=320)
    ap.add_argument("--chunks", type=int, nargs="+", default=[4, 16, 32])
    ap.add_argument("--clips", type=int, default=2)
    ap.add_argument("--attn", nargs="+", default=["sdpa", "flex"])
    ap.add_argument("--profile", action="store_true", help="kernel table of each chunked run (after the timed one)")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    src = source(plan["summary"]["dataset"])
    clips = [c for c in plan["clips"] if len(c["frames"]) >= args.frames][: args.clips]
    m = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=A,
                        use_sdpa=True, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(m, torch.device("cuda"))
    model.depth_head = None
    print(torch.cuda.get_device_name(), flush=True)
    for c in clips:
        imgs, _ = src.frames(c["seq"], c["frames"][: args.frames])
        x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))).float().div_(255)
        print(f"== {c['seq']}: {len(x)} frames {tuple(x.shape[-2:])}", flush=True)
        run(model, x[:24], 0)   # warm-up
        ref, dt = run(model, x, 0)
        print(f"[stream] {len(x) / dt:.2f} fps, peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
        again, _ = run(model, x, 0)
        print(f"   repeat: specials {diff(again['spec'], ref['spec'])} | pose {diff(again['pose'], ref['pose'])}", flush=True)
        for att, ch in [(a, c) for a in args.attn for c in args.chunks]:
            try:
                if att == "flex":
                    run(model, x[:8 + 2 * ch], ch, att)   # compile outside the timing
                out, dt = run(model, x, ch, att)
            except Exception as e:
                print(f"[{att} chunk {ch}] FAILED {type(e).__name__}: {str(e)[:300]}", flush=True)
                torch.cuda.empty_cache()
                continue
            print(f"[{att} chunk {ch}] {len(x) / dt:.2f} fps ({dt:.1f} s), peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)
            if args.profile:
                from torch.profiler import ProfilerActivity, profile
                with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof:
                    run(model, x[: 8 + 4 * ch], ch, att)
                print(prof.key_averages().table(sort_by="cuda_time_total", row_limit=15), flush=True)
            print(f"   specials {diff(out['spec'], ref['spec'])} | pose {diff(out['pose'], ref['pose'])} | "
                  f"cls {diff(out['cls'], ref['cls'])}", flush=True)
            for a, b in ((8, 72), (72, 200), (200, len(x))):   # before / around / after the window fills
                if b > a:
                    print(f"     frames {a}-{b}: specials {diff(out['spec'][a:b], ref['spec'][a:b])} | "
                          f"pose {diff(out['pose'][a:b], ref['pose'][a:b])}", flush=True)


if __name__ == "__main__":
    main()
