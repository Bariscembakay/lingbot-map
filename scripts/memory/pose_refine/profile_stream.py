#!/usr/bin/env python3
"""Where does lingbot's per-frame streaming time go? Profiles inference_streaming on N frames of
the first clip of a plan (same model settings as build_clip_cache.py) and prints the top ops.
Usage: profile_stream.py --plan plan.json [--frames 160]
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


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--frames", type=int, default=160)
    ap.add_argument("--profile-frames", type=int, default=96)
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    c = plan["clips"][0]
    src = source(plan["summary"]["dataset"])
    imgs, _ = src.frames(c["seq"], c["frames"][:args.frames])
    x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))).float().div_(255)
    m = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=8,
                        use_sdpa=True, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(m, torch.device("cuda"))
    model.depth_head = None
    print(f"frames {len(x)} {tuple(x.shape[-2:])}", flush=True)
    stamps = []
    h = model.aggregator.register_forward_hook(lambda *_: (torch.cuda.synchronize(), stamps.append(time.time())))
    with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_streaming(x, num_scale_frames=8, keyframe_interval=1, output_device=torch.device("cpu"))
    h.remove()
    dt = np.diff(stamps) * 1000
    for a in range(0, len(dt), 100):
        print(f"[per-frame] frames {a + 9}-{min(a + 108, len(dt) + 8)}: {dt[a:a + 100].mean():.0f} ms/frame", flush=True)
    for label, out_dev in (("output cpu", torch.device("cpu")), ("output gpu", None)):
        torch.cuda.synchronize()
        t = time.time()
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            model.inference_streaming(x, num_scale_frames=8, keyframe_interval=1, output_device=out_dev)
        torch.cuda.synchronize()
        print(f"[time] {label}: {len(x) / (time.time() - t):.2f} fps", flush=True)
    from torch.profiler import ProfilerActivity, profile
    with profile(activities=[ProfilerActivity.CPU, ProfilerActivity.CUDA]) as prof, torch.no_grad(), \
            torch.amp.autocast("cuda", dtype=torch.bfloat16):
        model.inference_streaming(x[:args.profile_frames], num_scale_frames=8, keyframe_interval=1, output_device=torch.device("cpu"))
    ka = prof.key_averages()
    print(ka.table(sort_by="cuda_time_total", row_limit=18), flush=True)
    print(ka.table(sort_by="cpu_time_total", row_limit=12), flush=True)


if __name__ == "__main__":
    main()
