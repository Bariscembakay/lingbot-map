#!/usr/bin/env python3
"""Rank ScanNet++ scenes by how much the summary state could possibly contribute.

The state only holds what the KV cache has evicted, so a scene where everything
currently visible was also visible 30 frames ago gives it nothing to do. This
computes the revisit score per frame from GT geometry and ranks scenes by it.

Selecting the dev set this way costs nothing and changes no part of lingbot-map --
it is the one lever on signal strength that is purely dataset curation.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import gt   # noqa: E402


def score_scene(root: str, scene: str, stride: int, clip_len: int,
                window: int, anchor: int) -> dict | None:
    iphone = Path(root) / "data" / scene / "iphone"
    jp = iphone / "pose_intrinsic_imu.json"
    if not jp.exists() or not (iphone / "depth.bin").exists():
        return None
    n_avail = len(json.loads(jp.read_text()))
    L = min(clip_len, n_avail // stride)
    if L < 128:
        return None
    ids = [i * stride for i in range(L)]
    shape = gt.detect_depth_shape(iphone / "depth.bin")
    depth = gt.read_iphone_depth(iphone / "depth.bin", ids, shape)
    c2w, intr = gt.read_iphone_meta(jp, ids)
    intr_s = gt.scale_intrinsics(intr, (1440, 1920), shape)
    c2w_rel = gt.relative_to_first(c2w @ gt.CONVENTIONS["opengl"])
    rv = gt.revisit_score(depth, intr_s, c2w_rel, window=window)
    return {
        "scene": scene, "frames": L, "avail": n_avail,
        "rv_mean": float(rv.mean()), "rv_p50": float(np.percentile(rv, 50)),
        "rv_p90": float(np.percentile(rv, 90)),
        "frac_over_10": float((rv > 0.10).mean()),
        "frac_over_25": float((rv > 0.25).mean()),
        "invalid": float((depth <= 0).mean()),
        "depth_shape": list(shape),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", default="/data/ScanNetpp")
    ap.add_argument("--out", default=".agents/scratch/scene_revisit.json")
    ap.add_argument("--limit", type=int, default=40)
    ap.add_argument("--stride", type=int, default=20)
    ap.add_argument("--clip-len", type=int, default=320)
    ap.add_argument("--window", type=int, default=64)
    ap.add_argument("--anchor", type=int, default=8)
    ap.add_argument("--split", default=None, help="restrict to a splits/*.txt list")
    a = ap.parse_args()

    if a.split:
        scenes = [s.strip() for s in Path(f"{a.root}/splits/{a.split}").read_text().split() if s.strip()]
    else:
        scenes = sorted(os.listdir(f"{a.root}/data"))
    scenes = scenes[:a.limit]

    rows, t0 = [], time.time()
    for k, s in enumerate(scenes):
        try:
            r = score_scene(a.root, s, a.stride, a.clip_len, a.window, a.anchor)
        except Exception as e:
            print(f"  {s}: {type(e).__name__} {e}", file=sys.stderr); continue
        if r:
            rows.append(r)
        if (k + 1) % 10 == 0:
            print(f"  {k+1}/{len(scenes)} scanned, {len(rows)} usable, "
                  f"{time.time()-t0:.0f}s", file=sys.stderr)

    rows.sort(key=lambda r: -r["frac_over_10"])
    Path(a.out).parent.mkdir(parents=True, exist_ok=True)
    Path(a.out).write_text(json.dumps(
        {"stride": a.stride, "window": a.window, "clip_len": a.clip_len, "scenes": rows},
        indent=2))

    arr = np.array([r["frac_over_10"] for r in rows])
    print(f"\n{len(rows)} scenes, stride {a.stride}, window {a.window}")
    print(f"frac of frames >10% stale -- mean {arr.mean():.3f}  "
          f"p10 {np.percentile(arr,10):.3f}  p50 {np.percentile(arr,50):.3f}  "
          f"p90 {np.percentile(arr,90):.3f}  max {arr.max():.3f}")
    print(f"\n{'scene':12s} {'L':>4s} {'>10%':>7s} {'>25%':>7s} {'rv_p90':>7s}")
    for r in rows[:12]:
        print(f"{r['scene']:12s} {r['frames']:4d} {r['frac_over_10']:6.1%} "
              f"{r['frac_over_25']:6.1%} {r['rv_p90']:7.3f}")
    print("  ...")
    for r in rows[-3:]:
        print(f"{r['scene']:12s} {r['frames']:4d} {r['frac_over_10']:6.1%} "
              f"{r['frac_over_25']:6.1%} {r['rv_p90']:7.3f}")
    print(f"\nwrote {a.out}")


if __name__ == "__main__":
    main()
