#!/usr/bin/env python3
"""Plan whole-trajectory cache clips: every frame (stride 1), one clip per valid run, no length cap,
only sequences that revisit a place (loop_anydir in dataset_loop_stats.json).

Lingbot is run on these clips in its default mode (keyframe interval ceil(N/320) beyond 320 frames),
so the refiner sees the same tokens it gets at test time, over the whole trajectory.
Output: the plan_clips.py JSON format (summary, args, clips[{seq, frames, stride, depth_med}], per_seq).
Usage: plan_full_trajectories.py --dataset tartanair-v2 --out plan.json [--min-len 256]
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from clip_sources import source
from plan_clips import runs_of_valid

STATS = Path("/group/compact-3dmem/campaigns/spatial_memory/_data_prep/dataset_loop_stats.json")
A = None


def plan_seq(seq):
    src = source(A.dataset)
    try:
        T = src.poses(seq)
    except Exception as e:
        return seq, [], {"error": f"{type(e).__name__}: {e}"}
    runs = runs_of_valid(T, A.min_len)
    clips = [{"seq": seq, "frames": list(range(a, b)), "stride": 1, "depth_med": None} for a, b in runs]
    return seq, clips, {"frames": len(T), "runs": [[a, b] for a, b in runs]}


def init(args):
    global A
    A = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--min-len", type=int, default=256)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    stats = json.loads(STATS.read_text())[args.dataset]["per_seq"]
    seqs = [r["seq"] for r in stats if r.get("loop_anydir")]
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args,)) as ex:
        res = list(ex.map(plan_seq, seqs, chunksize=4))
    clips = [c for _, cs, _ in res for c in cs]
    lens = np.array([len(c["frames"]) for c in clips])
    summary = {"dataset": args.dataset, "sequences": len(seqs), "errors": sum("error" in i for _, _, i in res),
               "clips": len(clips), "frames": int(lens.sum()), "clip_frames_median": float(np.median(lens)),
               "clip_frames_max": int(lens.max()), "mode": "whole trajectories, stride 1, loop sequences only"}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summary, "args": {k: str(v) for k, v in vars(args).items()},
                                    "clips": clips, "per_seq": {s: i for s, _, i in res}}))
    print(f"[plan] {summary}", flush=True)


if __name__ == "__main__":
    main()
