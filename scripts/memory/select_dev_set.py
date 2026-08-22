#!/usr/bin/env python3
"""Pick the dev set from a revisit-score screening, and split it three ways.

train and val_top are curated for signal; val_median is deliberately taken from
the middle of the distribution, because a curated set is the right thing to
develop on and the wrong thing to report on.
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--screening", default=".agents/scratch/scene_revisit_300.json")
    ap.add_argument("--out", required=True)
    ap.add_argument("--train", type=int, default=16)
    ap.add_argument("--val-top", type=int, default=2)
    ap.add_argument("--val-median", type=int, default=2)
    ap.add_argument("--min-frames", type=int, default=320)
    ap.add_argument("--gb-per-clip", type=float, default=5.6)
    a = ap.parse_args()

    rows = [r for r in json.loads(Path(a.screening).read_text())["scenes"]
            if r["frames"] >= a.min_frames]
    n_t, n_v = a.train, a.val_top
    mid = len(rows) // 2
    groups = {
        "train": rows[:n_t],
        "val_top": rows[n_t:n_t + n_v],
        "val_median": rows[mid:mid + a.val_median],
    }
    lines = [f"{r['scene']}:0:{split}" for split, rs in groups.items() for r in rs]
    Path(a.out).write_text("\n".join(lines) + "\n")

    for split, rs in groups.items():
        mean = sum(r["frac_over_10"] for r in rs) / max(len(rs), 1)
        print(f"  {split:11s} {len(rs):3d} scenes, mean revisit {mean:.1%}")
    pool = sum(r["frac_over_10"] for r in rows) / len(rows)
    print(f"  pool        {len(rows):3d} scenes, mean revisit {pool:.1%}")
    print(f"{len(lines)} clips -> ~{len(lines) * a.gb_per_clip:.0f} GB")
    print(f"wrote {a.out}")


if __name__ == "__main__":
    main()
