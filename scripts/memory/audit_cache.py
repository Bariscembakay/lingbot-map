#!/usr/bin/env python3
"""Audit a teacher cache before spending GPU-hours training on it.

Checks the things that are cheap here and expensive to discover in a loss curve:
array sizes against the metadata, format version, one geometry, whether the GT
pose target is trusted, and whether every clip came from the same checkpoint and
a clean tree.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
from collections import Counter


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--expect", type=int, default=None, help="expected clip count")
    a = ap.parse_args()

    metas = sorted(glob.glob(f"{a.cache}/*/*/meta.json")) or \
            sorted(glob.glob(f"{a.cache}/*/meta.json"))
    if not metas:
        raise SystemExit(f"no clips under {a.cache}")

    rows, bad = [], []
    for m in metas:
        d = json.load(open(m))
        dd = os.path.dirname(m)
        expected = {
            "taps.npy": d["num_frames"] * 4 * d["num_tokens"] * d["embed_dim"] * 2,
            "depth.npy": d["num_frames"] * d["height"] * d["width"] * 2,
            "conf.npy": d["num_frames"] * d["height"] * d["width"] * 2,
            "gt_depth.npy": d["num_frames"] * d["height"] * d["width"] * 2,
        }
        for f, sz in expected.items():
            p = os.path.join(dd, f)
            if not os.path.exists(p):
                bad.append((os.path.basename(dd), f, "missing"))
            elif os.path.getsize(p) != sz:
                bad.append((os.path.basename(dd), f,
                            f"{os.path.getsize(p)} != {sz}"))
        rows.append((os.path.basename(dd), d))

    def uni(key, fn=lambda d: d[key]):
        return sorted(set(fn(d) for _, d in rows))

    print(f"clips                {len(rows)}" +
          (f" / {a.expect}" if a.expect else ""))
    print(f"size-mismatched      {len(bad)}")
    for b in bad[:5]:
        print(f"    {b}")
    print(f"format_version       {uni('format_version')}")
    print(f"num_frames           {uni('num_frames')}")
    print(f"num_tokens           {uni('num_tokens')}")
    print(f"geometry             {uni('width')} x {uni('height')}")
    print(f"stride               {uni('stride')}")
    print(f"model_sha256         {[s[:12] for s in uni('model_sha256')]}")
    print(f"git_sha              {[s[:7] for s in uni('git_sha')]}  dirty={uni('git_dirty')}")
    print(f"gt_convention        {Counter(d['gt_convention'] for _, d in rows).most_common()}")

    trusted = sum(d["gt_pose_trusted"] for _, d in rows)
    pr = sorted(d["gt_pose_residual_deg"] for _, d in rows)
    print(f"gt_pose_trusted      {trusted}/{len(rows)}")
    print(f"pose residual deg    min {pr[0]:.2f}  median {pr[len(pr)//2]:.2f}  max {pr[-1]:.2f}")
    tm = max(d["stats"]["tap_absmax"] for _, d in rows)
    print(f"tap_absmax           {tm:.0f}   (fp16 range 65504)")
    rv = sorted(d["stats"]["revisit_frac_over_10pct"] for _, d in rows)
    print(f"revisit >10%         min {rv[0]:.1%}  mean {sum(rv)/len(rv):.1%}  max {rv[-1]:.1%}")
    inv = max(d["stats"]["gt_invalid_fraction"] for _, d in rows)
    print(f"gt invalid pixels    max {inv:.2%}")

    problems = []
    if bad:
        problems.append(f"{len(bad)} size-mismatched files")
    if len(uni("model_sha256")) > 1:
        problems.append("clips built from different checkpoints -- not one dataset")
    if any(uni("git_dirty")):
        problems.append("some clips built from a dirty tree -- not attributable")
    if trusted < len(rows):
        problems.append(f"{len(rows)-trusted} clips have untrusted GT poses "
                        f"(pose losses will refuse them)")
    print()
    if problems:
        print("PROBLEMS:")
        for p in problems:
            print(f"  - {p}")
    else:
        print("clean")


if __name__ == "__main__":
    main()
