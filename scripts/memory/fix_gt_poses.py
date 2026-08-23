#!/usr/bin/env python3
"""Recompute gt_c2w (and its trust flag) in place for an existing cache.

The taps, depth, conf and pose_enc in a built cache are unaffected by the pose
bug, so there is no reason to re-run the teacher over 106 GB. Only `gt_c2w.npy`
and two metadata fields change, and both are CPU work.
"""
from __future__ import annotations

import argparse
import glob
import json
import os
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import gt as gtmod                      # noqa: E402
from lingbot_map.memory.cache_format import GT_C2W, META        # noqa: E402
from lingbot_map.memory.losses import pose_enc_to_c2w           # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--scannetpp-root", default="/data/ScanNetpp")
    ap.add_argument("--dry-run", action="store_true")
    a = ap.parse_args()

    metas = sorted(glob.glob(f"{a.cache}/*/*/{META}")) or sorted(glob.glob(f"{a.cache}/*/{META}"))
    if not metas:
        raise SystemExit(f"no clips under {a.cache}")

    resids = []
    for mp in metas:
        dd = os.path.dirname(mp)
        m = json.loads(Path(mp).read_text())
        pose = np.load(os.path.join(dd, "pose_enc.npy"))
        pred = pose_enc_to_c2w(torch.from_numpy(pose), (m["height"], m["width"])).numpy()

        g = gtmod.prepare(
            a.scannetpp_root, m["scene"], m["frame_ids"], m["height"], m["width"],
            anchor_frames=m["scale_frames"], window=m["kv_cache_sliding_window"],
            pred_c2w=pred.astype(np.float64), convention="opencv_c2w",
        )
        resids.append(g["pose_residual_deg"])
        flag = "OK " if g["pose_trusted"] else "BAD"
        print(f"  {flag} {os.path.basename(dd):24s} residual {g['pose_residual_deg']:6.3f} deg  "
              f"scale {g['scale']:.3f}")
        if a.dry_run:
            continue
        np.save(os.path.join(dd, GT_C2W), g["gt_c2w"])
        m["gt_convention"] = g["convention"]
        m["gt_pose_trusted"] = bool(g["pose_trusted"])
        m["gt_pose_residual_deg"] = g["pose_residual_deg"]
        m["gt_scale"] = g["scale"]
        m["stats"]["gt_pose_residual_deg"] = g["pose_residual_deg"]
        m["stats"]["gt_pose_trusted"] = bool(g["pose_trusted"])
        m["stats"]["gt_convention_scores_deg"] = None
        Path(mp).write_text(json.dumps(m, indent=2))

    r = sorted(resids)
    print(f"\n{len(r)} clips  residual deg: min {r[0]:.3f} median {r[len(r)//2]:.3f} max {r[-1]:.3f}")
    print(f"trusted: {sum(x <= gtmod.POSE_TRUST_THRESHOLD_DEG for x in r)}/{len(r)}"
          f"  (threshold {gtmod.POSE_TRUST_THRESHOLD_DEG})")
    if a.dry_run:
        print("dry run -- nothing written")


if __name__ == "__main__":
    main()
