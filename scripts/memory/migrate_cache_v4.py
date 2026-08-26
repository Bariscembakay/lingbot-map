#!/usr/bin/env python3
"""Migrate a v3 clip cache to v4 by adding `gt_intrinsics.npy`.

v4 exists because the recall probe needs GT intrinsics: to build the query
raymap, and to unproject GT depth into the pointmap it is scored against.
Nothing else changed, and every v3 array is left untouched -- so this is a
migration rather than a rebuild, which matters because a rebuild costs a full
aggregator pass over the clip.

The intrinsics were always derivable: `meta.json` records `scene`, `frame_ids`
and the post-resize `height`/`width`, and `gt.prepare` already computed the
scaled matrix. Only `build_cache.py` never wrote it out.

Usage:
    migrate_cache_v4.py <clip_dir> [<clip_dir> ...] [--scannetpp-root PATH]
                        [--dry-run]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory.cache_format import FORMAT_VERSION, GT_INTRINSICS, META
from lingbot_map.memory.gt import read_iphone_meta, scale_intrinsics


def migrate(clip_dir: Path, scannetpp_root: Path, dry_run: bool) -> str:
    meta_path = clip_dir / META
    if not meta_path.exists():
        return f"SKIP  {clip_dir}: no {META}"
    meta = json.loads(meta_path.read_text())
    have = meta.get("format_version")
    if have == FORMAT_VERSION and (clip_dir / GT_INTRINSICS).exists():
        return f"OK    {clip_dir}: already v{FORMAT_VERSION}"
    if have != 3:
        return f"SKIP  {clip_dir}: format_version {have}, expected 3"

    jp = scannetpp_root / "data" / meta["scene"] / "iphone" / "pose_intrinsic_imu.json"
    if not jp.exists():
        return f"FAIL  {clip_dir}: {jp} missing"

    _, intr_full = read_iphone_meta(jp, meta["frame_ids"])
    # (1440, 1920) is the iPhone capture resolution the recorded intrinsics are
    # expressed in; gt.prepare scales from the same constant.
    intr = scale_intrinsics(
        intr_full, (1440, 1920), (meta["height"], meta["width"])
    ).astype(np.float32)

    n = meta["num_frames"]
    if intr.shape != (n, 3, 3):
        return f"FAIL  {clip_dir}: intrinsics {intr.shape}, expected {(n, 3, 3)}"

    if dry_run:
        return f"WOULD {clip_dir}: write {intr.shape}, bump 3 -> {FORMAT_VERSION}"

    np.save(clip_dir / GT_INTRINSICS, intr)
    # Bump the version only after the array is on disk, so an interrupted run
    # leaves a readable v3 clip rather than a v4 clip with nothing to read.
    meta["format_version"] = FORMAT_VERSION
    meta_path.write_text(json.dumps(meta, indent=2))
    return f"DONE  {clip_dir}: +{GT_INTRINSICS} {intr.shape}"


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--scannetpp-root", type=Path, default=Path("/data/ScanNetpp"))
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    bad = 0
    for clip in args.clips:
        line = migrate(clip, args.scannetpp_root, args.dry_run)
        print(line)
        bad += line.startswith("FAIL")
    return 1 if bad else 0


if __name__ == "__main__":
    raise SystemExit(main())
