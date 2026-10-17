"""Two more pose-headroom arms (see gtpose_oracle_build.py), both with GT poses:
  gtpose_framescale -- lingbot depth times a per-frame scale, median(GT/pred) over
                       valid pixels, so only the depth's per-frame scale is fixed
                       (shape and intrinsics stay lingbot's). Depth is then metric,
                       so GT translations are used as-is.
  gtpose_gtintr     -- lingbot depth unchanged, GT intrinsics instead of lingbot's
                       (whose focal is ~9% off on NRGBD), GT translations / s as
                       in gtpose_oracle.
Lingbot saves depth on the GT pixel grid, so the ratio is per pixel.
Usage (from benchmark/, PYTHONPATH=.): python gtpose_framescale_build.py <bench>
"""
import os, sys
from pathlib import Path
import numpy as np
from benchmark.io.image import load_exr, save_exr

SRC = Path("/group/compact-3dmem/campaigns/reproduction/default")
DST = Path("/group/compact-3dmem/campaigns/pose_oracle")
ORACLE = DST / "gtpose_oracle"
bench = sys.argv[1]


def gt_traj_metric(scene):
    return SRC / bench / scene / "gt/traj.txt"


for scene in sorted(p.name for p in (SRC / bench).iterdir()
                    if p.is_dir() and not p.name.startswith("_") and p.name != "eval"):
    s_src = SRC / bench / scene
    lm = s_src / "lingbot_map"

    d = DST / "gtpose_gtintr" / bench / scene
    m = d / "lingbot_map"
    m.mkdir(parents=True, exist_ok=True)
    for name, src in (("gt", s_src / "gt"),):
        if not (d / name).exists():
            os.symlink(src, d / name)
    for name, src in (("depth", lm / "depth"), ("confidence", lm / "confidence"), ("rgb", lm / "rgb"),
                      ("intrinsics.txt", s_src / "gt/intrinsics.txt"), (".complete.json", lm / ".complete.json"),
                      ("traj.txt", ORACLE / bench / scene / "lingbot_map/traj.txt")):
        if not (m / name).exists():
            os.symlink(src, m / name)

    d = DST / "gtpose_framescale" / bench / scene
    m = d / "lingbot_map"
    (m / "depth").mkdir(parents=True, exist_ok=True)
    if not (d / "gt").exists():
        os.symlink(s_src / "gt", d / "gt")
    for name, src in (("confidence", lm / "confidence"), ("rgb", lm / "rgb"),
                      ("intrinsics.txt", lm / "intrinsics.txt"), ("traj.txt", gt_traj_metric(scene))):
        if not (m / name).exists():
            os.symlink(src, m / name)
    ks = []
    for f in sorted((lm / "depth").glob("*.exr")):
        out = m / "depth" / f.name
        p = load_exr(f).astype(np.float32)
        g = load_exr(s_src / "gt/depth" / f.name).astype(np.float32)
        p2, g2 = p.squeeze(), g.squeeze()
        v = (g2 > 0) & (p2 > 0) & np.isfinite(p2) & np.isfinite(g2)
        k = float(np.median(g2[v] / p2[v])) if v.sum() > 100 else np.nan
        ks.append(k)
        if not out.exists():
            save_exr(p * (k if np.isfinite(k) else np.nanmedian(ks)), out)
    if not (m / ".complete.json").exists():
        os.symlink(lm / ".complete.json", m / ".complete.json")
    ks = np.array(ks)
    print(f"{bench}/{scene}: {len(ks)} frames, per-frame scale median {np.nanmedian(ks):.3f}, "
          f"CV {np.nanstd(ks) / np.nanmedian(ks) * 100:.1f}%", flush=True)
