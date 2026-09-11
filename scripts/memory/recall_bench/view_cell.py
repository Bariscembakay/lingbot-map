#!/usr/bin/env python3
"""Open one nrgbd_recall_s2 cell in viser.

  python view_cell.py --method ttt3r --scene staircase --tier 300
  python view_cell.py --method ours  --scene staircase --tier 300 --show err

Clouds are served at full resolution (an n500 fused cloud is ~30M points, so
give the browser a moment). --stride decimates client-side only if you ask; the
files on disk are always the full record.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("/group/compact-3dmem/campaigns/spatial_memory/recallbench")
# "recalled" = queried with a camera, image withheld (what the benchmark scores)
# "predicted" = the same model with the image visible (upper bound)
LAYERS = {"recalled": ["recalled_cloud_rgb.ply", "pred_cloud_rgb.ply"],
          "err": ["recalled_cloud_err.ply", "pred_cloud_err.ply"],
          "predicted": ["predicted_cloud_rgb.ply", "ingest_cloud_rgb.ply"],
          "predicted_err": ["predicted_cloud_err.ply", "ingest_cloud_err.ply"],
          "gt": ["gt_cloud_rgb.ply"]}


def read_ply(p: Path):
    """Minimal binary-little-endian xyz+rgb reader (what write_ply emits)."""
    with open(p, "rb") as f:
        n, names = 0, []
        while True:
            line = f.readline().decode("ascii", "replace").strip()
            if line.startswith("element vertex"):
                n = int(line.split()[-1])
            elif line.startswith("property"):
                names.append(line.split()[-1])
            elif line == "end_header":
                break
        has_rgb = "red" in names
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if has_rgb:
            dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        a = np.frombuffer(f.read(n * np.dtype(dt).itemsize), dtype=dt, count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], -1).astype(np.float32)
    rgb = (np.stack([a["red"], a["green"], a["blue"]], -1)
           if has_rgb else np.full((n, 3), 200, np.uint8))
    return xyz, rgb


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--method", required=True)
    ap.add_argument("--scene", required=True)
    ap.add_argument("--tier", type=int, required=True)
    ap.add_argument("--show", nargs="+", default=["recalled", "gt"],
                    choices=list(LAYERS))
    ap.add_argument("--stride", type=int, default=1,
                    help="client-side decimation; files stay full-resolution")
    ap.add_argument("--point-size", type=float, default=0.006)
    ap.add_argument("--port", type=int, default=8080)
    ap.add_argument("--root", type=Path, default=ROOT)
    args = ap.parse_args()

    import viser

    cell = args.root / args.method / f"{args.scene}_n{args.tier}"
    if not cell.exists():
        print(f"no such cell: {cell}")
        return 1

    srv = viser.ViserServer(port=args.port)
    for name in args.show:
        p = next((cell / c for c in LAYERS[name] if (cell / c).exists()),
                 cell / LAYERS[name][0])
        if not p.exists():
            print(f"  (missing {name}: {p.name})")
            continue
        xyz, rgb = read_ply(p)
        xyz, rgb = xyz[::args.stride], rgb[::args.stride]
        srv.scene.add_point_cloud(f"/{name}", points=xyz, colors=rgb,
                                  point_size=args.point_size)
        print(f"  {name:4s} {len(xyz):>9,d} pts  {p.name}")
    print(f"\nviser on port {args.port} "
          f"(ssh -L {args.port}:localhost:{args.port} <node>)")
    while True:
        __import__("time").sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
