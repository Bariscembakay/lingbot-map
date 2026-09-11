#!/usr/bin/env python3
"""Browser viewer for nrgbd_recall_s2: pick method / tier / scene, hit load.

  "$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python" \
      scripts/memory/recall_bench/view_recall.py --port 9999
  # then from your laptop:  ssh -L 9999:localhost:9999 baris_bakay@<node>

Layers (a cell has whichever exist):
  recalled       queried with a camera, IMAGE WITHHELD -- what the table scores
  predicted      same model WITH the image -- the upper bound
  recalled_err   per-point distance to GT, 5th-95th percentile
  predicted_err  same scale, for the predicted cloud
  gt             ground truth

Colours in the rgb layers are ground-truth image pixels, never predictions --
a natural-looking cloud is not evidence of good recall. Use the *_err layers
to judge quality, where colour IS the measurement.
"""
from __future__ import annotations

import argparse
from pathlib import Path

import numpy as np

ROOT = Path("/group/compact-3dmem/campaigns/spatial_memory/recallbench")
# legacy names kept so a part-renamed tree still loads
LAYERS = {
    "recalled": ["recalled_cloud_rgb.ply", "pred_cloud_rgb.ply"],
    "predicted": ["predicted_cloud_rgb.ply", "ingest_cloud_rgb.ply"],
    "recalled_err": ["recalled_cloud_err.ply", "pred_cloud_err.ply"],
    "predicted_err": ["predicted_cloud_err.ply", "ingest_cloud_err.ply"],
    "gt": ["gt_cloud_rgb.ply"],
}


def read_ply(p: Path):
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
        dt = [("x", "<f4"), ("y", "<f4"), ("z", "<f4")]
        if "red" in names:
            dt += [("red", "u1"), ("green", "u1"), ("blue", "u1")]
        a = np.frombuffer(f.read(n * np.dtype(dt).itemsize), dtype=dt, count=n)
    xyz = np.stack([a["x"], a["y"], a["z"]], -1).astype(np.float32)
    rgb = (np.stack([a["red"], a["green"], a["blue"]], -1)
           if "red" in names else np.full((len(xyz), 3), 200, np.uint8))
    return xyz, rgb


def discover(root: Path):
    """{method: {tier: {scene: cell_dir}}} over whatever is on disk."""
    out: dict = {}
    if not root.is_dir():
        return out
    for m in sorted(p.name for p in root.iterdir() if p.is_dir()):
        for cell in sorted((root / m).iterdir()):
            if not cell.is_dir() or "_n" not in cell.name:
                continue
            scene, tier = cell.name.rsplit("_n", 1)
            if not any((cell / c).exists() for cs in LAYERS.values() for c in cs):
                continue
            out.setdefault(m, {}).setdefault(tier, {})[scene] = cell
    return out


def layer_path(cell: Path, layer: str):
    return next((cell / c for c in LAYERS[layer] if (cell / c).exists()), None)


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=ROOT)
    ap.add_argument("--port", type=int, default=9999)
    ap.add_argument("--max-points", type=int, default=2_000_000,
                    help="per cloud; files on disk stay full-resolution")
    args = ap.parse_args()

    import viser

    tree = discover(args.root)
    if not tree:
        print(f"nothing found under {args.root}")
        return 1
    methods = sorted(tree)
    print(f"[view_recall] {args.root}")
    for m in methods:
        for t in sorted(tree[m], key=int):
            print(f"   {m:<8} n{t:<5} {len(tree[m][t])} scenes")

    srv = viser.ViserServer(port=args.port)
    loaded: dict = {}

    with srv.gui.add_folder("pick a cell"):
        d_method = srv.gui.add_dropdown("method", methods, initial_value=methods[0])
        tiers0 = sorted(tree[methods[0]], key=int)
        d_tier = srv.gui.add_dropdown("tier (n)", tiers0, initial_value=tiers0[0])
        scenes0 = sorted(tree[methods[0]][tiers0[0]])
        d_scene = srv.gui.add_dropdown("scene", scenes0, initial_value=scenes0[0])
        c_layers = {k: srv.gui.add_checkbox(k, initial_value=(k == "recalled"))
                    for k in LAYERS}
        b_load = srv.gui.add_button("load")
        b_all = srv.gui.add_button("load this scene+tier for ALL methods")
        b_clear = srv.gui.add_button("clear all")
    s_size = srv.gui.add_slider("point size", min=0.001, max=0.05, step=0.001,
                                initial_value=0.006)

    def refresh_tier():
        ts = sorted(tree.get(d_method.value, {}), key=int)
        if ts:
            d_tier.options = ts
            if d_tier.value not in ts:
                d_tier.value = ts[0]

    def refresh_scene():
        ss = sorted(tree.get(d_method.value, {}).get(d_tier.value, {}))
        if ss:
            d_scene.options = ss
            if d_scene.value not in ss:
                d_scene.value = ss[0]

    @d_method.on_update
    def _(_) -> None:
        refresh_tier(); refresh_scene()

    @d_tier.on_update
    def _(_) -> None:
        refresh_scene()

    def add(method: str, tier: str, scene: str, layer: str) -> None:
        cell = tree.get(method, {}).get(tier, {}).get(scene)
        if cell is None:
            return
        p = layer_path(cell, layer)
        if p is None:
            print(f"   (no {layer} for {method}/{scene}_n{tier})")
            return
        key = f"{method}/{scene}_n{tier}/{layer}"
        if key in loaded:
            return
        xyz, rgb = read_ply(p)
        if len(xyz) > args.max_points:                 # view-only decimation
            step = int(np.ceil(len(xyz) / args.max_points))
            xyz, rgb = xyz[::step], rgb[::step]
        h = srv.scene.add_point_cloud("/" + key.replace("/", "_"), points=xyz,
                                      colors=rgb, point_size=s_size.value)
        with srv.gui.add_folder(key) as folder:
            cb = srv.gui.add_checkbox("visible", initial_value=True)
            rm = srv.gui.add_button("remove")

        @cb.on_update
        def _(_, h=h, cb=cb) -> None:
            h.visible = cb.value

        @rm.on_click
        def _(_, h=h, key=key, folder=folder) -> None:
            h.remove(); folder.remove(); loaded.pop(key, None)

        loaded[key] = h
        print(f"   + {key}  {len(xyz):,} pts")

    @b_load.on_click
    def _(_) -> None:
        for k, cb in c_layers.items():
            if cb.value:
                add(d_method.value, d_tier.value, d_scene.value, k)

    @b_all.on_click
    def _(_) -> None:
        for m in methods:
            for k, cb in c_layers.items():
                if cb.value:
                    add(m, d_tier.value, d_scene.value, k)

    @b_clear.on_click
    def _(_) -> None:
        for h in list(loaded.values()):
            h.remove()
        loaded.clear()
        print("   cleared")

    @s_size.on_update
    def _(_) -> None:
        for h in loaded.values():
            h.point_size = s_size.value

    print(f"\nviser on port {args.port}: "
          f"ssh -L {args.port}:localhost:{args.port} $USER@<node>")
    import time
    while True:
        time.sleep(10)


if __name__ == "__main__":
    raise SystemExit(main())
