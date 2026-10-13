#!/usr/bin/env python3
"""Visual check that a clip's GT geometry and its Chorus voxels share one frame.

Per clip, writes into <out>/<clip>/ (all in ScanNet++ world metres):
  gt_clip.ply        GT depth of the training frames (subsample 2, <=160), lifted
                     and mapped exactly as training does: lift_pixels (clip frame,
                     canonical) -> C0 @ (s * p). Orange.
  chorus_voxels.ply  every Chorus voxel centre of the scene, coloured by a 3-PCA
                     of its upcast feature.
  gt_semantic.ply    the GT points coloured by the PCA colour of the voxel
                     ChorusTargets.lookup assigns them; red = no voxel within
                     max_dist (masked out of the loss).
The ScanNet++ mesh (independent of both) is printed as a path to load beside them.
Usage: python chorus_align_viz.py <clip_dir> [<clip_dir> ...] [--out DIR] [--stride 4]
"""
from __future__ import annotations

import argparse
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "lingbot-map"))
from lingbot_map.memory import semantic as sem  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

OUT = Path("/group/compact-3dmem/campaigns/spatial_memory/_chorus_align_check")


def write_ply(path: Path, xyz: np.ndarray, rgb: np.ndarray):
    rgb = (np.clip(rgb, 0, 1) * 255).astype(np.uint8)
    rec = np.empty(len(xyz), dtype=[("x", "<f4"), ("y", "<f4"), ("z", "<f4"),
                                     ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    rec["x"], rec["y"], rec["z"] = xyz.T.astype(np.float32)
    rec["red"], rec["green"], rec["blue"] = rgb.T
    head = ("ply\nformat binary_little_endian 1.0\n"
            f"element vertex {len(xyz)}\nproperty float x\nproperty float y\nproperty float z\n"
            "property uchar red\nproperty uchar green\nproperty uchar blue\nend_header\n")
    with open(path, "wb") as f:
        f.write(head.encode()); f.write(rec.tobytes())


def pca_colours(feat: np.ndarray, fit_rows: int = 50_000, seed: int = 0):
    """feat [N, D] -> (rgb [N, 3], project) with the PCA fitted on a row sample."""
    rng = np.random.default_rng(seed)
    fit = feat[rng.choice(len(feat), min(fit_rows, len(feat)), replace=False)].astype(np.float32)
    mu, sd = fit.mean(0), fit.std(0) + 1e-6
    _, _, vt = np.linalg.svd((fit - mu) / sd, full_matrices=False)
    z_fit = ((fit - mu) / sd) @ vt[:3].T
    lo, hi = np.percentile(z_fit, 1, 0), np.percentile(z_fit, 99, 0)

    def project(f):
        z = ((f.astype(np.float32) - mu) / sd) @ vt[:3].T
        return np.clip((z - lo) / (hi - lo), 0, 1)
    return project(feat), project


def one_clip(clip_dir: Path, align: dict, out: Path, stride: int):
    name = clip_dir.name
    ct = sem.ChorusTargets(name, align)
    c = ClipCache(clip_dir)
    idx = list(range(0, len(c), 2))[:160]   # train_state.Clip: subsample 2, max_frames 160
    depth, c2w, K = c.gt_depth, c.gt_c2w, c.gt_intrinsics
    pts = []
    for i in idx:
        d = torch.from_numpy(np.ascontiguousarray(depth[i])).float()
        v, u = np.mgrid[0:d.shape[0]:stride, 0:d.shape[1]:stride]
        v, u = v.ravel(), u.ravel()
        ok = d[v, u].numpy() > 0
        pts.append(sem.lift_pixels(d, torch.from_numpy(K[i]).float(), torch.from_numpy(c2w[i]).float(),
                                   v[ok], u[ok]))
    p_clip = np.concatenate(pts)
    X = (p_clip * ct.scale) @ ct.C0[:3, :3].T + ct.C0[:3, 3]
    dist, vi = ct.tree.query(X, distance_upper_bound=ct.max_dist)
    hit = np.isfinite(dist)
    _, ok_lookup = ct.lookup(p_clip[:2000])
    assert np.array_equal(ok_lookup.numpy(), hit[:2000]), "lookup disagrees with the export transform"

    feat = ct.feat.numpy()
    vox_rgb, project = pca_colours(feat)
    sem_rgb = np.tile(np.float32([1, 0, 0]), (len(X), 1))
    sem_rgb[hit] = vox_rgb[vi[hit]]

    d_all, _ = ct.tree.query(X)
    mesh = Path(f"/data/ScanNetpp/data/{align[name]['scene']}/scans/mesh_aligned_0.05.ply")
    msg = (f"{name} ({align[name]['scene']}, {len(idx)} frames, {len(X):,} pts): "
           f"GT->voxel median {100 * np.median(d_all):.1f} cm, p90 {100 * np.percentile(d_all, 90):.1f} cm, "
           f"hit<= {100 * ct.max_dist:.0f} cm {100 * hit.mean():.1f}%")
    try:
        import open3d as o3d
        mv = np.asarray(o3d.io.read_triangle_mesh(str(mesh)).vertices)
        dm, _ = cKDTree(mv).query(X)
        msg += f"; GT->mesh-vertex median {100 * np.median(dm):.1f} cm"
    except ImportError:
        pass
    d = out / name
    d.mkdir(parents=True, exist_ok=True)
    write_ply(d / "gt_clip.ply", X, np.tile(np.float32([1, 0.55, 0]), (len(X), 1)))
    write_ply(d / "chorus_voxels.ply", ct.xyz, vox_rgb)
    write_ply(d / "gt_semantic.ply", X, sem_rgb)
    print(msg, flush=True)
    return [d / "gt_clip.ply", d / "chorus_voxels.ply", d / "gt_semantic.ply", mesh,
            mesh.with_name("mesh_aligned_0.05_semantic.ply")]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clips", nargs="+", type=Path)
    ap.add_argument("--out", type=Path, default=OUT)
    ap.add_argument("--stride", type=int, default=4, help="pixel stride per axis")
    args = ap.parse_args()
    align = sem.load_align()
    for c in args.clips:
        files = one_clip(c, align, args.out, args.stride)
        print("  " + " ".join(str(f) for f in files))


if __name__ == "__main__":
    main()
