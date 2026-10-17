#!/usr/bin/env python3
"""Readiness check of raw training sequences: completeness, pose sanity, length and loops.

Per sequence: frame counts of images / depth / poses (must agree), poses finite with
orthonormal rotations, trajectory length, median step and step outliers (teleports), and loop
content: a frame is a revisit when an earlier frame >= G metres of path before it lies within
r metres (pos), and also within 60 deg of relative rotation (view, the matchable kind).
r / G per domain: indoor 1 / 5, TartanAir-style 3 / 30, driving 10 / 200 metres.
Usage: check_training_data.py --sets tartanair-v1,tartanair-v2,tartanground --out x.json
       sets: tartanair-v1 tartanair-v2 tartanground scannetpp scannet-v2 kitti360 matrixcity
"""
from __future__ import annotations

import argparse
import json
import re
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

DOMAIN = {"indoor": (1.0, 5.0), "tartan": (3.0, 30.0), "driving": (10.0, 200.0)}


def quat_to_R(q):  # x y z w
    x, y, z, w = (q / np.linalg.norm(q, axis=1, keepdims=True)).T
    return np.stack([np.stack([1 - 2 * (y * y + z * z), 2 * (x * y - z * w), 2 * (x * z + y * w)], -1),
                     np.stack([2 * (x * y + z * w), 1 - 2 * (x * x + z * z), 2 * (y * z - x * w)], -1),
                     np.stack([2 * (x * z - y * w), 2 * (y * z + x * w), 1 - 2 * (x * x + y * y)], -1)], 1)


def loops(p, R, r, G, max_pts=4000):
    n = len(p)
    idx = np.linspace(0, n - 1, min(n, max_pts)).round().astype(int)
    p, R = p[idx], R[idx]
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(p, axis=0), axis=1))])
    pos = np.zeros(len(p), bool)
    view = np.zeros(len(p), bool)
    for a in range(0, len(p), 512):
        d = np.linalg.norm(p[a:a + 512, None] - p[None], axis=-1)
        ok = (d < r) & (s[a:a + 512, None] - s[None] >= G)
        pos[a:a + 512] = ok.any(1)
        tr = np.einsum("aij,bij->ab", R[a:a + 512], R)          # trace(Ra^T Rb)
        ang = np.degrees(np.arccos(np.clip((tr - 1) / 2, -1, 1)))
        view[a:a + 512] = (ok & (ang < 60)).any(1)
    return float(pos.mean()), float(view.mean())


def summarise(name, p, R, n_img, n_depth, domain, extra=None):
    n = len(p)
    step = np.linalg.norm(np.diff(p, axis=0), axis=1) if n > 1 else np.zeros(1)
    med = float(np.median(step)) if n > 1 else 0.0
    orth = float(np.abs(np.einsum("nji,njk->nik", R, R) - np.eye(3)).max()) if n else float("nan")
    r, G = DOMAIN[domain]
    lp, lv = loops(p, R, r, G) if n > 10 else (0.0, 0.0)
    out = {"seq": name, "n_pose": n, "n_img": n_img, "n_depth": n_depth,
           "counts_match": (n_img in (n, None)) and (n_depth in (n, None)),
           "finite": bool(np.isfinite(p).all() and np.isfinite(R).all()), "rot_orth_err": orth,
           "path_m": float(step.sum()), "step_med": med,
           "jumps": int((step > max(10 * med, 1e-6)).sum()) if n > 1 else 0,
           "revisit_pos": lp, "revisit_view": lv}
    out.update(extra or {})
    return out


def nfiles(d, pat):
    return sum(1 for _ in Path(d).glob(pat)) if Path(d).is_dir() else 0


def tartan(traj, pose_file, img_dir, depth_dir, domain="tartan"):
    a = np.loadtxt(traj / pose_file)
    return summarise(str(traj).split("/data/", 1)[-1], a[:, :3], quat_to_R(a[:, 3:7]),
                     nfiles(traj / img_dir, "*.png"), nfiles(traj / depth_dir, "*"), domain)


def scannetpp(scene):
    d = json.loads((scene / "iphone/pose_intrinsic_imu.json").read_text())
    keys = sorted(d)
    T = np.stack([np.asarray(d[k]["aligned_pose"], np.float64) for k in keys])
    return summarise(scene.name, T[:, :3, 3], T[:, :3, :3], None, None, "indoor",
                     {"has_depth": (scene / "iphone/depth.bin").exists(), "has_video": (scene / "iphone/rgb.mkv").exists()})


def kitti360(drive):
    a = np.loadtxt(drive / "cam0_to_world.txt")
    T = a[:, 1:].reshape(-1, 4, 4)
    n_img = nfiles(drive / "image_00/data_rect", "*.png")
    return summarise(drive.name, T[:, :3, 3], T[:, :3, :3], None, None, "driving",
                     {"n_img_all": n_img, "pose_frames_span": int(a[-1, 0] - a[0, 0] + 1)})


def matrixcity(road):
    d = json.loads((road / "transforms.json").read_text())
    fr = sorted(d["frames"], key=lambda f: f["frame_index"])
    M = np.asarray([f["rot_mat"] for f in fr], np.float64)
    R = M[:, :3, :3] / np.linalg.norm(M[:, :3, :3], axis=1, keepdims=True)   # columns carry a 0.01 scale
    return summarise(str(road).split("/data/", 1)[-1], M[:, :3, 3], R, nfiles(road, "*.png"), None, "driving")


def scannetv2(scene):
    a = np.loadtxt(scene / "poses.txt").reshape(-1, 4, 4)
    ok = np.isfinite(a).all((1, 2))
    r = summarise(scene.name, a[ok, :3, 3], a[ok, :3, :3], nfiles(scene / "color", "*.jpg"), nfiles(scene / "depth", "*.png"),
                  "indoor", {"invalid_poses": int((~ok).sum())})
    r["counts_match"] = r["n_img"] == r["n_depth"] == len(a)
    return r


def jobs(name):
    if name == "tartanair-v1":
        return [(tartan, (t, "pose_left.txt", "image_left", "depth_left"))
                for t in sorted(Path("/data/tartanair-v1").glob("*/*/P*"))]
    if name == "tartanair-v2":
        return [(tartan, (t, "pose_lcam_front.txt", "image_lcam_front", "depth_lcam_front"))
                for t in sorted(Path("/data/tartanair-v2").glob("*/Data_*/P*"))]
    if name == "tartanground":
        return [(tartan, (t, "pose_lcam_front.txt", "image_lcam_front", "depth_lcam_front"))
                for t in sorted(Path("/data/tartanground").glob("*/Data_*/P*"))]
    if name == "scannetpp":
        return [(scannetpp, (s,)) for s in sorted(Path("/data/ScanNetpp/data").iterdir())
                if (s / "iphone/pose_intrinsic_imu.json").exists()]
    if name == "kitti360":
        return [(kitti360, (d,)) for d in sorted(Path("/data/kitti360").glob("*_sync"))]
    if name == "scannet-v2":
        return [(scannetv2, (s,)) for s in sorted(Path("/data/scannet-v2-extracted").iterdir()) if (s / "poses.txt").exists()]
    if name == "matrixcity":
        return [(matrixcity, (r,)) for r in sorted(Path("/data/matrix_city/small_city/street").glob("*/small_city_road_*"))
                if (r / "transforms.json").exists()]
    raise ValueError(name)


def run(fa):
    f, a = fa
    try:
        return f(*a)
    except Exception as e:  # report and keep going: one broken sequence must not hide the rest
        return {"seq": str(a[0]), "error": f"{type(e).__name__}: {e}"}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--sets", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    res = {}
    for name in args.sets.split(","):
        js = jobs(name)
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(run, js, chunksize=4))
        ok = [r for r in rows if "error" not in r]
        bad = [r for r in rows if "error" in r]
        n = np.array([r["n_pose"] for r in ok]) if ok else np.zeros(1)
        summ = {"sequences": len(rows), "errors": len(bad),
                "count_mismatch": sum(not r["counts_match"] for r in ok),
                "nonfinite": sum(not r["finite"] for r in ok),
                "max_rot_orth_err": max((r["rot_orth_err"] for r in ok), default=float("nan")),
                "frames_total": int(n.sum()), "frames_median": float(np.median(n)), "frames_max": int(n.max()),
                "seq_ge_1048": int((n >= 1048).sum()), "seq_ge_320": int((n >= 320).sum()),
                "path_m_median": float(np.median([r["path_m"] for r in ok])) if ok else 0,
                "with_jumps": sum(r["jumps"] > 0 for r in ok),
                "revisit_view_mean": float(np.mean([r["revisit_view"] for r in ok])) if ok else 0,
                "seq_with_view_loops_10pct": sum(r["revisit_view"] >= 0.10 for r in ok)}
        res[name] = {"summary": summ, "per_seq": rows}
        print(f"[{name}] " + ", ".join(f"{k} {v:.3g}" if isinstance(v, float) else f"{k} {v}" for k, v in summ.items()), flush=True)
        for r in bad[:5]:
            print(f"   error {r['seq']}: {r['error']}", flush=True)
        for r in [r for r in ok if not r["counts_match"]][:5]:
            print(f"   count mismatch {r['seq']}: pose {r['n_pose']} img {r['n_img']} depth {r['n_depth']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
