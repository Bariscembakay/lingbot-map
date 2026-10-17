#!/usr/bin/env python3
"""Continuous camera routes through MatrixCity street data (lingbot paper, Appendix A.4.2).

Street data is stored per street: N evenly spaced positions, each shot from 5 viewpoints
(0-3 horizontal, 4 zenith), viewpoint-first (all N positions of viewpoint 0, then 1, ...).
  1. Street identification: walking a collection's posed frames in order, the first position
     within eps of one already seen ends viewpoint 0; |U| = N and the street spans 5N frames.
  2. Intersections: every street becomes its endpoint-to-endpoint XY segment, extended by
     d_ext at both ends; intersecting pairs are graph edges.
  3. Random walk: step along a street; within d_prox of an intersection switch street with
     p_turn, keeping the viewpoint whose forward axis best matches the current one; reverse at
     dead ends. Only horizontal viewpoints 0-3 are walked (zenith views carry no trajectory).
Poses are MatrixCity's rot_mat (camera-to-world, Blender/OpenGL axes, a 0.01 scale on the
rotation columns, metres); written normalised, axes unchanged.
Usage: build_matrixcity_routes.py --out routes.json [--length 1048] [--per-kframes 5]
"""
from __future__ import annotations

import argparse
import json
from pathlib import Path

import numpy as np

ROOT = Path("/data/matrix_city/small_city/street/train")


def load(coll):
    d = json.loads((ROOT / coll / "transforms.json").read_text())
    fr = sorted(d["frames"], key=lambda f: f["frame_index"])
    M = np.asarray([f["rot_mat"] for f in fr], np.float64)
    M[:, :3, :3] /= np.linalg.norm(M[:, :3, :3], axis=1, keepdims=True)
    return np.array([f["frame_index"] for f in fr]), M, float(d["camera_angle_x"])


def streets(coll, eps):
    fidx, M, fov = load(coll)
    P = M[:, :3, 3]
    out, a = [], 0
    while a < len(P):
        U = [P[a]]
        k = a + 1
        while k < len(P) and np.min(np.linalg.norm(np.asarray(U) - P[k], axis=1)) >= eps:
            U.append(P[k])
            k += 1
        N = len(U)
        if a + 5 * N > len(P):
            raise ValueError(f"{coll}: street at frame {a} needs {5 * N} frames, {len(P) - a} left")
        views = [slice(a + v * N, a + (v + 1) * N) for v in range(5)]
        # each viewpoint block must revisit the same N positions in the same order
        dev = max(np.abs(P[s] - P[views[0]]).max() for s in views)
        out.append({"coll": coll, "N": N, "fidx": [fidx[s].tolist() for s in views], "c2w": [M[s] for s in views],
                    "pos": P[views[0]], "view_dev": float(dev), "fov_x": fov})
        a += 5 * N
    return out


def seg_intersect(p1, p2, q1, q2):
    d1, d2 = p2 - p1, q2 - q1
    den = d1[0] * d2[1] - d1[1] * d2[0]
    if abs(den) < 1e-9:
        return None
    w = q1 - p1
    t = (w[0] * d2[1] - w[1] * d2[0]) / den
    u = (w[0] * d1[1] - w[1] * d1[0]) / den
    return p1 + t * d1 if 0 <= t <= 1 and 0 <= u <= 1 else None


def graph(S, d_ext):
    segs = []
    for s in S:
        a, b = s["pos"][0, :2], s["pos"][-1, :2]
        e = (b - a) / max(np.linalg.norm(b - a), 1e-9)
        segs.append((a - d_ext * e, b + d_ext * e))
    edges = {i: [] for i in range(len(S))}
    for i in range(len(S)):
        for j in range(i + 1, len(S)):
            x = seg_intersect(*segs[i], *segs[j])
            if x is None:
                continue
            ki = int(np.argmin(np.linalg.norm(S[i]["pos"][:, :2] - x, axis=1)))
            kj = int(np.argmin(np.linalg.norm(S[j]["pos"][:, :2] - x, axis=1)))
            edges[i].append((j, ki, kj))
            edges[j].append((i, kj, ki))
    return edges


def walk(S, E, T, rng, d_prox, p_turn):
    s = int(rng.integers(len(S)))
    k = int(rng.integers(S[s]["N"]))
    v = int(rng.integers(4))
    d = int(rng.choice([-1, 1]))
    route, last_switch = [], -10
    for t in range(T):
        route.append((s, v, k))
        # switch only where the two streets' nearest positions are close, so the hop is one step
        cand = [(j, kj) for j, ki, kj in E[s]
                if np.linalg.norm(S[s]["pos"][k] - S[j]["pos"][kj]) < 1.5 * d_prox]
        if cand and t - last_switch > 2 and rng.random() < p_turn:
            j, kj = cand[int(rng.integers(len(cand)))]
            f = S[s]["c2w"][v][k][:3, 2]
            v = int(np.argmax([f @ S[j]["c2w"][w][kj][:3, 2] for w in range(4)]))
            s, k, last_switch = j, kj, t
            d = int(rng.choice([-1, 1]))
        nk = k + d
        if not 0 <= nk < S[s]["N"]:
            d = -d
            nk = k + d
        k = min(max(nk, 0), S[s]["N"] - 1)
    return route


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=1048)
    ap.add_argument("--per-kframes", type=float, default=5.0, help="routes per 1000 posed frames (paper: 5)")
    ap.add_argument("--eps", type=float, default=0.01)
    ap.add_argument("--d-ext", type=float, default=10.0)
    ap.add_argument("--d-prox", type=float, default=5.0)
    ap.add_argument("--p-turn", type=float, default=0.5)
    ap.add_argument("--seed", type=int, default=0)
    args = ap.parse_args()
    S = [s for c in sorted(p.name for p in ROOT.iterdir() if (p / "transforms.json").exists()) for s in streets(c, args.eps)]
    E = graph(S, args.d_ext)
    n_frames = sum(5 * s["N"] for s in S)
    deg = np.array([len(E[i]) for i in range(len(S))])
    print(f"[mc] {len(S)} streets, {n_frames} posed frames, N median {int(np.median([s['N'] for s in S]))}, "
          f"max viewpoint position mismatch {max(s['view_dev'] for s in S):.4f} m, "
          f"{deg.sum() // 2} intersections, isolated streets {(deg == 0).sum()}", flush=True)
    rng = np.random.default_rng(args.seed)
    n_routes = max(1, int(round(n_frames / 1000 * args.per_kframes)))
    routes = []
    for _ in range(n_routes):
        r = walk(S, E, args.length, rng, args.d_prox, args.p_turn)
        routes.append({"frames": [[S[s]["coll"], int(S[s]["fidx"][v][k])] for s, v, k in r],
                       "c2w": [S[s]["c2w"][v][k].round(6).tolist() for s, v, k in r],
                       "streets_visited": len({s for s, _, _ in r})})
    P = [np.asarray(r["c2w"])[:, :3, 3] for r in routes]
    steps = np.concatenate([np.linalg.norm(np.diff(p, axis=0), axis=1) for p in P])
    print(f"[mc] {len(routes)} routes x {args.length} frames; streets per route median "
          f"{int(np.median([r['streets_visited'] for r in routes]))}; step median {np.median(steps):.2f} m, "
          f"max {steps.max():.2f} m", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"fov_x": S[0]["fov_x"], "convention": "c2w, Blender/OpenGL axes, metres",
                                    "image": "street/train/<coll>/<frame_index:04d>.png", "args": vars(args) | {"out": str(args.out)},
                                    "routes": routes}))


if __name__ == "__main__":
    main()
