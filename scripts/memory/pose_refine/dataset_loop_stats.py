#!/usr/bin/env python3
"""Per training dataset: trajectories, environments, length range (raw frames), path length and how many
trajectories revisit a place. A revisit = two frames within R metres, viewing directions within 45 deg
(or any direction, counted separately), with at least L metres driven between them.
Usage: dataset_loop_stats.py --out stats.json [--datasets a,b] [--workers 16]
"""
from __future__ import annotations

import argparse
import json
import sys
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path(__file__).resolve().parent))
sys.path.insert(0, str(Path(__file__).resolve().parents[3]))
from clip_sources import source  # noqa: E402

# (R metres, L metres): room-scale vs street-scale revisits
THRESH = {"scannet-v2": (0.5, 5.0), "scannetpp": (0.5, 5.0), "tartanair-v1": (2.0, 20.0),
          "tartanair-v2": (2.0, 20.0), "tartanground": (2.0, 20.0), "matrixcity": (5.0, 50.0),
          "kitti360": (5.0, 50.0)}
_src = {}


def seq_stats(job):
    ds, seq = job
    if ds not in _src:
        _src[ds] = source(ds)
    try:
        T = _src[ds].poses(seq)
    except Exception as e:
        return {"seq": seq, "error": f"{type(e).__name__}: {e}"[:200]}
    n = len(T)
    ok = np.isfinite(T).all((1, 2))
    P, F = T[ok, :3, 3], T[ok, :3, 2]
    out = {"seq": seq, "frames": int(n), "valid": int(ok.sum())}
    if len(P) < 2:
        return out | {"path_m": 0.0, "loop": False, "loop_anydir": False}
    s = np.concatenate([[0], np.cumsum(np.linalg.norm(np.diff(P, axis=0), axis=1))])
    R, L = THRESH[ds]
    kf, last = [0], 0.0
    for i in range(1, len(P)):
        if s[i] - last >= R / 2:
            kf.append(i)
            last = s[i]
    kf = np.array(kf)
    pairs = cKDTree(P[kf]).query_pairs(R, output_type="ndarray")
    loop = anydir = False
    if len(pairs):
        a, b = kf[pairs[:, 0]], kf[pairs[:, 1]]
        far = np.abs(s[a] - s[b]) >= L
        anydir = bool(far.any())
        loop = bool((far & ((F[a] * F[b]).sum(1) >= np.cos(np.radians(45)))).any())
    return out | {"path_m": float(s[-1]), "loop": loop, "loop_anydir": anydir}


def env_of(ds, seq):
    if ds in ("tartanair-v1", "tartanair-v2", "tartanground"):
        return seq.split("/")[0]
    if ds == "kitti360":
        return seq.split("/")[0]
    return seq


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", required=True)
    ap.add_argument("--datasets", default=",".join(THRESH))
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    res = {}
    for ds in args.datasets.split(","):
        seqs = source(ds).seqs()
        with ProcessPoolExecutor(args.workers) as ex:
            rows = list(ex.map(seq_stats, [(ds, q) for q in seqs], chunksize=4))
        good = [r for r in rows if "error" not in r and r["valid"] >= 2]
        fr = np.array([r["frames"] for r in good])
        pm = np.array([r["path_m"] for r in good])
        summ = {"trajectories": len(good), "errors": len(rows) - len(good),
                "environments": len({env_of(ds, r["seq"]) for r in good}),
                "frames_min": int(fr.min()), "frames_median": int(np.median(fr)), "frames_max": int(fr.max()),
                "frames_total": int(fr.sum()),
                "path_m_min": round(float(pm.min()), 1), "path_m_median": round(float(np.median(pm)), 1),
                "path_m_max": round(float(pm.max()), 1),
                "loop_trajectories": sum(r["loop"] for r in good),
                "loop_anydir_trajectories": sum(r["loop_anydir"] for r in good),
                "thresh_R_m_L_m": THRESH[ds]}
        print(ds, json.dumps(summ), flush=True)
        res[ds] = {"summary": summ, "per_seq": rows}
        Path(args.out).write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
