#!/usr/bin/env python3
"""How much pose error does lingbot leave on its own training sets, by prefix length? Reads the cached
streaming poses of the 1048-frame cache clips and scores the first t frames against GT
(Sim3 ATE, consecutive RPE, AUC@3), the same metrics as the refiners' "lingbot" baseline. Each t
samples its own clips among those at least t frames long; --exclude drops clips with broken GT.
Usage: lingbot_error_vs_length.py --out result.json [--per-dataset 60] [--exclude gt_jump_scan.json]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402

DATASETS = ["tartanair-v1", "tartanair-v2", "tartanground", "matrixcity", "kitti360", "scannetpp", "scannet-v2"]


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--per-dataset", type=int, default=60)
    ap.add_argument("--lengths", default="64,160,320,640,1048")
    ap.add_argument("--cache-root", type=Path, default=Path("/data"))
    ap.add_argument("--exclude", type=Path, help="json {dataset: [[clip npz name, ...], ...]} of clips with broken GT")
    args = ap.parse_args()
    ts = [int(x) for x in args.lengths.split(",")]
    rng = np.random.default_rng(0)
    res = {}
    for ds in DATASETS:
        root = args.cache_root / f"lingbot-cache1048-{ds}"
        if not root.exists():
            root = Path(f"/group/compact-3dmem/datasets/_staging/lingbot-cache1048-{ds}")
        bad = {b[0] for b in json.loads(args.exclude.read_text()).get(ds, [])} if args.exclude else set()
        fs = [f for f in sorted(root.glob("*.npz")) if f.name not in bad]
        n_of = {f: int(f.stem.rsplit("_n", 1)[1]) for f in fs}
        rows = {t: [] for t in ts}
        for t in ts:    # every t scores its own sample of the clips at least t frames long
            ok = [f for f in fs if n_of[f] >= t]
            pick = [ok[i] for i in rng.choice(len(ok), min(args.per_dataset, len(ok)), replace=False)] if ok else []
            for f in pick:
                z = np.load(f)
                hw = tuple(int(x) for x in z["hw"])
                P = tp.pose_enc_to_c2w(torch.from_numpy(z["pose_enc"]).float(), hw).double()
                G = torch.from_numpy(z["gt_c2w"]).double()
                scale = float(z["gt_scale"])
                m = tp.traj_metrics(P[:t], G[:t], scale)
                path = float(np.linalg.norm(np.diff(G[:t, :3, 3].numpy(), axis=0), axis=1).sum()) * scale
                m["path_m"] = path
                m["ate_pct_path"] = 100 * m["ate"] / max(path, 1e-6)
                rows[t].append(m)
        res[ds] = {"clips_used": len(fs), "excluded_bad_gt": len(bad),
                   "by_t": {str(t): {"n": len(rows[t]), **{k: float(np.median([r[k] for r in rows[t]])) if rows[t] else None
                                     for k in ("ate", "ate_pct_path", "rpe_r", "auc3", "path_m")}} for t in ts}}
        print(ds, json.dumps(res[ds]), flush=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
