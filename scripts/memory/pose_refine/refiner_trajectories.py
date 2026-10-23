#!/usr/bin/env python3
"""Example trajectories of every refiner on training and held-out (val) clips of the tap-23 packs.

Per pack dataset: the --n longest held-out clips (the trainer's longval selection) and --n training
clips of the same length. Each clip at its full length (<= 1048 frames): GT, lingbot and every refiner,
Sim3-aligned to GT, drawn from above GT's plane of motion, ATE in the legend. Arrays (.npz), plots
(.png) and an index.csv of ATEs go to OUT/<split>/<dataset>/. Test-set trajectories come from
eval_refiner_bench.py.
Usage: refiner_trajectories.py --runs RUN_DIR[,RUN_DIR...] [--n 2] [--device cpu]
"""
from __future__ import annotations

import argparse
import csv
import json
import sys
from pathlib import Path

import numpy as np
import torch

HERE = Path(__file__).resolve().parent
sys.path.insert(0, str(HERE))
import eval_refiner_bench as eb  # noqa: E402
import train_bidir_refiner as br  # noqa: E402
import train_pose_refine as tp  # noqa: E402

OUT = Path("/group/compact-3dmem/campaigns/spatial_memory/refiner_bench/_trajectories")
PACKS = ("tartanair-v1", "tartanair-v2", "tartanground", "matrixcity", "scannetpp", "scannet-v2")


def pick(ds, n):
    idx = np.load(br.PACK_ROOT / ds / "index.npz")
    mm = np.load(br.PACK_ROOT / ds / "spec23.npy", mmap_mode="r")
    bad = {b[0].rsplit("__f", 1)[0] for b in json.loads(br.GT_JUMPS.read_text()).get(ds, [])}
    val_scenes = {l.strip().split("/")[-1].split("_c")[0] for l in br.VAL24.read_text().split()}
    held, train = [], []
    for i in range(len(idx["name"])):
        name = str(idx["name"][i])
        if name.rsplit("__f", 1)[0] in bad:
            continue
        c = br.PackClip(ds, mm, idx, i)
        h = (ds == "scannetpp" and name.split("__")[0] in val_scenes) or br.longval_split(ds, c.seq)
        (held if h else train).append(c)
    key = lambda c: (-c.T, c.name)          # noqa: E731
    return {"val": sorted(held, key=key)[:n], "train": sorted(train, key=key)[:n]}


def plot(path, title, G, trajs):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    c = G.mean(0)
    axes = np.linalg.svd(G - c, full_matrices=False)[2][:2]
    fig, ax = plt.subplots(figsize=(7, 7))
    ax.plot(*((G - c) @ axes.T).T, color="k", lw=1.6, label="GT")
    for name, (P, ate) in trajs.items():
        s, R, t = tp.umeyama(P, G)
        ax.plot(*((s * P @ R.T + t - c) @ axes.T).T, lw=1.0, label=f"{name[:60]} (ATE {ate:.3g} m)")
    ax.set_aspect("equal")
    ax.set_title(title)
    ax.legend(fontsize=6, loc="best")
    fig.savefig(path, dpi=130, bbox_inches="tight")
    plt.close(fig)


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--runs", required=True)
    ap.add_argument("--n", type=int, default=2)
    ap.add_argument("--device", default="cpu")
    args = ap.parse_args()
    dev = torch.device(args.device)
    clips = {ds: pick(ds, args.n) for ds in PACKS}
    runs = [r for r in args.runs.split(",") if r]
    models = {Path(r).name: eb.load_refiner(r, dev)[0] for r in runs}
    rows = []
    for ds, splits in clips.items():
        for split, cs in splits.items():
            out = OUT / split / ds
            out.mkdir(parents=True, exist_ok=True)
            for c in cs:
                c = c.to(dev)
                G = (c.gt_c2w.double() * 1).cpu().numpy()
                G[:, :3, 3] *= c.scale
                res = {"lingbot": c.lingbot.float()}
                for name, m in models.items():
                    res[name] = m(c.get_spec(c.T)[None].float(), c.lingbot[None].float())[-1][0].float()
                trajs, arrays = {}, {"gt_c2w": G}
                for name, p in res.items():
                    P = tp.pose_enc_to_c2w(p, c.hw).double().cpu().numpy()
                    P[:, :3, 3] *= c.scale
                    ate = float(tp.traj_metrics(torch.from_numpy(P), torch.from_numpy(G), 1.0)["ate"])
                    trajs[name] = (P[:, :3, 3], ate)
                    arrays[f"c2w__{name}"] = P
                    rows.append({"split": split, "dataset": ds, "clip": c.name, "frames": c.T, "arm": name, "ate_m": ate})
                np.savez_compressed(out / f"{c.name}.npz", **arrays)
                plot(out / f"{c.name}.png", f"{split} / {ds} / {c.name} ({c.T} frames)", G[:, :3, 3], trajs)
                print(f"[traj] {split} {ds} {c.name}: " + ", ".join(f"{k[:24]} {v[1]:.3g}" for k, v in trajs.items()), flush=True)
    with open(OUT / "index.csv", "w", newline="") as f:
        w = csv.DictWriter(f, fieldnames=list(rows[0]))
        w.writeheader()
        w.writerows(rows)


if __name__ == "__main__":
    main()
