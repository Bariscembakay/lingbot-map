#!/usr/bin/env python3
"""ABot-Recon (arXiv 2608.27529) on our benchmark scenes, scored exactly as eval_lingbot_rel.py scores our arms.

run   (ABot-Recon env): stream a BSS scene's gt/rgb frames (already 504x280, ABot's input size) through the
      released model with its loop closure on; saves <out>/<bench>_<scene>.npz with the camera-to-world
      poses before (noloop) and after (loop) the pose graph.
score (lingbot env): Sim3 ATE, consecutive-frame RPE and AUC of both against gt/traj.txt, written as
      <bench>_<scene>.json next to the npz; --summary prints the table.
Usage: abot_bench.py run --scenes <BSS scene dir>... --out DIR --abot-root DIR
       abot_bench.py score --scenes <BSS scene dir>... --out DIR
       abot_bench.py score --summary --out DIR
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np


def load_bss_traj(p):
    a = np.loadtxt(p, comments="#")
    c2w = np.tile(np.eye(4), (len(a), 1, 1))
    c2w[:, :3] = a[:, 1:].reshape(-1, 3, 4)
    return c2w


def run(args):
    import torch
    from abot_recon.api import ABotRecon

    root = args.abot_root
    model = ABotRecon.from_pretrained(root / "checkpoints/abot_recon.safetensors", local_files_only=True,
                                      loop_closure=True, loop_salad_checkpoint=root / "checkpoints/loop/dino_salad.ckpt",
                                      loop_dino_checkpoint=root / "checkpoints/loop/dinov2_vitb14_pretrain.pth",
                                      loop_output_dir=args.out / "_loop", output_local_points=False,
                                      output_confidence=False)
    args.out.mkdir(parents=True, exist_ok=True)
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}"
        paths = sorted((scene / "gt/rgb").glob("*.png"))
        torch.cuda.reset_peak_memory_stats()
        t0 = time.time()
        res = model.infer(paths, output_local_points=False, output_world_points=False, output_confidence=False,
                          loop_closure=True)
        dt = time.time() - t0
        model.reset()
        noloop = res.camera_poses_noloop.float().cpu().numpy()
        loop = res.camera_poses_loop.float().cpu().numpy() if res.camera_poses_loop is not None else noloop
        tmp = args.out / f".{name}.{os.getpid()}.npz"
        np.savez(tmp, noloop=noloop, loop=loop, seconds=np.float32(dt),
                 peak_gib=np.float32(torch.cuda.max_memory_allocated() / 2**30))
        os.replace(tmp, args.out / f"{name}.npz")
        print(f"[abot] {name}: {len(paths)} frames in {dt:.0f}s ({len(paths) / dt:.1f} fps), "
              f"peak {torch.cuda.max_memory_allocated() / 2**30:.1f} GiB", flush=True)


def score(args):
    if args.summary:
        rows = [json.loads(f.read_text()) for f in sorted(args.out.glob("*.json"))]
        for bench in sorted({r["bench"] for r in rows}):
            rs = [r for r in rows if r["bench"] == bench]
            for r in rs:
                print(f"{bench:>6} {r['scene']:>18} {r['frames']:>6} | ATE noloop {r['noloop']['ate']:8.3f} "
                      f"loop {r['loop']['ate']:8.3f} | rpe_r {r['noloop']['rpe_r']:.3f} / {r['loop']['rpe_r']:.3f}")
            print(f"{bench:>6} {'mean (' + str(len(rs)) + ')':>18} {'':>6} | ATE noloop "
                  f"{np.mean([r['noloop']['ate'] for r in rs]):8.3f} loop {np.mean([r['loop']['ate'] for r in rs]):8.3f}")
        return
    sys.path.insert(0, str(Path(__file__).resolve().parents[1] / "memory/pose_refine"))
    import torch
    import train_pose_refine as tp
    for scene in args.scenes:
        name = f"{scene.parent.name}_{scene.name}"
        z = np.load(args.out / f"{name}.npz")
        G = torch.from_numpy(load_bss_traj(scene / "gt/traj.txt"))
        rec = {"bench": scene.parent.name, "scene": scene.name, "frames": len(G),
               "seconds": float(z["seconds"]), "peak_gib": float(z["peak_gib"])}
        for k in ("noloop", "loop"):
            P = torch.from_numpy(z[k].astype(np.float64))
            assert len(P) == len(G), (len(P), len(G))
            rec[k] = {m: float(v) for m, v in tp.traj_metrics(P, G, 1.0).items()}
        (args.out / f"{name}.json").write_text(json.dumps(rec, indent=1))
        print(f"[score] {name}: ATE noloop {rec['noloop']['ate']:.3f} loop {rec['loop']['ate']:.3f}", flush=True)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("cmd", choices=["run", "score"])
    ap.add_argument("--scenes", nargs="+", type=Path, default=[])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--abot-root", type=Path)
    ap.add_argument("--summary", action="store_true")
    args = ap.parse_args()
    run(args) if args.cmd == "run" else score(args)


if __name__ == "__main__":
    main()
