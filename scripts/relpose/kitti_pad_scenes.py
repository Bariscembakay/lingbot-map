#!/usr/bin/env python3
"""KITTI odometry 00-10 as BSS-like scenes in ABot-Recon's framing (arXiv 2608.27529): width-locked resize to
504 and an ImageNet-mean vertical pad to 280, i.e. the full field of view. Our benchmark's KITTI scenes are a
cover-fit centre crop to 504x280 instead (about 55% of the width), which is the difference this tests.
Frames come from abot_recon.preprocessing.preprocess_image itself (ABot-Recon env); GT is our benchmark
scene's gt/traj.txt, the same poses.
Usage: kitti_pad_scenes.py --kitti DIR --bench-kitti DIR --out DIR [--seqs 00 01 ...]
"""
import argparse
import os
import shutil
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np
from PIL import Image


def pad_one(task):
    src, dst = task
    from abot_recon.preprocessing import preprocess_image
    x, _ = preprocess_image(Image.open(src))
    Image.fromarray((x.permute(1, 2, 0).numpy() * 255).round().astype(np.uint8)).save(dst)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--kitti", type=Path, required=True, help="KITTI odometry root (sequences/<id>/image_2)")
    ap.add_argument("--bench-kitti", type=Path, required=True, help="our benchmark's kitti dir (<id>/gt/traj.txt)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--seqs", nargs="+", default=[f"{i:02d}" for i in range(11)])
    ap.add_argument("--workers", type=int, default=os.cpu_count())
    args = ap.parse_args()
    for seq in args.seqs:
        src = sorted((args.kitti / "sequences" / seq / "image_2").glob("*.png"))
        gt = args.out / seq / "gt"
        (gt / "rgb").mkdir(parents=True, exist_ok=True)
        n_gt = sum(1 for line in open(args.bench_kitti / seq / "gt/traj.txt") if not line.startswith("#"))
        assert n_gt == len(src), (seq, n_gt, len(src))
        shutil.copy(args.bench_kitti / seq / "gt/traj.txt", gt / "traj.txt")
        todo = [(s, gt / "rgb" / f"{i:06d}.png") for i, s in enumerate(src) if not (gt / "rgb" / f"{i:06d}.png").exists()]
        with ProcessPoolExecutor(args.workers) as ex:
            list(ex.map(pad_one, todo, chunksize=64))
        print(f"[pad] {seq}: {len(src)} frames", flush=True)


if __name__ == "__main__":
    main()
