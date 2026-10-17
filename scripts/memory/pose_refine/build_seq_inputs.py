#!/usr/bin/env python3
"""Pose-refiner training inputs from driving sequences lingbot was trained on (KITTI-360, VKITTI2).

Each sequence is cut into runs that start every --start-every frames and last up to
--run-len frames; each run is its own lingbot stream (its own frame 0 and anchor scale),
run exactly as the benchmark runs lingbot on KITTI: images cover-fit resized and
centre-cropped to 504x280 (benchmark/datasets/kitti.py), then the benchmark's
LingbotMapMethod with its auto keyframe interval. GT is metric and relative to the run's
first frame (gt_scale = 1), so the training loss must be scale-free.
Usage: build_seq_inputs.py kitti360 --seq 2013_05_28_drive_0000_sync --out DIR
       build_seq_inputs.py vkitti2 --seq Scene01 --out DIR
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import yaml

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "benchmark"))
from methods.lingbot_map import LingbotMapMethod  # noqa: E402  (benchmark/methods is not inside the package)
from lingbot_map.memory.losses import pose_enc_to_c2w  # noqa: E402

DATA = Path("/data")
TW, TH = 504, 280


def cover_fit(rgb, K):
    """benchmark/datasets/kitti.py::_cover_fit_center_crop, on a 3x3 K."""
    H, W = rgb.shape[:2]
    s = max(TW / W, TH / H)
    rW, rH = int(round(W * s)), int(round(H * s))
    x0, y0 = (rW - TW) // 2, (rH - TH) // 2
    out = cv2.resize(rgb, (rW, rH), interpolation=cv2.INTER_AREA)[y0:y0 + TH, x0:x0 + TW]
    K = K.copy()
    K[:2] *= s
    K[0, 2] -= x0
    K[1, 2] -= y0
    return out, K


def kitti360(seq):
    """-> list of segments, each [(image_path, c2w, K)] over consecutive posed frames."""
    root = DATA / "kitti360"
    a = np.loadtxt(root / seq / "cam0_to_world.txt")
    idx, c2w = a[:, 0].astype(int), a[:, 1:].reshape(-1, 4, 4)
    calib = {l.split(":")[0]: l.split(":")[1].split() for l in (root / "calibration/perspective.txt").read_text().splitlines() if ":" in l}
    K = np.array(calib["P_rect_00"], float).reshape(3, 4)[:, :3]
    img_dir = next(p for p in (root / "data_2d_raw" / seq / "image_00/data_rect",
                               root / seq / "image_00/data_rect") if p.exists())
    segs, cur = [], []
    have = {int(f.stem) for f in img_dir.glob("*.png")}   # drive 0002 has poses beyond its images
    for i, T in zip(idx, c2w):
        if i not in have:
            continue
        if cur and i != cur[-1][3] + 1:
            segs.append(cur)
            cur = []
        cur.append((img_dir / f"{i:010d}.png", T, K, i))
    segs.append(cur)
    return [[f[:3] for f in s] for s in segs]


def vkitti2(scene):
    segs = []
    for var in sorted(p for p in (DATA / "vkitti2" / scene).iterdir() if p.is_dir()):
        ext = np.loadtxt(var / "extrinsic.txt", skiprows=1)
        intr = np.loadtxt(var / "intrinsic.txt", skiprows=1)
        for cam in (0, 1):
            e, k = ext[ext[:, 1] == cam], intr[intr[:, 1] == cam]
            seg = []
            for row, kr in zip(e, k):
                f = int(row[0])
                w2c = row[2:].reshape(4, 4)
                K = np.array([[kr[2], 0, kr[4]], [0, kr[3], kr[5]], [0, 0, 1]])
                seg.append((var / f"frames/rgb/Camera_{cam}/rgb_{f:05d}.jpg", np.linalg.inv(w2c), K))
            segs.append((f"{var.name}_cam{cam}", seg))
    return segs


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("dataset", choices=["kitti360", "vkitti2"])
    ap.add_argument("--seq", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--run-len", type=int, default=4000)
    ap.add_argument("--start-every", type=int, default=1000)
    ap.add_argument("--min-len", type=int, default=128)
    ap.add_argument("--use-sdpa", action="store_true")
    args = ap.parse_args()
    if args.dataset == "kitti360":
        segs = [(f"seg{i}", s) for i, s in enumerate(kitti360(args.seq))]
    else:
        segs = vkitti2(args.seq)
    cfg = yaml.safe_load((REPO / "benchmark/configs/methods/lingbot_map.yaml").read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["use_sdpa"] = args.use_sdpa or kw.get("use_sdpa", False)
    method = LingbotMapMethod(**kw)
    model, psi = method.model, method.model.aggregator.patch_start_idx
    out = args.out / "specials"
    out.mkdir(parents=True, exist_ok=True)
    for sname, seg in segs:
        for s0 in range(0, max(1, len(seg) - args.min_len + 1), args.start_every):
            run = seg[s0:s0 + args.run_len]
            if len(run) < args.min_len:
                continue
            name = f"{args.seq}_{sname}_s{s0:05d}"
            f = out / f"{name}.npz"
            if f.exists():
                continue
            imgs, Ks = zip(*(cover_fit(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), K)
                             for p, _, K in run))
            images = torch.from_numpy(np.stack(imgs)).permute(0, 3, 1, 2).float().div_(255)
            chunks = []
            h = model.aggregator.register_forward_hook(
                lambda _m, _i, o: chunks.append(torch.stack(o[0], 2)[0][:, :, :psi].half().cpu()))
            try:
                pred = method._run_inference(images)
            finally:
                h.remove()
            spec = torch.cat(chunks).numpy()
            pose = pred["pose_enc"][0].float()
            assert len(spec) == len(run) == len(pose)
            G = np.stack([T for _, T, _ in run])
            gt = np.linalg.inv(G[0])[None] @ G
            # lingbot's own trajectory vs GT on this run, for spotting bad runs/conventions
            P = pose_enc_to_c2w(pose, (TH, TW)).double().numpy()
            rp, rg = np.linalg.inv(P[:-1]) @ P[1:], np.linalg.inv(gt[:-1]) @ gt[1:]
            m = np.einsum("nji,njk->nik", rp[:, :3, :3], rg[:, :3, :3])
            rot_res = float(np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean())
            tmp = f.with_suffix(f".tmp{os.getpid()}.npz")  # concurrent copies of a job must not share it
            np.savez(tmp, specials=spec, pose_enc=pose.numpy(), gt_c2w=gt.astype(np.float32),
                     K=np.stack(Ks).astype(np.float32), hw=np.int32([TH, TW]), gt_scale=np.float64(1.0),
                     source=f"{args.dataset}/{args.seq}/{sname}@{s0}", rel_rot_residual_deg=np.float32(rot_res))
            tmp.rename(f)
            print(f"[seq] {name}: {len(run)} frames, lingbot-vs-GT relative rotation residual "
                  f"{rot_res:.3f} deg/frame", flush=True)


if __name__ == "__main__":
    main()
