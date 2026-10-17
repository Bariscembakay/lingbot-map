#!/usr/bin/env python3
"""Extract ScanNet v2 .sens files (format v4, ScanNet's SensorData.py) to frames.

Per scan, under OUT/<scene>/:
  color/<i>.jpg     the stored JPEG bytes, unchanged (1296x968)
  depth/<i>.png     16-bit PNG of the stored uint16 depth (zlib-decompressed), millimetres (640x480)
  poses.txt         one camera-to-world 4x4 per line (16 numbers, row-major); invalid poses
                    (ScanNet marks them -inf) are kept as written
  intrinsics.json   color/depth intrinsics and extrinsics, image sizes, depth shift
A scan is skipped when OUT/<scene>/poses.txt exists (written last), so reruns resume.
Usage: extract_scannet_sens.py --root /data/scannet-v2/scans --out DIR [--workers 48]
"""
from __future__ import annotations

import argparse
import json
import struct
import zlib
from multiprocessing import Pool
from pathlib import Path

import numpy as np
from PIL import Image


def extract(args):
    sens, out = args
    scene = sens.stem
    d = out / scene
    if (d / "poses.txt").exists():
        return scene, -1
    (d / "color").mkdir(parents=True, exist_ok=True)
    (d / "depth").mkdir(parents=True, exist_ok=True)
    with open(sens, "rb") as f:
        version = struct.unpack("I", f.read(4))[0]
        assert version == 4, version
        n = struct.unpack("Q", f.read(8))[0]
        f.read(n)                                           # sensor name
        mats = [np.frombuffer(f.read(64), np.float32).reshape(4, 4) for _ in range(4)]
        ccomp, dcomp = struct.unpack("ii", f.read(8))
        cw, ch, dw, dh = struct.unpack("IIII", f.read(16))
        shift = struct.unpack("f", f.read(4))[0]
        nf = struct.unpack("Q", f.read(8))[0]
        assert ccomp == 2 and dcomp == 1, (ccomp, dcomp)    # jpeg colour, zlib uint16 depth
        poses = np.empty((nf, 16), np.float32)
        for i in range(nf):
            poses[i] = np.frombuffer(f.read(64), np.float32)
            _, _, cs, ds = struct.unpack("QQQQ", f.read(32))
            (d / "color" / f"{i:06d}.jpg").write_bytes(f.read(cs))
            dep = np.frombuffer(zlib.decompress(f.read(ds)), np.uint16).reshape(dh, dw)
            Image.fromarray(dep).save(d / "depth" / f"{i:06d}.png")
    (d / "intrinsics.json").write_text(json.dumps({
        "intrinsic_color": mats[0].tolist(), "extrinsic_color": mats[1].tolist(),
        "intrinsic_depth": mats[2].tolist(), "extrinsic_depth": mats[3].tolist(),
        "color_wh": [cw, ch], "depth_wh": [dw, dh], "depth_shift": shift, "frames": int(nf)}))
    np.savetxt(d / "poses.txt", poses, fmt="%.7g")
    return scene, int(nf)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--root", type=Path, default=Path("/data/scannet-v2/scans"))
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--workers", type=int, default=48)
    args = ap.parse_args()
    jobs = [(s / f"{s.name}.sens", args.out) for s in sorted(args.root.iterdir()) if (s / f"{s.name}.sens").exists()]
    print(f"[sens] {len(jobs)} scans", flush=True)
    done = 0
    with Pool(args.workers) as p:
        for scene, nf in p.imap_unordered(extract, jobs):
            done += 1
            if done % 50 == 0 or done == len(jobs):
                print(f"[sens] {done}/{len(jobs)} (last {scene}: {nf} frames)", flush=True)


if __name__ == "__main__":
    main()
