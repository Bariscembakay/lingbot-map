#!/usr/bin/env python3
"""Open-vocabulary segmentation of one cached frame, read out of the Chorus voxels.

Every pixel's ray is cut at its GT depth with the frame's GT pose (clip frame,
canonical units), mapped to ScanNet++ world metres with C0 @ (s * p) and given
the voxel ChorusTargets.lookup assigns it -- the exact path the semantic
training labels take. The voxel's Chorus lang feature (SigLIP2 space) is
classified against the ScanNet++ top-100 SigLIP2 text embeddings.

Writes <out>/<clip>/frame_<i>/:
  seg.png        RGB | voxel-read segmentation | mesh-annotation GT | overlay
  voxels.ply     the voxels this frame's rays hit, coloured by predicted class
  hit_points.ply the ray hit points in world metres, coloured by the frame's RGB
Usage: python chorus_frame_seg.py <clip_dir> <cache_frame> [<cache_frame> ...]
"""
from __future__ import annotations

import argparse
import csv
import json
import subprocess
import sys
from pathlib import Path

import numpy as np
import torch
from scipy.spatial import cKDTree

sys.path.insert(0, str(Path.home() / "lingbot-map"))
sys.path.insert(0, str(Path(__file__).parent))
from lingbot_map.memory import semantic as sem  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402
from chorus_align_viz import OUT, write_ply  # noqa: E402

BENCH = Path.home() / "Chorus/pointcept/datasets/preprocessing/scannetpp/metadata/semantic_benchmark"


def top100():
    classes = [c for c in (BENCH / "top100.txt").read_text().split("\n") if c]
    remap = {r["class"]: r["semantic_map_to"] or r["class"] for r in csv.DictReader(open(BENCH / "map_benchmark.csv"))}
    text = torch.load(BENCH / "top100_text_embeddings_siglip2_so400m.pt", map_location="cpu").float().numpy()
    pal = np.random.default_rng(3).uniform(0.15, 1.0, (len(classes), 3))
    pal[:3] = [[.75, .75, .75], [.55, .55, .55], [.35, .35, .35]]   # wall / ceiling / floor
    return classes, remap, text / np.linalg.norm(text, axis=1, keepdims=True), pal


def decode_frame(video: Path, fid: int, w: int, h: int) -> np.ndarray:
    # build_cache.py scales the native frame straight to the cache grid (lanczos).
    cmd = ["ffmpeg", "-v", "error", "-i", str(video), "-vf",
           rf"select='eq(n\,{fid})',scale={w}:{h}:flags=lanczos", "-vframes", "1",
           "-f", "rawvideo", "-pix_fmt", "rgb24", "-"]
    return np.frombuffer(subprocess.run(cmd, capture_output=True, check=True).stdout, np.uint8).reshape(h, w, 3)


def mesh_labels(scene: str, cid: dict, remap: dict):
    import open3d as o3d
    S = Path(f"/data/ScanNetpp/data/{scene}/scans")
    mv = np.asarray(o3d.io.read_triangle_mesh(str(S / "mesh_aligned_0.05.ply")).vertices)
    seg = np.asarray(json.loads((S / "segments.json").read_text())["segIndices"])
    lab = np.full(len(mv), -1)
    seg2v: dict[int, list[int]] = {}
    for i, s in enumerate(seg):
        seg2v.setdefault(s, []).append(i)
    for g in json.loads((S / "segments_anno.json").read_text())["segGroups"]:
        c = cid.get(remap.get(g["label"], g["label"]), -1)
        for s in g["segments"]:
            lab[seg2v.get(s, [])] = c
    return cKDTree(mv), lab


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("clip", type=Path)
    ap.add_argument("frames", type=int, nargs="+", help="cache frame indices")
    ap.add_argument("--out", type=Path, default=OUT)
    args = ap.parse_args()
    import matplotlib; matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from matplotlib.patches import Patch

    classes, remap, text, pal = top100()
    cid = {c: i for i, c in enumerate(classes)}
    name = args.clip.name
    align = sem.load_align()
    ct = sem.ChorusTargets(name, align)
    scene = align[name]["scene"]
    lang = np.load(Path(sem.CHORUS_ROOT) / scene / "lang.npy")
    mtree, mlab = mesh_labels(scene, cid, remap)
    c = ClipCache(args.clip)
    h, w = c.meta.height, c.meta.width
    fids = json.loads((args.clip / "meta.json").read_text())["frame_ids"]
    for i in args.frames:
        dep = torch.from_numpy(np.ascontiguousarray(c.gt_depth[i])).float()
        v, u = np.nonzero(dep.numpy() > 0)
        p = sem.lift_pixels(dep, torch.from_numpy(c.gt_intrinsics[i]).float(),
                            torch.from_numpy(c.gt_c2w[i]).float(), v, u)
        X = (p * ct.scale) @ ct.C0[:3, :3].T + ct.C0[:3, 3]
        dist, vi = ct.tree.query(X, distance_upper_bound=ct.max_dist)
        hit = np.isfinite(dist)
        _, ok = ct.lookup(p)
        assert np.array_equal(ok.numpy(), hit), "lookup disagrees"
        f = lang[vi[hit]].astype(np.float32)
        cls = (f / np.linalg.norm(f, axis=1, keepdims=True)) @ text.T
        pred = np.full((h, w), -1); pred[v[hit], u[hit]] = cls.argmax(1)
        dm, mi = mtree.query(X)
        gt = np.full((h, w), -1); gt[v, u] = np.where(dm < 0.05, mlab[mi], -1)

        rgb = decode_frame(Path(f"/data/ScanNetpp/data/{scene}/iphone/rgb.mkv"), fids[i], w, h)
        def paint(lbl):
            im = np.zeros((h, w, 3)); m = lbl >= 0
            im[m] = pal[lbl[m]]
            return im
        m = (gt >= 0) & (pred >= 0)
        acc = float((pred[m] == gt[m]).mean())
        ious = [((pred == k) & (gt == k) & m).sum() / (((pred == k) | (gt == k)) & m).sum() for k in np.unique(gt[m])]
        d = args.out / name / f"frame_{i:03d}"
        d.mkdir(parents=True, exist_ok=True)
        fig, ax = plt.subplots(2, 2, figsize=(14, 10.5))
        ax[0, 0].imshow(rgb); ax[0, 0].set_title(f"{name} cache frame {i} (video frame {fids[i]})")
        ax[0, 1].imshow(paint(pred)); ax[0, 1].set_title(f"Chorus voxels (lang) x SigLIP2 text, black = no voxel <= {100 * ct.max_dist:.0f} cm")
        ax[1, 0].imshow(paint(gt)); ax[1, 0].set_title("ScanNet++ mesh annotation (top-100), black = unlabelled")
        ax[1, 1].imshow(0.45 * rgb / 255 + 0.55 * paint(pred)); ax[1, 1].set_title(f"overlay   pixel acc {acc:.2f}, mIoU {np.mean(ious):.2f}")
        for a in ax.flat:
            a.axis("off")
        shown = [k for k, _ in sorted(((k, (pred == k).sum() + (gt == k).sum()) for k in set(pred[pred >= 0]) | set(gt[gt >= 0])),
                                      key=lambda x: -x[1])][:18]
        fig.legend([Patch(color=pal[k]) for k in shown], [classes[k] for k in shown], loc="lower center", ncol=6, fontsize=10)
        fig.tight_layout(rect=(0, 0.06, 1, 1)); fig.savefig(d / "seg.png", dpi=90); plt.close(fig)

        uv, first = np.unique(vi[hit], return_index=True)
        write_ply(d / "voxels.ply", ct.xyz[uv], pal[cls.argmax(1)[first]])
        write_ply(d / "hit_points.ply", X, rgb[v, u] / 255.0)
        print(f"{name} frame {i}: {len(v)} rays, {hit.mean():.2f} hit a voxel, {len(uv)} voxels, "
              f"pixel acc {acc:.3f}, mIoU {np.mean(ious):.3f} -> {d}", flush=True)


if __name__ == "__main__":
    main()
