#!/usr/bin/env python3
"""Training-sequence sources for the 1048-frame lingbot caches: one adapter per dataset.

Each adapter exposes
  seqs()                     sequence ids
  poses(seq)                 (N, 4, 4) OpenCV camera-to-world in metres, raw world frame;
                             frames whose pose is invalid are NaN
  frames(seq, ids)           (images uint8 [F, H, W, 3] at the size lingbot is fed, K [F, 3, 3] at that size)
  depth(seq, ids)            (depth metres [F, h, w], K [F, 3, 3] at the depth size), or None
  metric                     True when there is no usable depth (driving): GT stays metric
Conventions (verified against lingbot, campaigns/spatial_memory/_data_prep/gt_conventions.md):
TartanAir NED body -> OpenCV, MatrixCity OpenGL -> OpenCV, KITTI-360 cropped around its principal
point, ScanNet++ / ScanNet v2 already OpenCV. Images are resized as the benchmark feeds lingbot
(area budget 255000, align 14) except KITTI-360 (504x280 principal-point-centred crop).
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path

import numpy as np

os.environ.setdefault("OPENCV_IO_ENABLE_OPENEXR", "1")
import cv2  # noqa: E402

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))

NED_TO_CV = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)
GL_TO_CV = np.diag([1.0, -1.0, -1.0])


def bench_size(W, H):
    from scripts.memory.build_cache import benchmark_resize
    return benchmark_resize(W, H, 518, 14, 255000)


def resize_all(paths, W, H, K):
    w, h = bench_size(W, H)
    imgs = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA)
                     for p in paths])
    K = K.astype(np.float64).copy()
    K[0] *= w / W
    K[1] *= h / H
    return imgs, np.repeat(K[None], len(paths), 0)


def quat_to_R(q):
    from check_training_data import quat_to_R as f
    return f(q)


class Tartan:
    metric = False
    LAYOUT = {  # root, trajectory glob, image pattern, depth pattern, pose file, (W, H, f)
        "tartanair-v1": ("/data/tartanair-v1", "*/*/P*", "image_left/{:06d}_left.png", "depth_left/{:06d}_left_depth.npy", "pose_left.txt", (640, 480, 320.0)),
        "tartanair-v2": ("/data/tartanair-v2", "*/Data_*/P*", "image_lcam_front/{:06d}_lcam_front.png", "depth_lcam_front/{:06d}_lcam_front_depth.png", "pose_lcam_front.txt", (640, 640, 320.0)),
        "tartanground": ("/data/tartanground", "*/Data_*/P*", "image_lcam_front/{:06d}_lcam_front.png", "depth_lcam_front/{:06d}_lcam_front_depth.png", "pose_lcam_front.txt", (640, 640, 320.0)),
    }

    def __init__(self, name):
        self.name = name
        self.root, self.glob, self.img, self.dep, self.pose, (self.W, self.H, f) = self.LAYOUT[name]
        self.K = np.array([[f, 0, self.W / 2], [0, f, self.H / 2], [0, 0, 1]])

    def seqs(self):
        return [str(p.relative_to(self.root)) for p in sorted(Path(self.root).glob(self.glob))]

    def poses(self, seq):
        a = np.loadtxt(Path(self.root) / seq / self.pose)
        T = np.tile(np.eye(4), (len(a), 1, 1))
        T[:, :3, :3] = quat_to_R(a[:, 3:7]) @ NED_TO_CV
        T[:, :3, 3] = a[:, :3]
        return T

    def frames(self, seq, ids):
        return resize_all([Path(self.root) / seq / self.img.format(i) for i in ids], self.W, self.H, self.K)

    def depth(self, seq, ids):
        out = []
        for i in ids:
            p = Path(self.root) / seq / self.dep.format(i)
            if p.suffix == ".npy":
                out.append(np.load(p).astype(np.float32))
            else:   # TartanAir v2: float32 depth packed into the 4 bytes of an RGBA PNG
                out.append(cv2.imread(str(p), cv2.IMREAD_UNCHANGED).view("<f4")[..., 0])
        d = np.stack(out)
        d[~np.isfinite(d) | (d > 1000)] = 0
        return d, np.repeat(self.K[None], len(ids), 0)


class MatrixCity:
    name, metric = "matrixcity", False
    ROUTES = Path("/group/compact-3dmem/campaigns/spatial_memory/_data_prep/matrixcity_street_routes_1048f_seed0.json")
    IMG = Path("/data/matrix_city/small_city/street/train")
    DEP = Path("/data/matrix_city/small_city_depth/street/train")
    K = np.array([[500.0, 0, 500], [0, 500.0, 500], [0, 0, 1]])
    # EXR depth is in centimetres: bottom-centre ground ~2.6 m from a camera at z = 3 m, medians 7.6-57 m
    DEPTH_UNIT = 0.01

    def __init__(self):
        self.routes = json.loads(self.ROUTES.read_text())["routes"]

    def seqs(self):
        return [f"route{k:03d}" for k in range(len(self.routes))]

    def poses(self, seq):
        T = np.asarray(self.routes[int(seq[5:])]["c2w"], np.float64)
        T[:, :3, :3] = T[:, :3, :3] @ GL_TO_CV
        return T

    def frames(self, seq, ids):
        fr = self.routes[int(seq[5:])]["frames"]
        return resize_all([self.IMG / fr[i][0] / f"{fr[i][1]:04d}.png" for i in ids], 1000, 1000, self.K)

    def depth(self, seq, ids):
        import OpenEXR    # pip OpenCV builds lack EXR support
        fr = self.routes[int(seq[5:])]["frames"]
        out = []
        for i in ids:
            with OpenEXR.File(str(self.DEP / f"{fr[i][0]}_depth" / f"{fr[i][1]:04d}.exr")) as f:
                x = np.asarray(next(iter(f.channels().values())).pixels, np.float32)
            out.append(x[..., 0] if x.ndim == 3 else x)
        d = np.stack(out)
        d[~np.isfinite(d) | (d >= 65000)] = 0          # sky is stored as the float16 maximum, 65504
        return d * self.DEPTH_UNIT, np.repeat(self.K[None], len(ids), 0)


class ScanNetV2:
    name, metric = "scannet-v2", False
    ROOT = Path("/data/scannet-v2-extracted")

    def seqs(self):
        return sorted(p.name for p in self.ROOT.iterdir() if (p / "poses.txt").exists())

    def meta(self, seq):
        return json.loads((self.ROOT / seq / "intrinsics.json").read_text())

    def poses(self, seq):
        a = np.loadtxt(self.ROOT / seq / "poses.txt").reshape(-1, 4, 4)
        bad = ~np.isfinite(a).all((1, 2))
        a[bad] = np.nan
        return a.astype(np.float64)

    def frames(self, seq, ids):
        m = self.meta(seq)
        W, H = m["color_wh"]
        return resize_all([self.ROOT / seq / "color" / f"{i:06d}.jpg" for i in ids], W, H, np.asarray(m["intrinsic_color"])[:3, :3])

    def depth(self, seq, ids):
        m = self.meta(seq)
        d = np.stack([cv2.imread(str(self.ROOT / seq / "depth" / f"{i:06d}.png"), cv2.IMREAD_UNCHANGED) for i in ids]).astype(np.float32)
        return d / 1000.0, np.repeat(np.asarray(m["intrinsic_depth"])[None, :3, :3], len(ids), 0)


class ScanNetPP:
    name, metric = "scannetpp", False
    ROOT = Path("/data/ScanNetpp")

    def seqs(self):
        return sorted(p.name for p in (self.ROOT / "data").iterdir() if (p / "iphone/pose_intrinsic_imu.json").exists()
                      and (p / "iphone/rgb.mkv").exists() and (p / "iphone/depth.bin").exists())

    def _meta(self, seq):
        return json.loads((self.ROOT / "data" / seq / "iphone/pose_intrinsic_imu.json").read_text())

    def poses(self, seq):
        from scripts.memory.build_cache import probe_video
        d = self._meta(seq)
        n = len(d)
        nb = probe_video(self.ROOT / "data" / seq / "iphone/rgb.mkv")[2]
        n = n if nb is None else min(n, nb)
        return np.stack([np.asarray(d[f"frame_{i:06d}"]["aligned_pose"], np.float64) for i in range(n)])

    def frames(self, seq, ids):
        from scripts.memory.build_cache import decode_frames
        video = self.ROOT / "data" / seq / "iphone/rgb.mkv"
        w, h = bench_size(1920, 1440)
        imgs = decode_frames(video, list(ids), w, h)           # needs a constant stride, which clips have
        d = self._meta(seq)
        K = np.stack([np.asarray(d[f"frame_{i:06d}"]["intrinsic"], np.float64) for i in ids])
        K[:, 0] *= w / 1920
        K[:, 1] *= h / 1440
        return imgs, K

    def depth(self, seq, ids):
        from lingbot_map.memory.gt import detect_depth_shape, read_iphone_depth, scale_intrinsics
        ip = self.ROOT / "data" / seq / "iphone"
        dh, dw, codec = detect_depth_shape(ip / "depth.bin")
        dep = read_iphone_depth(ip / "depth.bin", list(ids), (dh, dw, codec)).astype(np.float32)
        d = self._meta(seq)
        K = np.stack([np.asarray(d[f"frame_{i:06d}"]["intrinsic"], np.float64) for i in ids])
        return dep, scale_intrinsics(K, (1440, 1920), (dh, dw))


class Kitti360:
    name, metric = "kitti360", True

    def __init__(self):
        import build_seq_inputs as bs
        self.bs = bs
        self._segs = {}

    def seqs(self):
        out = []
        for d in sorted(Path("/data/kitti360").glob("*_sync")):
            if not (d / "image_00/data_rect").exists():
                continue
            for k, seg in enumerate(self._seg(d.name)):
                if len(seg) >= 64:
                    out.append(f"{d.name}/seg{k}")
        return out

    def _seg(self, drive):
        if drive not in self._segs:
            self._segs[drive] = self.bs.kitti360(drive)
        return self._segs[drive]

    def _run(self, seq):
        drive, k = seq.split("/seg")
        return self._seg(drive)[int(k)]

    def poses(self, seq):
        return np.stack([T for _, T, _ in self._run(seq)]).astype(np.float64)

    def frames(self, seq, ids):
        run = self._run(seq)
        out = [self.bs.pp_center_fit(cv2.cvtColor(cv2.imread(str(run[i][0])), cv2.COLOR_BGR2RGB), run[i][2]) for i in ids]
        return np.stack([o[0] for o in out]), np.stack([o[1] for o in out])

    def depth(self, seq, ids):
        return None


def source(name):
    if name in Tartan.LAYOUT:
        return Tartan(name)
    return {"matrixcity": MatrixCity, "scannet-v2": ScanNetV2, "scannetpp": ScanNetPP, "kitti360": Kitti360}[name]()
