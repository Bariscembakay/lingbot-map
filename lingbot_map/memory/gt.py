"""ScanNet++ iPhone ground truth: depth, poses, canonical scale, revisit score.

Formats established by inspection, not documentation:

* `iphone/depth.bin` is a sequence of `uint32` little-endian payload lengths each
  followed by an **LZ4 block** compressing 256x192 `uint16` millimetres. Verified:
  values 1.5-4.7 m indoors, 4-8 mm median spatial gradient, 2 mm frame-to-frame.
* `iphone/pose_intrinsic_imu.json` is keyed `frame_000000` with per-frame
  `pose`, `aligned_pose` (registered to the laser scan), `intrinsic` (for the full
  1920x1440 image) and `imu`.

The camera convention is **detected, not assumed** -- see `detect_convention`.
Getting it wrong silently mirrors the geometry, and it is exactly the kind of
error a loss curve hides.
"""
from __future__ import annotations

import json
import struct
from pathlib import Path
from typing import Dict, List, Optional, Sequence, Tuple

import cv2
import lz4.block
import numpy as np

DEPTH_H, DEPTH_W = 192, 256
MM_PER_M = 1000.0

# c2w_opencv = c2w_source @ FLIP. ARKit/OpenGL is +Y up, -Z forward; OpenCV is
# +Y down, +Z forward, so the usual fix is diag(1,-1,-1,1).
CONVENTIONS: Dict[str, np.ndarray] = {
    "opencv": np.diag([1.0, 1.0, 1.0, 1.0]),
    "opengl": np.diag([1.0, -1.0, -1.0, 1.0]),
    "flip_x": np.diag([-1.0, 1.0, -1.0, 1.0]),
    "flip_xy": np.diag([-1.0, -1.0, 1.0, 1.0]),
}


def read_iphone_depth(path: Path | str, frame_ids: Sequence[int]) -> np.ndarray:
    """[L, 192, 256] float32 metres. 0 marks invalid."""
    want = set(frame_ids)
    order = {f: k for k, f in enumerate(frame_ids)}
    out = np.zeros((len(frame_ids), DEPTH_H, DEPTH_W), np.float32)
    nbytes = DEPTH_H * DEPTH_W * 2
    found = 0
    with open(path, "rb") as f:
        idx = 0
        while True:
            hdr = f.read(4)
            if len(hdr) < 4:
                break
            n = struct.unpack("<I", hdr)[0]
            if idx in want:
                raw = lz4.block.decompress(f.read(n), uncompressed_size=nbytes)
                out[order[idx]] = (
                    np.frombuffer(raw, np.uint16).reshape(DEPTH_H, DEPTH_W).astype(np.float32)
                    / MM_PER_M
                )
                found += 1
                if found == len(frame_ids):
                    break
            else:
                f.seek(n, 1)
            idx += 1
    if found != len(frame_ids):
        raise RuntimeError(f"{path}: found {found} of {len(frame_ids)} requested frames")
    return out


def read_iphone_meta(path: Path | str, frame_ids: Sequence[int],
                     pose_key: str = "aligned_pose") -> Tuple[np.ndarray, np.ndarray]:
    """(poses [L,4,4], intrinsics [L,3,3]) for the full-resolution image."""
    d = json.loads(Path(path).read_text())
    poses, intr = [], []
    for f in frame_ids:
        rec = d[f"frame_{f:06d}"]
        poses.append(np.asarray(rec[pose_key], np.float64))
        intr.append(np.asarray(rec["intrinsic"], np.float64))
    return np.stack(poses), np.stack(intr)


def resize_depth(depth: np.ndarray, height: int, width: int) -> np.ndarray:
    """Nearest-neighbour, so invalid (0) pixels never bleed into valid ones."""
    return np.stack([
        cv2.resize(d, (width, height), interpolation=cv2.INTER_NEAREST) for d in depth
    ])


def scale_intrinsics(intr: np.ndarray, src_hw: Tuple[int, int],
                     dst_hw: Tuple[int, int]) -> np.ndarray:
    out = intr.copy()
    sy, sx = dst_hw[0] / src_hw[0], dst_hw[1] / src_hw[1]
    out[:, 0, 0] *= sx; out[:, 0, 2] *= sx
    out[:, 1, 1] *= sy; out[:, 1, 2] *= sy
    return out


def relative_to_first(c2w: np.ndarray) -> np.ndarray:
    """Re-express c2w poses with frame 0 as the world origin, as the model does."""
    return np.linalg.inv(c2w[0])[None] @ c2w


def unproject(depth: np.ndarray, intr: np.ndarray, c2w: np.ndarray,
              stride: int = 8) -> np.ndarray:
    """[N, 3] world points from a subsampled pixel grid. Invalid depths dropped."""
    L, H, W = depth.shape
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    uv1 = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], -1)      # [P,3]
    pts = []
    for i in range(L):
        z = depth[i, vs.ravel(), us.ravel()]
        ok = z > 0
        if not ok.any():
            continue
        cam = (np.linalg.inv(intr[i]) @ uv1[ok].T).T * z[ok, None]
        pts.append((c2w[i, :3, :3] @ cam.T).T + c2w[i, :3, 3])
    return np.concatenate(pts) if pts else np.zeros((0, 3))


def canonical_scale(depth: np.ndarray, intr: np.ndarray, c2w_rel: np.ndarray,
                    anchor_frames: int) -> float:
    """Paper §3.2: mean distance of the anchor-frame GT point cloud from the origin.

    All GT depths and translations are divided by this, which is what puts GT in
    the same units the model was trained to predict.
    """
    pts = unproject(depth[:anchor_frames], intr[:anchor_frames], c2w_rel[:anchor_frames])
    if len(pts) == 0:
        raise RuntimeError("no valid GT depth in the anchor frames")
    return float(np.linalg.norm(pts, axis=-1).mean())


def detect_convention(c2w_raw: np.ndarray, pred_c2w: np.ndarray) -> Tuple[str, Dict[str, float]]:
    """Pick the camera convention that best matches the model's own trajectory.

    Compares *relative* rotations, which are invariant to the world frame and to
    scale, so this tests only the axis convention.

    Pairs are taken across **long baselines**, not consecutive frames: a slow room
    scan has near-identity inter-frame rotations, and flipping axes of a
    near-identity rotation barely changes it, so the consecutive-frame version
    separated the four candidates by only ~5 degrees -- comparable to the
    teacher's own error. Long baselines give large rotations and a wide margin.
    """
    def rel_rots(c2w):
        n = len(c2w)
        i = np.arange(n)
        j = (i + n // 2) % n
        a, b = c2w[i, :3, :3], c2w[j, :3, :3]
        return np.einsum("nij,njk->nik", np.transpose(a, (0, 2, 1)), b)

    def geodesic_deg(ra, rb):
        m = np.einsum("nij,njk->nik", np.transpose(ra, (0, 2, 1)), rb)
        cos = np.clip((np.trace(m, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    target = rel_rots(pred_c2w)
    scores = {}
    for name, flip in CONVENTIONS.items():
        scores[name] = float(geodesic_deg(rel_rots(c2w_raw @ flip), target).mean())
    best = min(scores, key=scores.get)
    return best, scores


def revisit_score(depth: np.ndarray, intr: np.ndarray, c2w_rel: np.ndarray,
                  window: int, voxel: float = 0.05, stride: int = 8) -> np.ndarray:
    """Per frame: fraction of visible surface last observed more than `window` frames ago.

    This is the pre-flight measurement that says whether a clip can show a Loss-1
    effect at all. If the currently visible surface is all still inside the KV
    cache, the summary state has nothing to contribute and a flat loss curve would
    say nothing about the architecture.

    A voxel hash of last-seen frame index, so the whole clip costs one pass.
    """
    L, H, W = depth.shape
    vs, us = np.meshgrid(np.arange(0, H, stride), np.arange(0, W, stride), indexing="ij")
    uv1 = np.stack([us.ravel(), vs.ravel(), np.ones(us.size)], -1)
    last_seen: Dict[Tuple[int, int, int], int] = {}
    out = np.zeros(L, np.float32)

    for i in range(L):
        z = depth[i, vs.ravel(), us.ravel()]
        ok = z > 0
        if not ok.any():
            continue
        cam = (np.linalg.inv(intr[i]) @ uv1[ok].T).T * z[ok, None]
        world = (c2w_rel[i, :3, :3] @ cam.T).T + c2w_rel[i, :3, 3]
        keys = np.floor(world / voxel).astype(np.int64)
        stale = 0
        seen = 0
        for k in map(tuple, keys):
            prev = last_seen.get(k)
            if prev is not None:
                seen += 1
                if i - prev > window:
                    stale += 1
            last_seen[k] = i
        out[i] = stale / max(seen, 1)
    return out


def prepare(scannetpp_root: Path | str, scene: str, frame_ids: Sequence[int],
            height: int, width: int, anchor_frames: int, window: int,
            pred_c2w: Optional[np.ndarray] = None,
            convention: str = "auto") -> dict:
    """Everything Loss 1 needs, in the model's canonical units."""
    iphone = Path(scannetpp_root) / "data" / scene / "iphone"
    depth_small = read_iphone_depth(iphone / "depth.bin", frame_ids)
    c2w_raw, intr_full = read_iphone_meta(iphone / "pose_intrinsic_imu.json", frame_ids)

    scores = None
    if convention == "auto":
        if pred_c2w is None:
            raise ValueError("convention='auto' needs pred_c2w to compare against")
        convention, scores = detect_convention(c2w_raw, pred_c2w)
    c2w = c2w_raw @ CONVENTIONS[convention]
    c2w_rel = relative_to_first(c2w)

    intr_small = scale_intrinsics(intr_full, (1440, 1920), (DEPTH_H, DEPTH_W))
    s = canonical_scale(depth_small, intr_small, c2w_rel, anchor_frames)

    revisit = revisit_score(depth_small, intr_small, c2w_rel, window)

    depth = resize_depth(depth_small, height, width) / s
    c2w_canon = c2w_rel.copy()
    c2w_canon[:, :3, 3] /= s

    return {
        "gt_depth": depth.astype(np.float32),
        "gt_c2w": c2w_canon.astype(np.float32),
        "gt_intrinsics": scale_intrinsics(intr_full, (1440, 1920), (height, width)).astype(np.float32),
        "scale": s,
        "convention": convention,
        "convention_scores_deg": scores,
        "revisit": revisit,
        "invalid_fraction": float((depth <= 0).mean()),
    }
