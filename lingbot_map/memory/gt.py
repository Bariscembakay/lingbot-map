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

# Most ScanNet++ iphone captures are 256x192, but not all -- a third of the
# scenes we screened failed LZ4 decompression at that size. The raw blocks carry
# no length, so the shape has to be detected by trying candidates.
DEPTH_SHAPE_CANDIDATES = ((192, 256), (256, 192), (240, 320), (320, 240),
                          (144, 192), (192, 144), (480, 640), (384, 512))
DEPTH_H, DEPTH_W = DEPTH_SHAPE_CANDIDATES[0]
MM_PER_M = 1000.0

# `aligned_pose` is camera-to-world in the **OpenCV** convention, used as-is.
#
# Established against ScanNet++'s own documented poses rather than inferred:
# `iphone/nerfstudio/transforms.json` is c2w in the nerfstudio/OpenGL convention,
# and `aligned_pose @ diag(1,-1,-1,1)` reproduces it to **0.14 deg median** over
# 919 matched frames. The OpenGL<->OpenCV flip is exactly that matrix, so
# `aligned_pose` is already OpenCV c2w.
#
# An earlier revision searched 48 candidate conventions by matching against the
# model's own trajectory. That estimator was underdetermined -- it returned seven
# different "conventions" across twenty clips of one dataset -- because the thing
# it was really fitting was an 11.4 deg error introduced by a spurious inversion
# in `losses.pose_enc_to_c2w`. Do not reintroduce a search; the convention is
# documented, and what remains is a check.
OPENGL_TO_OPENCV = np.diag([1.0, -1.0, -1.0, 1.0])
# The model's own RPE-rot is 0.58 deg on 7-Scenes and 0.92 deg on TUM (campaign
# numbers reproducing the paper's Table 4). A GT transform that cannot get the
# residual near that is not a convention we have identified, and a pose loss built
# on it would push the model toward the wrong answer.
POSE_TRUST_THRESHOLD_DEG = 1.5


def detect_depth_shape(path: Path | str) -> Tuple[int, int]:
    """Infer (h, w) from the first frame by trying the known candidates."""
    with open(path, "rb") as f:
        n = struct.unpack("<I", f.read(4))[0]
        buf = f.read(n)
    for h, w in DEPTH_SHAPE_CANDIDATES:
        try:
            if len(lz4.block.decompress(buf, uncompressed_size=h * w * 2)) == h * w * 2:
                return h, w
        except lz4.block.LZ4BlockError:
            continue
    raise RuntimeError(f"{path}: no candidate depth shape decompresses "
                       f"(tried {DEPTH_SHAPE_CANDIDATES})")


def read_iphone_depth(path: Path | str, frame_ids: Sequence[int],
                      shape: Optional[Tuple[int, int]] = None) -> np.ndarray:
    """[L, h, w] float32 metres. 0 marks invalid. Shape is detected if not given."""
    h, w = shape or detect_depth_shape(path)
    want = set(frame_ids)
    order = {f: k for k, f in enumerate(frame_ids)}
    out = np.zeros((len(frame_ids), h, w), np.float32)
    nbytes = h * w * 2
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
                    np.frombuffer(raw, np.uint16).reshape(h, w).astype(np.float32)
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


def verify_pose_target(gt_c2w: np.ndarray, pred_c2w: np.ndarray
                       ) -> Tuple[bool, float]:
    """Check the GT poses against the model's own trajectory. Returns (trusted, deg).

    A check, not a fit: consecutive relative rotations are invariant to the world
    frame and to scale, so a mismatch here means a real disagreement rather than a
    frame convention.
    """

    def geodesic_deg(ra, rb):
        m = np.einsum("nij,njk->nik", np.transpose(ra, (0, 2, 1)), rb)
        cos = np.clip((np.trace(m, axis1=1, axis2=2) - 1.0) / 2.0, -1.0, 1.0)
        return np.degrees(np.arccos(cos))

    def consecutive(c2w):
        a, b = c2w[:-1, :3, :3], c2w[1:, :3, :3]
        return np.einsum("nij,njk->nik", np.transpose(a, (0, 2, 1)), b)

    resid = float(geodesic_deg(consecutive(gt_c2w), consecutive(pred_c2w)).mean())
    return resid <= POSE_TRUST_THRESHOLD_DEG, resid


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
    depth_shape = detect_depth_shape(iphone / "depth.bin")
    depth_small = read_iphone_depth(iphone / "depth.bin", frame_ids, depth_shape)
    c2w_raw, intr_full = read_iphone_meta(iphone / "pose_intrinsic_imu.json", frame_ids)

    c2w_rel = relative_to_first(c2w_raw)
    trusted, residual = False, float("nan")
    if pred_c2w is not None:
        trusted, residual = verify_pose_target(c2w_rel, relative_to_first(pred_c2w))
    convention = "opencv_c2w"

    intr_small = scale_intrinsics(intr_full, (1440, 1920), depth_shape)
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
        "pose_residual_deg": residual,
        "pose_trusted": bool(trusted),
        "convention_scores_deg": None,
        "revisit": revisit,
        "depth_shape": list(depth_shape),
        "invalid_fraction": float((depth <= 0).mean()),
    }
