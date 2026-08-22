"""On-disk format for the precomputed teacher cache.

One directory per clip. Arrays are contiguous over the frame axis so a training
pass reads them sequentially; `np.memmap` keeps resident memory bounded
regardless of clip length.
"""
from __future__ import annotations

import json
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

TAPS = "taps.npy"
DEPTH = "depth.npy"
CONF = "conf.npy"
POSE = "pose_enc.npy"
GT_DEPTH = "gt_depth.npy"
GT_C2W = "gt_c2w.npy"
REVISIT = "revisit.npy"
META = "meta.json"

FORMAT_VERSION = 2

# fp16 carries 10 mantissa bits against bfloat16's 8, so storing a bf16
# aggregator output as fp16 is lossless -- but only inside fp16's much smaller
# exponent range. Writers must check, because silent inf here would poison every
# label built from it.
FP16_MAX = 65504.0


@dataclass
class ClipMeta:
    format_version: int
    scene: str
    clip_index: int
    frame_ids: List[int]
    stride: int
    num_frames: int
    height: int
    width: int
    patch_h: int
    patch_w: int
    num_tokens: int
    patch_start_idx: int
    tap_layers: List[int]
    embed_dim: int
    tap_dtype: str
    scale_frames: int
    kv_cache_sliding_window: int
    keyframe_interval: int
    model_sha256: str
    git_sha: str
    git_dirty: bool
    gt_scale: float
    gt_convention: str
    stats: Dict[str, Any]

    @property
    def num_patches(self) -> int:
        return self.patch_h * self.patch_w

    def to_json(self) -> str:
        return json.dumps(self.__dict__, indent=2)

    @classmethod
    def load(cls, clip_dir: Path | str) -> "ClipMeta":
        d = json.loads((Path(clip_dir) / META).read_text())
        if d.get("format_version") != FORMAT_VERSION:
            raise ValueError(
                f"{clip_dir}: cache format {d.get('format_version')} != "
                f"expected {FORMAT_VERSION}"
            )
        return cls(**d)


class ClipCache:
    """Read-only view over one clip directory."""

    def __init__(self, clip_dir: Path | str):
        self.dir = Path(clip_dir)
        self.meta = ClipMeta.load(self.dir)
        m = self.meta
        self.taps = np.memmap(
            self.dir / TAPS, dtype=m.tap_dtype, mode="r",
            shape=(m.num_frames, len(m.tap_layers), m.num_tokens, m.embed_dim),
        )
        self.depth = np.memmap(
            self.dir / DEPTH, dtype="float16", mode="r",
            shape=(m.num_frames, m.height, m.width),
        )
        self.conf = np.memmap(
            self.dir / CONF, dtype="float16", mode="r",
            shape=(m.num_frames, m.height, m.width),
        )
        self.pose_enc = np.load(self.dir / POSE)
        self.gt_depth = np.memmap(
            self.dir / GT_DEPTH, dtype="float16", mode="r",
            shape=(m.num_frames, m.height, m.width),
        )
        self.gt_c2w = np.load(self.dir / GT_C2W)
        self.revisit = np.load(self.dir / REVISIT)

    def __len__(self) -> int:
        return self.meta.num_frames

    def patch_taps(self, i: int) -> np.ndarray:
        """The 4 tap grids for frame i, patch tokens only, as the read must emit."""
        return np.asarray(self.taps[i][:, self.meta.patch_start_idx:])
