"""Reading the teacher cache into the tensors one training step needs.

The split of a tap tensor matters and is easy to invert: entries are
`cat([frame_intermediates, global_intermediates], -1)` (`aggregator/base.py:603`),
so `[..., :1024]` is the stream *after the frame block* and `[..., 1024:]` is the
stream *after the global block*. The latter is "the aggregator's output" -- the
write's input, the read's query, and the half the read replaces.
"""
from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Optional, Sequence

import numpy as np
import torch

from .cache_format import ClipCache
from . import streams as ST

TAP23 = ST.TAP23   # index of layer 23 within the 4 cached taps
HALF = ST.HALF


def find_clips(root: Path | str) -> List[Path]:
    root = Path(root)
    return sorted(p for p in root.iterdir() if p.is_dir() and (p / "meta.json").exists())


class ClipReader:
    """One clip, held as memmaps, served as torch tensors on demand."""

    def __init__(self, clip_dir: Path | str, device: torch.device | str = "cpu",
                 dtype: torch.dtype = torch.float32):
        self.cache = ClipCache(clip_dir)
        self.meta = self.cache.meta
        self.device = torch.device(device)
        self.dtype = dtype

    def __len__(self) -> int:
        return self.meta.num_frames

    def _t(self, arr) -> torch.Tensor:
        return torch.from_numpy(np.ascontiguousarray(arr)).to(self.device, self.dtype)

    def tokens(self, i: int, tap: int = TAP23) -> torch.Tensor:
        """[1, P, 1024] the post-global-block stream of `tap` for frame i."""
        return self._t(self.cache.taps[i, tap, :, HALF:]).unsqueeze(0)

    def token_dict(self, i: int, taps) -> Dict[int, torch.Tensor]:
        return {t: self.tokens(i, t) for t in taps}

    def frame_half(self, i: int, tap: int = TAP23) -> torch.Tensor:
        """[1, P, 1024] the half of `tap` the read does not touch."""
        return self._t(self.cache.taps[i, tap, :, :HALF]).unsqueeze(0)

    def raw_tap(self, i: int, tap: int) -> torch.Tensor:
        """[1, 1, P, 2048] an untouched tap, as the frozen head expects it."""
        return self._t(self.cache.taps[i, tap]).unsqueeze(0).unsqueeze(0)

    def rebuild_tap(self, i: int, tap: int, refined: torch.Tensor) -> torch.Tensor:
        """[1, P, 2048] `tap` with its second half replaced by `refined`.

        Both heads consume this: the depth head as the deepest of four taps, the
        camera head as `[:, :, 0]`. The camera head needs the **full 2048-d**
        token, not the 1024-d refined half -- passing the half is a silent
        dimension error that surfaces deep inside the head's LayerNorm.
        """
        return torch.cat([self.frame_half(i, tap), refined], dim=-1)

    def head_inputs(self, i: int, refined: Dict[int, torch.Tensor]) -> List[torch.Tensor]:
        """The 4-tap list for the frozen heads, with every refined tap substituted.

        Taps absent from `refined` pass through untouched.
        """
        out = []
        for t in range(4):
            if t in refined:
                out.append(self.rebuild_tap(i, t, refined[t]).unsqueeze(1))
            else:
                out.append(self.raw_tap(i, t))
        return out

    def stream(self, i: int, s: int) -> torch.Tensor:
        """[1, P, 1024] one of the 8 half-tap streams of frame i. See streams.py."""
        tap, _ = ST.split(s)
        return self._t(self.cache.taps[i, tap, :, ST.slice_for(s)]).unsqueeze(0)

    def stream_cat(self, i: int, ids: Sequence[int]) -> torch.Tensor:
        """[1, len(ids)*P, 1024] several streams of frame i, concatenated as tokens.

        This is the write's KV. Concatenating along the token axis rather than the
        feature axis keeps the write at dim 1024 whatever it ingests, so the state
        and the read never change shape with `write_input`.
        """
        if len(ids) == 1:
            return self.stream(i, ids[0])
        return torch.cat([self.stream(i, s) for s in ids], dim=1)

    def stream_mean(self, ids: Sequence[int], num_frames: int = 16) -> torch.Tensor:
        """[len(ids), 1024] mean of each stream over a subsample of frames.

        Used to initialise the raymap query embeddings so that at gate 0 -- where
        the reader is the identity -- the frozen head decodes the *mean tap* rather
        than noise. That is what makes the Loss-2 term start on-manifold with a
        finite loss instead of needing a warmup stage.
        """
        idx = np.linspace(0, len(self) - 1, min(num_frames, len(self))).astype(int)
        out = []
        for s in ids:
            tap, _ = ST.split(s)
            block = np.asarray(self.cache.taps[idx][:, tap, :, ST.slice_for(s)])
            out.append(block.reshape(-1, ST.HALF).mean(0))
        return self._t(np.stack(out))

    def teacher_camera_tokens(self) -> torch.Tensor:
        """[L, 2048] token 0 of tap 23 -- what the camera bridge replays."""
        return self._t(self.cache.taps[:, TAP23, 0, :])

    def gt_depth(self, i: int) -> torch.Tensor:
        return self._t(self.cache.gt_depth[i]).unsqueeze(0)

    def teacher_depth(self, i: int) -> torch.Tensor:
        return self._t(self.cache.depth[i]).unsqueeze(0)

    def teacher_conf(self, i: int) -> torch.Tensor:
        return self._t(self.cache.conf[i]).unsqueeze(0)

    def gt_c2w(self, idx: Sequence[int] | slice) -> torch.Tensor:
        return self._t(self.cache.gt_c2w[idx])

    def pose_enc(self, i: int) -> torch.Tensor:
        return self._t(self.cache.pose_enc[i]).unsqueeze(0)

    def revisit(self, i: int) -> float:
        return float(self.cache.revisit[i])

    @property
    def hw(self):
        return self.meta.height, self.meta.width


def sample_query(rng: np.random.Generator, step: int, lag: int,
                 first_written: int, unroll: int,
                 reachable_prob: float = 0.5) -> Optional[int]:
    """Pick a past frame the state could actually answer for.

    Frame `q` is written at step `q + lag`, and the read at step `i` consumes
    `S_{i-1}`, so the state holds `[first_written, i - 1 - lag]`. Two bands matter
    and they are not the same thing:

      reachable  q in [i - 1 - lag - unroll, i - 1 - lag]
                 the write that stored q is inside the BPTT window, so gradient
                 can still teach the write *what* to keep;
      tail       q in [first_written, i - 1 - lag - unroll)
                 the write is long detached, so the only lesson available is
                 "do not destroy what is already there" -- which is a property of
                 the shared write weights and therefore still learnable.

    Anything above `i - 1 - lag` is in the KV cache and was never written to the
    state; sampling it would train the read to invent. The previous version of
    this function sampled `[0, step - min_gap]` and so, under the disjoint
    schedule, spent half its samples on exactly that unanswerable region.
    """
    hi = step - 1 - lag
    if hi < first_written:
        return None
    cut = hi - unroll
    if rng.random() < reachable_prob or cut <= first_written:
        lo = max(first_written, cut)
        return int(rng.integers(lo, hi + 1))
    return int(rng.integers(first_written, cut))
