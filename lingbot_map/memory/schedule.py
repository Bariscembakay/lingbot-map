"""Which frame the write ingests at each step.

Two interchangeable regimes:

- ``disjoint``  -- the state is the compressed version of the *evicted* past, so
  state and KV cache partition the sequence exactly, with no overlap and no gap.
- ``overlap``   -- the state absorbs every frame as it is processed, so it
  duplicates what the cache still holds. This is CUT3R's regime, where the state
  is the only memory.

The disjoint lag is not a free parameter. Attention sees the cache *after*
eviction (`layers/attention.py`: append :700, evict :703, attend :729), so at
step ``i`` the model sees ``{0..scale-1} u {i-window+1..i}``. The read consumes
``S_{i-1}``, not ``S_i``. Requiring the two to tile the sequence gives

    (i-1) - lag == i - window   =>   lag == window - 1

Off by one in either direction leaves a frame in neither memory, or in both.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Optional, Tuple

DISJOINT = "disjoint"
OVERLAP = "overlap"


@dataclass(frozen=True)
class WriteSchedule:
    mode: str = DISJOINT
    scale_frames: int = 8
    sliding_window: int = 64

    def __post_init__(self) -> None:
        if self.mode not in (DISJOINT, OVERLAP):
            raise ValueError(f"unknown mode {self.mode!r}")

    @property
    def lag(self) -> int:
        return self.sliding_window - 1 if self.mode == DISJOINT else 0

    @property
    def first_written_frame(self) -> int:
        """Anchor frames are never evicted, so in disjoint mode they never arrive."""
        return self.scale_frames if self.mode == DISJOINT else 0

    def frame_for_step(self, i: int) -> Optional[int]:
        """Frame index the write ingests at step `i`, or None if it writes nothing."""
        j = i - self.lag
        return j if j >= self.first_written_frame else None

    def state_covers(self, i: int) -> Optional[Tuple[int, int]]:
        """Inclusive range held by the state the read at step `i` sees (S_{i-1})."""
        hi = (i - 1) - self.lag
        lo = self.first_written_frame
        return (lo, hi) if hi >= lo else None

    def cache_covers(self, i: int) -> Tuple[Tuple[int, int], Optional[Tuple[int, int]]]:
        """Inclusive ranges the aggregator's attention sees at step `i`.

        Returns (anchors, window). The window is None only before any frame past
        the anchors has been processed.
        """
        anchors = (0, min(self.scale_frames - 1, i))
        resident = self.scale_frames + self.sliding_window
        lo = max(i - self.sliding_window + 1, self.scale_frames) if i + 1 > resident \
            else self.scale_frames
        window = (lo, i) if i >= self.scale_frames else None
        return anchors, window


def coverage_report(sched: WriteSchedule, num_frames: int) -> dict:
    """Per-step audit of what the state and cache jointly cover.

    Used by the validation suite: in disjoint mode `missing` and `duplicated`
    must both be empty at every step.
    """
    missing, duplicated = [], []
    for i in range(num_frames):
        seen = set()
        dupe = set()
        for rng in (*sched.cache_covers(i), sched.state_covers(i)):
            if rng is None:
                continue
            for f in range(rng[0], rng[1] + 1):
                (dupe if f in seen else seen).add(f)
        gap = set(range(i + 1)) - seen
        if gap:
            missing.append((i, sorted(gap)))
        if dupe:
            duplicated.append((i, sorted(dupe)))
    return {
        "mode": sched.mode,
        "lag": sched.lag,
        "num_writes": sum(sched.frame_for_step(i) is not None for i in range(num_frames)),
        "first_write_step": next(
            (i for i in range(num_frames) if sched.frame_for_step(i) is not None), None
        ),
        "steps_with_gap": len(missing),
        "steps_with_overlap": len(duplicated),
        "first_gap": missing[0] if missing else None,
    }
