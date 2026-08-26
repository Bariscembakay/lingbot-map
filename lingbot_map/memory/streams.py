"""The stream vocabulary shared by the write, the read and Loss 2.

A cached tap entry is `cat([frame_intermediates, global_intermediates], -1)`
(`aggregator/base.py:603`), so each of the 4 taps carries **two** 1024-d streams.
Numbering them once, in one place, is what keeps the write's input, the raymap
query's output and the recall target from drifting apart:

    stream s  <->  (tap = s // 2, half = s % 2)      half 0 = pre-global
                                                     half 1 = post-global

The frozen depth head consumes all 8 (four 2048-d taps). The write has
historically consumed exactly one -- stream 7, tap 23's post-global half -- which
is why a recall target over all 8 would be unreachable for 7 of them.
"""
from __future__ import annotations

from typing import Sequence, Tuple

HALF = 1024
NUM_TAPS = 4
NUM_STREAMS = 2 * NUM_TAPS
TAP23 = 3                      # index of aggregator layer 23 among the 4 taps

# --- write input presets -----------------------------------------------------
WRITE_TAP23_HALF = "tap23_half"      # what every run before 2026-08-24 used
WRITE_ALL_SECOND = "all_second"
WRITE_ALL_FULL = "all_full"

WRITE_STREAMS = {
    WRITE_TAP23_HALF: (7,),
    WRITE_ALL_SECOND: (1, 3, 5, 7),
    WRITE_ALL_FULL: tuple(range(NUM_STREAMS)),
}

# --- raymap query presets ----------------------------------------------------
# How many independent attention reads the past-camera query gets. More query
# streams means each half-tap attends to the state on its own; fewer means they
# share one read and are separated afterwards by a linear map.
QUERY_PER_HALF = "per_half"    # V1: 8 queries/patch, one per half-tap
QUERY_PER_TAP = "per_tap"     # V2: 4 queries/patch, one per tap
QUERY_SINGLE = "single"       # V3: 1 query/patch, expanded to 4x2048 after the read

QUERY_STREAMS = {QUERY_PER_HALF: NUM_STREAMS, QUERY_PER_TAP: NUM_TAPS, QUERY_SINGLE: 1}


def split(stream: int) -> Tuple[int, int]:
    return stream // 2, stream % 2


def slice_for(stream: int) -> slice:
    _, half = split(stream)
    return slice(0, HALF) if half == 0 else slice(HALF, 2 * HALF)


def write_streams(preset: str) -> Tuple[int, ...]:
    if preset not in WRITE_STREAMS:
        raise ValueError(f"unknown write_input {preset!r}")
    return WRITE_STREAMS[preset]


def num_query_streams(mode: str) -> int:
    if mode not in QUERY_STREAMS:
        raise ValueError(f"unknown query_mode {mode!r}")
    return QUERY_STREAMS[mode]


def query_stream_ids(mode: str) -> Tuple[int, ...]:
    """Which cached stream each query stream stands for, in query order.

    `per_tap` and `single` stand for post-global halves: that is the half the
    token path refines and the only half the write has ever been fed.
    """
    if mode == QUERY_PER_HALF:
        return tuple(range(NUM_STREAMS))
    if mode == QUERY_PER_TAP:
        return tuple(2 * t + 1 for t in range(NUM_TAPS))
    if mode == QUERY_SINGLE:
        return (2 * TAP23 + 1,)
    raise ValueError(f"unknown query_mode {mode!r}")


def recall_targets(query_mode: str, write_preset: str) -> Tuple[Tuple[int, int], ...]:
    """(query_stream_index, cached_stream_id) pairs a recall loss may supervise.

    Only streams the write actually ingested are supervisable: the state cannot
    recall a stream it was never given, and training it to try teaches the read
    to invent rather than the write to retain. The intersection is taken in the
    query's own ordering so the caller can index its outputs directly.
    """
    written = write_streams(write_preset)
    n_q = num_query_streams(query_mode)
    order: Sequence[int] = query_stream_ids(query_mode)
    pairs = tuple((qi, s) for qi, s in enumerate(order) if s in written)
    if not pairs:
        raise ValueError(
            f"query_mode={query_mode!r} and write_input={write_preset!r} share no "
            f"stream, so a recall loss would have no reachable target "
            f"(query covers {tuple(order)}, write holds {written})")
    assert len(pairs) <= n_q
    return pairs
