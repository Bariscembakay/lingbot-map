"""Drive the frozen camera head with a teacher cache, refining only the current frame.

The camera head consumes `aggregated_tokens_list[-1][:, :, 0]` -- token 0 of tap
23, exactly what the read refines -- and carries its own causal KV cache. If that
cache saw refined tokens it would become a second recurrence interleaved with the
state's, with its own BPTT path.

This bridge sidesteps that: the cache holds the **teacher's** values for every
frame < i, and only frame i's token is the refined one. Gradient reaches the read
through frame i alone, and nothing recurs.

Two properties of the frozen head make it cheap and exact:
  * its cache never evicts (the eviction guard is `shape[3] > 1`, and with one
    token per frame that is always False), so a single streaming pass leaves a
    complete in-order cache that can be sliced per step;
  * appends rebind the dict entry via `torch.cat` (`layers/attention.py:285`)
    rather than mutating in place, so per-step views into the teacher cache are
    safe to reuse.
"""
from __future__ import annotations

from typing import Dict, List

import torch


def _as_token_list(token: torch.Tensor) -> List[torch.Tensor]:
    """[B, S, C] -> the `aggregated_tokens_list` shape the head expects."""
    return [token.unsqueeze(2)]


def _fresh_cache(num_iterations: int, trunk_depth: int) -> List[Dict]:
    return [
        {"_skip_append": False,
         **{f"{kv}_{j}": None for j in range(trunk_depth) for kv in ("k", "v")}}
        for _ in range(num_iterations)
    ]


@torch.no_grad()
def build_teacher_cache(head, camera_tokens: torch.Tensor) -> tuple[List[Dict], torch.Tensor]:
    """Stream the whole clip's teacher camera tokens through the frozen head.

    camera_tokens: [L, C] teacher token 0 of tap 23, i.e. `taps[:, 3, 0, :]`.
    Returns (cache, pose_enc) where pose_enc [L, 9] is the teacher's own answer --
    useful as the exactness check in `validate.py`.
    """
    num_iter, depth = head.num_iterations, head.trunk_depth
    head.kv_cache = _fresh_cache(num_iter, depth)
    head.frame_idx = 0

    poses = []
    for i in range(camera_tokens.shape[0]):
        tok = camera_tokens[i].view(1, 1, -1)
        out = head(_as_token_list(tok), causal_inference=True, num_frame_per_block=1)
        poses.append(out[-1][0, 0])

    cache = head.kv_cache
    head.kv_cache = None
    head.frame_idx = 0
    return cache, torch.stack(poses)


def cache_prefix(cache: List[Dict], i: int) -> List[Dict]:
    """Views of the teacher cache truncated to frames [0, i). No copies."""
    out = []
    for it in cache:
        d = {"_skip_append": False}
        for key, val in it.items():
            if key == "_skip_append":
                continue
            d[key] = None if i == 0 else val[:, :, :i]
        out.append(d)
    return out


def pose_at(head, cache: List[Dict], i: int, camera_token: torch.Tensor) -> torch.Tensor:
    """Predict frame i's pose from `camera_token`, with frames < i taken from the teacher.

    camera_token: [B, C]. Returns [B, 9], differentiable w.r.t. camera_token only.
    """
    head.kv_cache = cache_prefix(cache, i)
    # frame_idx drives `is_scale_frames`, which switches the head to batch-mode
    # attention; it must be nonzero for a streaming step or frame i would be
    # treated as an anchor.
    head.frame_idx = i
    try:
        out = head(_as_token_list(camera_token.unsqueeze(1)),
                   causal_inference=True, num_frame_per_block=1)
    finally:
        head.kv_cache = None
        head.frame_idx = 0
    return out[-1][:, 0]
