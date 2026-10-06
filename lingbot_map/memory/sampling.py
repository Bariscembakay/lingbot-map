"""CUT3R-style frame sampling for one training sequence.

A port of CUT3R's `ScanNetpp_Multi._get_views` and
`BaseMultiViewDataset.get_seq_from_start_id` / `blockwise_shuffle`
(~/CUT3R/src/dust3r/datasets/{scannetpp.py,base/base_multiview_dataset.py}),
with the same constants, over clip-frame indices `0..n-1`. Our clip frames are
already 19-20 raw frames apart, which is about as sparse as CUT3R's processed
ScanNet++ frames, so its gaps (max_interval=3) are used as-is in clip frames.

One deliberate deviation: CUT3R's fallback branch draws from a precomputed
covisibility group; we have none, so it draws from a random window of the clip.
"""
import itertools

import numpy as np


def _blockwise_shuffle(x, rng, block_shuffle):
    if block_shuffle is None:
        return rng.permutation(x).tolist()
    blocks = [x[i:i + block_shuffle] for i in range(0, len(x), block_shuffle)]
    return [v for b in blocks for v in rng.permutation(b).tolist()]


def get_seq_from_start_id(num_views, pos_ref, n, rng, min_interval=1,
                          max_interval=25, video_prob=0.5, fix_interval_prob=0.5,
                          block_shuffle=None):
    """Positions in `range(n)` starting at `pos_ref`; (positions, is_video)."""
    all_possible_pos = np.arange(pos_ref, n)
    remaining_sum = n - 1 - pos_ref

    if remaining_sum >= num_views - 1:
        if remaining_sum == num_views - 1:
            return [pos_ref + i for i in range(num_views)], True
        max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1))
        intervals = [rng.choice(range(min_interval, max_interval + 1))
                     for _ in range(num_views - 1)]
        if rng.random() < video_prob:
            if rng.random() < fix_interval_prob:
                fixed = rng.choice(range(
                    1, min(remaining_sum // (num_views - 1) + 1, max_interval + 1)))
                intervals = [fixed] * (num_views - 1)
            is_video = True
        else:
            is_video = False
        pos = list(itertools.accumulate([pos_ref] + intervals))
        pos = [p for p in pos if p < n]
        cand = [p for p in all_possible_pos if p not in pos]
        pos = pos + rng.choice(cand, num_views - len(pos), replace=False).tolist()
        pos = sorted(pos) if is_video else _blockwise_shuffle(pos, rng, block_shuffle)
    else:
        # Fewer frames left than views: revisit (repeat) a shorter sequence.
        uniq_num = remaining_sum
        new_pos_ref = rng.choice(np.arange(pos_ref + 1))
        new_remaining_sum = n - 1 - new_pos_ref
        new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
        new_intervals = [rng.choice(range(1, new_max_interval + 1))
                         for _ in range(uniq_num - 1)]
        revisit_random = rng.random()
        video_random = rng.random()
        if rng.random() < fix_interval_prob and video_random < video_prob:
            fixed = rng.choice(range(1, new_max_interval + 1))
            new_intervals = [fixed] * (uniq_num - 1)
        pos = list(itertools.accumulate([new_pos_ref] + new_intervals))
        is_video = False
        if revisit_random < 0.5 or video_prob == 1.0:
            is_video = video_random < video_prob
            if not is_video:
                pos = _blockwise_shuffle(pos, rng, block_shuffle)
            k = num_views // uniq_num
            pos = pos * k + pos[:num_views - len(pos) * k]
        elif revisit_random < 0.9:
            pos = rng.choice(pos, num_views, replace=True).tolist()
        else:
            pos = sorted(rng.choice(pos, num_views, replace=True).tolist())
    assert len(pos) == num_views
    return [int(p) for p in pos], is_video


def cut3r_views(rng, n, num_views, allow_repeat=False, max_interval=3):
    """Clip-frame indices for one sequence, as CUT3R samples ScanNet++."""
    cut_off = num_views if not allow_repeat else max(num_views // 3, 3)
    n_starts = n - cut_off + 1
    rand_val = rng.random()
    if rand_val < 0.7 and n_starts > 0:
        start = int(rng.choice(n_starts))
        pos, _ = get_seq_from_start_id(
            num_views, start, n, rng, max_interval=max_interval,
            video_prob=0.8, fix_interval_prob=0.5, block_shuffle=16)
        return pos
    # Ordered video with varying intervals; shuffled when rand_val > 0.75.
    max_id = min(n, int(num_views * (2 + 2 * rng.random())))
    w0 = int(rng.integers(0, n - max_id + 1))
    idx = sorted((w0 + rng.permutation(max_id)[:num_views]).tolist())
    if rand_val > 0.75:
        idx = rng.permutation(idx).tolist()
    return [int(i) for i in idx]
