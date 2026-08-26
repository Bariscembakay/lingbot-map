"""Summary memory: a recurrent cell between the aggregator and the heads.

    S_i         = WRITE(x_i_written, S_{i-1})
    refined x_i = READ (x_i,         S_{i-1})

Write and read run in parallel and both consume the **old** state, so the read
provably cannot see, through the state, the frame it is refining.

`x` is the aggregator's last-layer token stream (1024-d, after block group 23) --
not the 2048-d tap. The refined stream is re-concatenated with
`frame_intermediates[23]` to rebuild tap 23 for the heads.

Which frame the write ingests is set by `WriteSchedule`; see `schedule.py` for
why the disjoint lag is `sliding_window - 1` and not something rounder.
"""
from __future__ import annotations

from typing import Dict, List, Optional, Sequence, Tuple

import torch
import torch.nn as nn

from . import streams as ST
from .raymap import RaymapEncoder
from .read import ReadTransformer
from .schedule import DISJOINT, WriteSchedule
from .write import WriteTransformer


class SummaryMemory(nn.Module):
    def __init__(
        self,
        dim: int = 1024,
        num_slots: int = 512,
        num_heads: int = 16,
        write_layers: int = 2,
        read_layers: int = 2,
        mlp_ratio: float = 4.0,
        qk_norm: bool = True,
        write_residual_gate: bool = False,
        write_state_norm: bool = True,
        read_gate_init: float = 0.0,
        read_relative_residual: bool = True,
        query_gate_init: float = 1.0,
        position_mode: str = "none",
        refine_taps: Sequence[int] = (0, 1, 2, 3),
        share_read: bool = False,
        write_mode: str = DISJOINT,
        scale_frames: int = 8,
        sliding_window: int = 64,
        frozen_state: bool = False,
        num_freqs: int = 10,
        patch_size: int = 14,
        init_std: float = 1.0,
        write_input: str = ST.WRITE_TAP23_HALF,
        query_mode: str = ST.QUERY_SINGLE,
        patch_start_idx: int = 6,
    ):
        super().__init__()
        self.dim = dim
        self.num_slots = num_slots
        # Control arm: identical read transformer, state pinned at its init and
        # never written. Loss 1 can improve from a plain residual adapter on the
        # tokens alone, so without this arm an improvement proves nothing.
        self.frozen_state = frozen_state

        self.schedule = WriteSchedule(write_mode, scale_frames, sliding_window)
        # std 1.0, not the usual small init: the write ends in a LayerNorm, which
        # pins each slot to unit variance, so a small init would put step 0's state
        # on a different scale from every later step and the read would see two
        # regimes.
        self.state_init = nn.Parameter(torch.randn(num_slots, dim) * init_std)
        # Which of the 8 half-tap streams the write ingests, and how many
        # independent attention reads a past-camera query gets. Both are sweep
        # axes; `streams.py` owns the vocabulary so they cannot drift apart.
        self.write_input = write_input
        self.write_stream_ids = ST.write_streams(write_input)
        self.query_mode = query_mode
        self.query_stream_ids = ST.query_stream_ids(query_mode)
        self.write = WriteTransformer(dim, write_layers, num_heads, mlp_ratio,
                                      qk_norm, write_residual_gate, position_mode,
                                      write_state_norm,
                                      num_write_streams=len(self.write_stream_ids))
        self.refine_taps = tuple(refine_taps)
        self.share_read = share_read
        def mk_read():
            return ReadTransformer(dim, read_layers, num_heads, mlp_ratio,
                                   qk_norm, read_gate_init, read_relative_residual,
                                   query_gate_init)
        if share_read:
            shared = mk_read()
            self.readers = nn.ModuleDict({str(t): shared for t in self.refine_taps})
        else:
            self.readers = nn.ModuleDict({str(t): mk_read() for t in self.refine_taps})
        n_q = ST.num_query_streams(query_mode)
        self.patch_start_idx = patch_start_idx
        self.raymap = RaymapEncoder(dim, num_freqs=num_freqs, patch_size=patch_size,
                                    num_query_streams=n_q,
                                    patch_start_idx=patch_start_idx)
        # A hindsight decode needs the frozen head's full 4x2048 input, and the
        # reader emits 1024-d streams. `per_half` already has all 8 and needs no
        # parameters; the other two variants buy the missing width with a linear
        # map, whose weight starts at zero and whose bias is set to the cache's mean
        # tap -- so the decode begins at the mean tap's scale and centre instead of
        # at noise.
        if query_mode == ST.QUERY_PER_HALF:
            self.to_taps = None
        elif query_mode == ST.QUERY_PER_TAP:
            self.to_taps = nn.ModuleList(
                [nn.Linear(dim, 2 * dim) for _ in range(ST.NUM_TAPS)])
        else:
            self.to_taps = nn.ModuleList([nn.Linear(dim, ST.NUM_TAPS * 2 * dim)])
        if self.to_taps is not None:
            for lin in self.to_taps:
                nn.init.zeros_(lin.weight)
                nn.init.zeros_(lin.bias)
            # Zeros are only a placeholder: `set_query_init` replaces them with a
            # pass-through, because a zero weight makes `query_taps` constant and a
            # hindsight loss would then have exactly zero gradient to the state, the
            # reader and the write. V17 fails if this is ever left at zero.

    def new_state(self, batch_size: int = 1) -> torch.Tensor:
        """[B, N, D]. Reset at every clip start -- each clip is its own scene."""
        return self.state_init.unsqueeze(0).expand(batch_size, -1, -1)

    def step(
        self,
        state: torch.Tensor,
        tokens: Dict[int, torch.Tensor],
        write_tokens: Optional[torch.Tensor] = None,
        position: Optional[torch.Tensor] = None,
    ) -> Tuple[torch.Tensor, Dict[int, torch.Tensor]]:
        """One recurrent step.

        state:        [B, N, D]  = S_{i-1}
        tokens:       {tap: [B, P, D]} the query for each tap being refined
        write_tokens: [B, P, D]  the last-layer tokens of the frame the schedule
                                 says to ingest, or None when it writes nothing.
        Returns (S_i, {tap: refined tokens}).
        """
        refined = {t: self.readers[str(t)](tokens[t], state) for t in self.refine_taps}
        if self.frozen_state or write_tokens is None:
            return state, refined
        return self.write(state, write_tokens, position), refined

    @torch.no_grad()
    def set_query_init(self, stream_means: torch.Tensor) -> None:
        """stream_means: [8, dim] -- the cache's mean of each half-tap stream.

        Sets two things, and the second one is load-bearing:

        * the raymap's per-stream query embeddings, so a query enters the reader
          centred and scaled like a real half-tap;
        * the `to_taps` projection as a **pass-through plus mean**: the query stream
          standing for cached stream `s` is copied straight into slot `s`, and every
          slot the state cannot answer for is filled from the bias with that
          stream's mean.

        So at init the head receives the state's own answer where the state has one
        and the cache mean everywhere else -- on-manifold, finite loss, and every
        upstream parameter still receives gradient. A zero weight would give the
        same loss and no gradient at all (V17).
        """
        if stream_means.shape != (ST.NUM_STREAMS, self.dim):
            raise ValueError(f"expected [{ST.NUM_STREAMS}, {self.dim}], "
                             f"got {tuple(stream_means.shape)}")
        means = stream_means.to(self.state_init.dtype)
        self.raymap.set_stream_init(means[list(self.query_stream_ids)])
        if self.to_taps is None:
            return
        D = self.dim
        eye = torch.eye(D, dtype=means.dtype)
        if self.query_mode == ST.QUERY_PER_TAP:
            for k, lin in enumerate(self.to_taps):
                s = self.query_stream_ids[k]           # the half this query stands for
                half = s % 2
                lin.weight.zero_()
                lin.weight[half * D:(half + 1) * D].copy_(eye)
                lin.bias.zero_()
                lin.bias[(1 - half) * D:(2 - half) * D].copy_(means[2 * k + (1 - half)])
        else:
            lin = self.to_taps[0]
            s = self.query_stream_ids[0]
            lin.weight.zero_()
            lin.weight[s * D:(s + 1) * D].copy_(eye)
            lin.bias.copy_(means.reshape(-1))
            lin.bias[s * D:(s + 1) * D].zero_()

    def read_at_camera(self, state: torch.Tensor, pose_enc: torch.Tensor,
                       height: int, width: int) -> torch.Tensor:
        """Loss 2: query a past camera. Never updates the state.

        Returns [B, S, T, D] -- S query streams of T = patch_start_idx + P tokens.
        One reader handles the whole query as a single sequence (the design's "same
        read transformer"); the raymap's per-stream embeddings are what tell it
        which stream each token belongs to.
        """
        q = self.raymap(pose_enc, height, width)            # [B, S*T, D]
        refined = self.readers[str(self.refine_taps[-1])](q, state, query_path=True)
        s = len(self.query_stream_ids)
        return refined.view(refined.shape[0], s, -1, refined.shape[-1])

    def query_taps(self, refined: torch.Tensor) -> list:
        """[B, S, T, D] refined query streams -> the frozen head's 4-tap list.

        Each entry is [B, 1, T, 2*D], matching what `data.head_inputs` produces for
        the token path, so the frozen depth head is called identically either way.
        """
        B, s, T, D = refined.shape
        if self.query_mode == ST.QUERY_PER_HALF:
            taps = [torch.cat([refined[:, 2 * k], refined[:, 2 * k + 1]], dim=-1)
                    for k in range(ST.NUM_TAPS)]
        elif self.query_mode == ST.QUERY_PER_TAP:
            taps = [self.to_taps[k](refined[:, k]) for k in range(ST.NUM_TAPS)]
        else:
            wide = self.to_taps[0](refined[:, 0])            # [B, T, 4*2D]
            taps = list(wide.view(B, T, ST.NUM_TAPS, 2 * D).unbind(dim=2))
        return [x.unsqueeze(1) for x in taps]

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
