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

from typing import Dict, Optional, Sequence, Tuple

import torch
import torch.nn as nn

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
        self.write = WriteTransformer(dim, write_layers, num_heads, mlp_ratio,
                                      qk_norm, write_residual_gate, position_mode,
                                      write_state_norm)
        self.refine_taps = tuple(refine_taps)
        self.share_read = share_read
        def mk_read():
            return ReadTransformer(dim, read_layers, num_heads, mlp_ratio,
                                   qk_norm, read_gate_init, read_relative_residual)
        if share_read:
            shared = mk_read()
            self.readers = nn.ModuleDict({str(t): shared for t in self.refine_taps})
        else:
            self.readers = nn.ModuleDict({str(t): mk_read() for t in self.refine_taps})
        self.raymap = RaymapEncoder(dim, num_freqs=num_freqs, patch_size=patch_size)

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

    def read_at_camera(self, state: torch.Tensor, pose_enc: torch.Tensor,
                       height: int, width: int) -> torch.Tensor:
        """Loss 2: query a past or novel camera. Never updates the state."""
        tap = self.refine_taps[-1]
        return self.readers[str(tap)](self.raymap(pose_enc, height, width), state)

    def num_trainable(self) -> int:
        return sum(p.numel() for p in self.parameters() if p.requires_grad)
