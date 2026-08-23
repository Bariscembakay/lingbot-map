#!/usr/bin/env python3
"""Measure the real cost of a training step, and project a full run.

Replaces the paper FLOP estimate. Reports per-step and per-update wall clock,
peak VRAM, and where the time goes -- the recurrence, the frozen heads, or the
backward -- so the next knob is chosen from a measurement rather than a guess.
"""
from __future__ import annotations

import argparse
import sys
import time
from pathlib import Path

import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory import camera_bridge as cb                  # noqa: E402
from lingbot_map.memory import frozen, losses as L                  # noqa: E402
from lingbot_map.memory.data import ClipReader, find_clips          # noqa: E402
from lingbot_map.memory.model import SummaryMemory                  # noqa: E402


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--cache", required=True)
    ap.add_argument("--heads", default="/group/compact-3dmem/checkpoints/lingbot-map/frozen_heads.pt")
    ap.add_argument("--refine-taps", default="0,1,2,3")
    ap.add_argument("--num-slots", type=int, default=512)
    ap.add_argument("--write-layers", type=int, default=2)
    ap.add_argument("--read-layers", type=int, default=2)
    ap.add_argument("--unroll", type=int, default=32)
    ap.add_argument("--head-every", type=int, default=4)
    ap.add_argument("--steps", type=int, default=96, help="steps to time (after warmup)")
    ap.add_argument("--warmup", type=int, default=16)
    ap.add_argument("--with-camera", action="store_true")
    ap.add_argument("--target-updates", type=int, default=20000)
    ap.add_argument("--device", default="cuda")
    a = ap.parse_args()

    device = torch.device(a.device)
    taps = tuple(int(t) for t in a.refine_taps.split(","))
    use_depth = any(t != 3 for t in taps)

    reader = ClipReader(find_clips(a.cache)[0], device=device)
    H, W = reader.hw
    depth_head, camera_head = frozen.load_frozen(a.heads, device, a.with_camera)
    mem = SummaryMemory(dim=1024, num_slots=a.num_slots,
                        write_layers=a.write_layers, read_layers=a.read_layers,
                        refine_taps=taps).to(device)
    opt = torch.optim.AdamW(mem.parameters(), lr=1e-4)

    tcache = tc2w = None
    if a.with_camera:
        tcache, _ = cb.build_teacher_cache(camera_head, reader.teacher_camera_tokens())
        with torch.no_grad():
            tc2w = L.pose_enc_to_c2w(
                torch.from_numpy(reader.cache.pose_enc).to(device), (H, W))

    torch.cuda.reset_peak_memory_stats()
    state = mem.new_state(1)
    pending, t0, n_upd, n_sup = [], None, 0, 0
    total = a.warmup + a.steps

    for i in range(total):
        if i == a.warmup:
            torch.cuda.synchronize(); t0 = time.time()
        x = reader.token_dict(i, taps)
        state, refined = mem.step(state, x, write_tokens=x[taps[-1]])
        if i % a.head_every == 0:
            parts = {}
            head_in = (reader.head_inputs(i, refined) if use_depth
                       else [reader.rebuild_tap(i, 3, refined[3]).unsqueeze(1)])
            if use_depth:
                d, c = depth_head(head_in, images=torch.zeros(1, 1, 3, H, W, device=device),
                                  patch_start_idx=reader.meta.patch_start_idx)
                parts.update(L.depth_loss(d[0, ..., 0], c[0], reader.gt_depth(i)))
            if a.with_camera:
                pose = cb.pose_at(camera_head, tcache, i, head_in[-1][0, :, 0].float())
                pc2w = L.pose_enc_to_c2w(pose, (H, W))
                parts.update(L.abs_pose_loss(pc2w, reader.gt_c2w([i])))
                lo = max(0, i - 63)
                win = torch.cat([tc2w[lo:i], pc2w], dim=0)
                parts.update(L.rel_pose_loss(win, reader.gt_c2w(slice(lo, i + 1)),
                                             anchor=win.shape[0] - 1))
            pending.append(L.total_loss(parts))
            n_sup += 1
        if (i + 1) % a.unroll == 0 and pending:
            loss = torch.stack(pending).mean()
            opt.zero_grad(set_to_none=True); loss.backward()
            torch.nn.utils.clip_grad_norm_(mem.parameters(), 1.0)
            opt.step(); state = state.detach(); pending = []
            n_upd += 1

    torch.cuda.synchronize()
    dt = time.time() - t0
    peak = torch.cuda.max_memory_allocated() / 1e9
    per_step = dt / a.steps
    L_clip = len(reader)
    upd_per_clip = max(n_upd, 1) / a.steps * L_clip
    clip_pass = per_step * L_clip
    per_upd = dt / max(n_upd, 1)
    hours = per_upd * a.target_updates / 3600

    print(f"config           taps={taps} slots={a.num_slots} unroll={a.unroll} "
          f"head_every={a.head_every} camera={a.with_camera}")
    print(f"trainable        {mem.num_trainable()/1e6:.1f} M")
    print(f"timed            {a.steps} steps, {n_sup} supervised, {n_upd} updates in {dt:.1f}s")
    print(f"per step         {per_step*1000:.0f} ms")
    print(f"per update       {per_upd:.2f} s")
    print(f"clip pass ({L_clip}f) {clip_pass:.1f} s   ~{upd_per_clip:.0f} updates")
    print(f"peak VRAM        {peak:.1f} GB (card 143.8)")
    print(f"=> {a.target_updates} updates  {hours:.1f} h"
          f"   ({a.target_updates/max(upd_per_clip,1):.0f} clip passes)")


if __name__ == "__main__":
    main()
