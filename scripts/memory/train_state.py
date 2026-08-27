#!/usr/bin/env python3
"""Train the CUT3R state on lingbot-map tokens, with a recall-only objective.

Design record: `.agents/spatial_memory_design.md`. The loop in one line: roll the
clip through the write, and at every step query the state at several cameras we
have already visited, scoring the answers against those frames' ground truth.

Two properties this file must not lose:

* **No detach anywhere inside a clip.** A probe at t must credit the write at q,
  which is t-q recurrence steps back. Truncating that is what makes CUT3R's own
  TBPTT unable to train recall, whatever objective it is given.
* **The probe never touches the taps.** Its only path to frame q's content is
  the state. Validator V6 asserts it; do not add a shortcut here.
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch

sys.path.insert(0, str(Path(__file__).resolve().parents[2]))

from lingbot_map.memory.cache_format import ClipCache  # noqa: E402
from lingbot_map.memory.cut3r_state import StateMemory, load_cut3r_weights  # noqa: E402
from lingbot_map.memory.probe_data import build_raymap, gt_pointmaps  # noqa: E402
from lingbot_map.memory.recall_loss import probe_loss  # noqa: E402

TAP23 = 3   # index into the cache's 4 tap layers (4, 11, 17, 23)


class Clip:
    """One cached clip, subsampled, with everything a probe needs on device."""

    def __init__(self, path: Path, subsample: int, device, max_frames: int | None):
        c = ClipCache(path)
        n = len(c)
        idx = list(range(0, n, subsample))
        if max_frames:
            idx = idx[:max_frames]
        self.idx = idx
        self.meta = c.meta
        self.device = device
        # fp16 on disk; cast on GPU rather than single-threaded on the CPU.
        self.taps = torch.from_numpy(
            np.ascontiguousarray(c.taps[idx][:, TAP23])).to(device)
        self.gt_depth = torch.from_numpy(
            np.ascontiguousarray(c.gt_depth[idx])).to(device).float()
        self.gt_c2w = torch.from_numpy(c.gt_c2w[idx]).to(device).float()
        self.K = torch.from_numpy(c.gt_intrinsics[idx]).to(device).float()
        self.h, self.w = c.meta.height, c.meta.width
        self.patch_hw = (c.meta.patch_h, c.meta.patch_w)

    def __len__(self):
        return len(self.idx)

    def tap(self, t):
        return self.taps[t][None].float()

    def probe_inputs(self, qs, convention):
        q = torch.tensor(qs, device=self.device)
        c2w0 = self.gt_c2w[0][None]
        rm = build_raymap(self.K[q], self.gt_c2w[q], c2w0, self.h, self.w, convention)
        xs, xw, valid = gt_pointmaps(self.gt_depth[q], self.K[q], self.gt_c2w[q], c2w0)
        return rm, xs, xw, valid


def sample_queries(rng, t, n_past, probe_current):
    """`min(n_past, t)` distinct past cameras, plus the current one if enabled."""
    qs = list(rng.choice(t, size=min(n_past, t), replace=False)) if t > 0 else []
    if probe_current:
        qs.append(t)
    return [int(q) for q in qs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, type=Path)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--cut3r-ckpt",
                    default="/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--n-past", type=int, default=4)
    # Probe every k-th frame. NOT an optional axis: the whole clip's graph is
    # retained (no detach, by design), and each probe pass holds ~180 MB, so
    # probing every frame costs ~180 MB x frames. Measured: 48, 96 and 160
    # frames ALL hit 44 GB on an a6000 -- the ceiling is reached before the
    # clip ends, which is why shortening the clip never helped. At every-frame
    # density a 160-frame clip needs ~144 GB, so it does not fit an H200 either.
    ap.add_argument("--probe-every", type=int, default=1)
    ap.add_argument("--probe-current", default="on", choices=["on", "off"])
    ap.add_argument("--raymap-convention", default="cut3r", choices=["cut3r", "true"])
    # Controls. no-write is the decisive one: if the probe still works with the
    # state pinned to s0, the scene is in the weights and not in the memory.
    ap.add_argument("--no-write", action="store_true")
    ap.add_argument("--zero-state", action="store_true")
    ap.add_argument("--no-grad-ckpt", action="store_true")
    # bf16 for the decoder. Checkpointing stores block INPUTS, and in fp32 that
    # is ~65 MB per pass: 576 passes at 96 frames is ~37 GB, which is what
    # OOM-killed jobs 753366/753437 on a 44 GB a6000. CUT3R trains every stage
    # with amp=1. The DPT heads stay fp32 regardless -- ProbeHead wraps them in
    # autocast(enabled=False), matching lingbot-map's own head handling.
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--wandb", default="online", choices=["online", "offline", "off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--viz-every", type=int, default=500)
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    run = None
    if args.wandb != "off":
        try:
            import wandb
            # Run dirs live on persistent /group, not the run's /scratch out dir.
            wdir = Path("/group/compact-3dmem/wandb")
            if not (wdir.is_dir() and os.access(wdir, os.W_OK)):
                wdir = args.out
            run = wandb.init(project="spatial_memory", name=args.out.name,
                             mode=args.wandb, config=vars(args), dir=str(wdir))
        except Exception as e:  # e.g. no ~/.netrc on msp3 -- never kill the run
            print(f"[wandb] disabled: {e}", flush=True)

    clips = [Clip(p, args.subsample, device, args.max_frames) for p in args.clips]
    print(f"[data] {len(clips)} clip(s), {len(clips[0])} frames each, "
          f"{clips[0].h}x{clips[0].w}, patch grid {clips[0].patch_hw}", flush=True)

    model = StateMemory(patch_size=clips[0].meta.patch_size if hasattr(
        clips[0].meta, "patch_size") else 14,
        grad_ckpt=not args.no_grad_ckpt).to(device)
    load_cut3r_weights(model, args.cut3r_ckpt)
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_train/1e6:.1f} M trainable", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    hist = []
    t0 = time.time()
    for step in range(args.updates):
        clip = clips[rng.integers(len(clips))]
        state, spos = model.init_state(1, device)
        total, parts, nprobe = 0.0, {"l21_self": 0.0, "l21_world": 0.0}, 0
        state_norms = []

        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(args.amp == "bf16" and device.type == "cuda"))
        for t in range(len(clip)):
            with amp:
                if not args.no_write:
                    state = model.write(state, spos, clip.tap(t), clip.patch_hw)
            state_norms.append(float(state.detach().norm()))

            if t % args.probe_every:
                continue
            qs = sample_queries(rng, t, args.n_past, args.probe_current == "on")
            if not qs:
                continue
            rm, xs, xw, valid = clip.probe_inputs(qs, args.raymap_convention)
            st = torch.zeros_like(state) if args.zero_state else state
            with amp:
                # Batch the queries: they all read the same state, so one pass.
                out = model.probe(st.expand(len(qs), -1, -1),
                                  spos.expand(len(qs), -1, -1), rm, (clip.h, clip.w))
            # Heads already returned fp32; keep the loss there too.
            out = {k: v.float() for k, v in out.items()}
            loss, p = probe_loss(out, xs, xw, valid)
            total = total + loss
            for k in parts:
                parts[k] += p[k]
            nprobe += 1

        loss = total / max(1, nprobe)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        sched.step()

        rec = {"step": step, "loss": float(loss.detach()),
               "l21_self": parts["l21_self"] / max(1, nprobe),
               "l21_world": parts["l21_world"] / max(1, nprobe),
               "grad_norm": gnorm, "state_norm": float(np.mean(state_norms)),
               "peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                           if device.type == "cuda" else 0.0),
               "sec": time.time() - t0}
        hist.append(rec)
        if run is not None:
            run.log(rec, step=step)
        if step % args.log_every == 0:
            print(f"[{step:5d}] loss {rec['loss']:8.4f} | L21 self "
                  f"{rec['l21_self']:.4f} world {rec['l21_world']:.4f} | "
                  f"|g| {gnorm:7.3f} | |s| {rec['state_norm']:8.1f} | "
                  f"{rec['peak_gb']:5.1f}GB | {rec['sec']:6.1f}s", flush=True)
            (args.out / "history.json").write_text(json.dumps(hist))
        if step and step % args.save_every == 0:
            torch.save({"model": model.state_dict(), "step": step,
                        "args": vars(args)}, args.out / "last.pt")
        if step % args.viz_every == 0:
            dump_viz(model, clips[0], args, device, step)

    torch.save({"model": model.state_dict(), "step": args.updates,
                "args": vars(args)}, args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(hist))
    dump_viz(model, clips[0], args, device, args.updates)
    if run is not None:
        run.finish()
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


@torch.no_grad()
def dump_viz(model, clip, args, device, step):
    """Everything a viewer needs, at a spread of lags.

    Saved as one npz per checkpoint so a visualisation can be rebuilt offline
    without a GPU: predicted and GT pointmaps in the world frame, confidences,
    ray origins, and the lag each probe was taken at.
    """
    model.eval()
    T = len(clip) - 1
    state, spos = model.init_state(1, device)
    for t in range(len(clip)):
        if not args.no_write:
            state = model.write(state, spos, clip.tap(t), clip.patch_hw)
    lags = [l for l in (0, 1, 5, 20, 60, 120, T) if l <= T]
    qs = [T - l for l in lags]
    rm, xs, xw, valid = clip.probe_inputs(qs, args.raymap_convention)
    st = torch.zeros_like(state) if args.zero_state else state
    out = model.probe(st.expand(len(qs), -1, -1), spos.expand(len(qs), -1, -1),
                      rm, (clip.h, clip.w))
    d = args.out / "viz"
    d.mkdir(exist_ok=True)
    np.savez_compressed(
        d / f"probe_{step:06d}.npz",
        lags=np.array(lags), qs=np.array(qs), t=T,
        pred_world=out["pts3d_in_other_view"].float().cpu().numpy(),
        pred_self=out["pts3d_in_self_view"].float().cpu().numpy(),
        conf_world=out["conf"].float().cpu().numpy(),
        conf_self=out["conf_self"].float().cpu().numpy(),
        gt_world=xw.float().cpu().numpy(), gt_self=xs.float().cpu().numpy(),
        valid=valid.cpu().numpy(), ray_o=rm[:, :3].float().cpu().numpy(),
        scene=clip.meta.scene, frame_ids=np.array(clip.idx),
    )
    model.train()


if __name__ == "__main__":
    raise SystemExit(main())
