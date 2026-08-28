#!/usr/bin/env python3
"""Train the CUT3R state on lingbot-map tokens, with a recall-only objective.

Design record: `.agents/spatial_memory_design.md`. The loop in one line: roll the
clip through the write, and at every step query the state at several cameras we
have already visited, scoring the answers against those frames' ground truth.

Two properties this file must not lose:

* **No detach inside a clip by default.** A probe at t then credits the write
  at q, t-q recurrence steps back. `--tbptt K` deliberately truncates this to
  test whether local credit suffices (design doc: "Full BPTT vs truncated") --
  it detaches every K writes and backwards each window at its boundary, which
  also frees that window's probe activations, so memory stops scaling with
  probe count.
* **The probe never touches the taps.** Its only path to frame q's content is
  the state. Validator V6 asserts it; do not add a shortcut here.
"""
from __future__ import annotations

import argparse
import glob as globmod
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
from lingbot_map.memory.probe_data import (  # noqa: E402
    build_raymap, gt_pointmaps, relative_c2w, true_rays)
from lingbot_map.memory.recall_loss import probe_loss  # noqa: E402

TAP23 = 3   # index into the cache's 4 tap layers (4, 11, 17, 23)


class Clip:
    """One cached clip, subsampled, with everything a probe needs on device."""

    def __init__(self, path: Path, subsample: int, device, max_frames: int | None,
                 taps: str = "23"):
        c = ClipCache(path)
        n = len(c)
        idx = list(range(0, n, subsample))
        if max_frames:
            idx = idx[:max_frames]
        self.idx = idx
        self.meta = c.meta
        self.device = device
        # fp16 on disk; cast on GPU rather than single-threaded on the CPU.
        if taps == "all":
            # channel-concat the four taps (4/11/17/23) -> 8192-d; sweep axis (b).
            # CPU-resident: at 32+8 clips this is ~105 GB, which no GPU holds --
            # consumers move one frame's stack to the device per step (~66 MB).
            t4 = torch.from_numpy(np.ascontiguousarray(c.taps[idx]))
            self.taps = t4.permute(0, 2, 1, 3).reshape(t4.shape[0], t4.shape[2], -1)
        else:
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
        return self.taps[t][None].to(self.device).float()

    def probe_inputs(self, qs, convention, anchor: int = 0):
        """qs are ABSOLUTE frame indices; `anchor` is the stream's first frame
        (the coordinate origin), so a random training window anchors at its own
        start rather than the clip's."""
        q = torch.tensor(qs, device=self.device)
        c2w0 = self.gt_c2w[anchor][None]
        rm = build_raymap(self.K[q], self.gt_c2w[q], c2w0, self.h, self.w, convention)
        xs, xw, valid = gt_pointmaps(self.gt_depth[q], self.K[q], self.gt_c2w[q], c2w0)
        return rm, xs, xw, valid

    def probe_rays(self, qs, anchor: int = 0, unit: bool = True):
        q = torch.tensor(qs, device=self.device)
        return true_rays(self.K[q], self.gt_c2w[q], self.gt_c2w[anchor][None],
                         self.h, self.w, unit=unit)


def expand_paths(paths):
    """Expand wildcard clip args in-process: a /data glob cannot expand at
    submit time (the registry mounts only after `dataset pull` on the node)."""
    out = []
    for p in paths:
        sp = str(p)
        if any(ch in sp for ch in "*?["):
            out.extend(Path(x) for x in sorted(globmod.glob(sp)))
        else:
            out.append(Path(sp))
    return out


def run_probe(model, st, spos_q, rm, hw, rays=None):
    """Probe with standard output keys for either head type.

    raydepth reconstructs `o + s*d` on TRUE rays and derives the self frame by
    the (known) query pose -- one geometry serves both loss terms, which is the
    point: a rigid map preserves L2, so the world term adds registration
    supervision without a second head to memorise in."""
    out = model.probe(st, spos_q, rm, hw)
    if getattr(model, "head_type", "dpt") not in ("raydepth", "lingbot", "smallread"):
        return {k: v.float() for k, v in out.items()}
    o, d, c2w = rays
    pts_w = o + out["ray_depth"].float().unsqueeze(-1) * d
    R, t = c2w[:, :3, :3], c2w[:, :3, 3]
    pts_s = torch.einsum("bji,bhwj->bhwi", R, pts_w - t[:, None, None, :])
    c = out["conf"].float()
    return {"pts3d_in_self_view": pts_s, "conf_self": c,
            "pts3d_in_other_view": pts_w, "conf": c}


def sample_queries(rng, t, n_past, probe_current):
    """`min(n_past, t)` distinct past cameras, plus the current one if enabled."""
    qs = list(rng.choice(t, size=min(n_past, t), replace=False)) if t > 0 else []
    if probe_current:
        qs.append(t)
    return [int(q) for q in qs]


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, type=Path)
    ap.add_argument("--val-clips", nargs="*", type=Path, default=[])
    ap.add_argument("--val-every", type=int, default=200)
    ap.add_argument("--out", required=True, type=Path)
    # Warm-start for curriculum stages (frames 16 -> 32 -> ...): loads a
    # checkpoint saved by this script, AFTER the CUT3R weights.
    ap.add_argument("--init-from", type=Path, default=None)
    ap.add_argument("--cut3r-ckpt",
                    default="/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth")
    ap.add_argument("--updates", type=int, default=2000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    # Stream length per update: a RANDOM window of the clip, anchored at its
    # own first frame. Curriculum starts short (CUT3R trains 4 -> 64 views the
    # same way) and later stages re-launch longer with --init-from.
    ap.add_argument("--frames", type=int, default=16)
    # Scenes per update. Memorisation gradients are scene-specific and cancel
    # across the batch; the read-the-state gradient is common and adds.
    ap.add_argument("--batch", type=int, default=8)
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
    # (a) tap 23 only vs (b) all four channel-concat; design doc "Sweep axes".
    ap.add_argument("--taps", default="23", choices=["23", "all"])
    # dpt = CUT3R's two DPT heads. raydepth = one scalar ray distance on true
    # rays, ~0.8 M params -- the read-capacity axis at its head end.
    ap.add_argument("--head", default="dpt", choices=["dpt", "raydepth", "lingbot", "smallread"])
    # State capacity axis. Multiples of 768 tile the loaded prior (see
    # load_cut3r_weights); the decoders are length-agnostic over state tokens.
    ap.add_argument("--state-tokens", type=int, default=768)
    # Controls. no-write is the decisive one: if the probe still works with the
    # state pinned to s0, the scene is in the weights and not in the memory.
    ap.add_argument("--no-write", action="store_true")
    # K > 0: detach every K writes, backward per window (gradient
    # accumulation; still one optimizer step per clip). 0 = full BPTT.
    # Default 8 since 2026-08-27: loss curves match full BPTT through the
    # early overfit while memory stops scaling with probe count; the lag
    # sweep at completion is the standing referee for the choice.
    ap.add_argument("--tbptt", type=int, default=8)
    ap.add_argument("--zero-state", action="store_true")
    # s0 is 590k trainable params sitting exactly where the memory should be --
    # in the no-write control it IS the storage. Freezing it removes that
    # memorisation channel (posterior-collapse discussion, 2026-08-28).
    ap.add_argument("--freeze-s0", action="store_true")
    # Write-only learning: freeze the whole probe path (raymap encoder, read
    # blocks, head) AND s0. Only meaningful with --init-from a checkpoint whose
    # read is trained -- a frozen random head is a dead path.
    ap.add_argument("--freeze-read", action="store_true")
    # Restore the write stacks to CUT3R init (fresh in_proj) AFTER --init-from,
    # so the read stays trained while the write starts over.
    ap.add_argument("--reinit-write", action="store_true")
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

    clips = [Clip(p, args.subsample, device, args.max_frames, args.taps)
             for p in expand_paths(args.clips)]
    val_clips = [Clip(p, args.subsample, device, args.max_frames, args.taps)
                 for p in expand_paths(args.val_clips)]
    assert len({(c.h, c.w, c.patch_hw) for c in clips + val_clips}) == 1
    print(f"[data] {len(clips)} train / {len(val_clips)} val clip(s), "
          f"{len(clips[0])} frames each, {clips[0].h}x{clips[0].w}, "
          f"patch grid {clips[0].patch_hw}", flush=True)

    model = StateMemory(patch_size=clips[0].meta.patch_size if hasattr(
        clips[0].meta, "patch_size") else 14,
        tap_dim=clips[0].taps.shape[-1], state_tokens=args.state_tokens,
        grad_ckpt=not args.no_grad_ckpt, head_type=args.head).to(device)
    load_cut3r_weights(model, args.cut3r_ckpt)
    if args.init_from:
        sd = torch.load(args.init_from, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        print(f"[init] warm-started from {args.init_from} (step {sd.get('step')})",
              flush=True)
    if args.reinit_write:
        raw = torch.load(args.cut3r_ckpt, map_location="cpu", weights_only=False)
        sd = raw["model"] if "model" in raw else raw
        sd = {k[7:] if k.startswith("module.") else k: v for k, v in sd.items()}
        own = model.state_dict()
        w = {k: v for k, v in sd.items()
             if k.split(".")[0] in ("dec_blocks", "dec_blocks_state",
                                    "dec_norm_state")
             and k in own and own[k].shape == v.shape}
        model.load_state_dict(w, strict=False)
        model.in_proj.reset_parameters()
        print(f"[reinit-write] {len(w)} tensors restored to CUT3R; in_proj reset",
              flush=True)
    if args.freeze_read:
        mods = [model.raymap, model.head, model.register_tokens,
                model.decoder_embed_state]
        if hasattr(model, "read_blocks"):
            mods += [model.read_blocks, model.read_norm]
        for m in mods:
            m.requires_grad_(False)
        print("[freeze-read] probe path + s0 frozen; write-only training",
              flush=True)
    if args.freeze_s0:
        model.register_tokens.requires_grad_(False)
        model.decoder_embed_state.requires_grad_(False)
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    print(f"[model] {n_train/1e6:.1f} M trainable", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    hist = []
    B = min(args.batch, len(clips))
    N = args.frames
    hw = (clips[0].h, clips[0].w)
    t0 = time.time()
    for step in range(args.updates):
        sel = [clips[i] for i in rng.choice(len(clips), size=B, replace=False)]
        starts = [int(rng.integers(0, max(1, len(c) - N + 1))) for c in sel]
        state, spos = model.init_state(B, device)
        parts, nprobe = {"l21_self": 0.0, "l21_world": 0.0}, 0
        # `window` holds the not-yet-backwarded loss; with tbptt it is settled
        # at every boundary, otherwise once at clip end. Normalise by the stop
        # count (known upfront) so both modes optimise the same objective.
        window, loss_sum = 0.0, 0.0
        frames_since_cut = 0
        n_stops = sum(1 for t in range(N)
                      if t % args.probe_every == 0
                      and (t > 0 or args.probe_current == "on"))
        state_norms = []
        opt.zero_grad(set_to_none=True)

        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(args.amp == "bf16" and device.type == "cuda"))
        for t in range(N):
            tap = torch.stack([c.taps[s0 + t] for c, s0 in zip(sel, starts)]
                              ).to(device).float()
            with amp:
                if not args.no_write:
                    state = model.write(state, spos, tap, clips[0].patch_hw)
            state_norms.append(float(state.detach().norm(dim=(1, 2)).mean()))

            if t % args.probe_every == 0:
                # Same local t for every scene, so nq is equal across the batch
                # and the B*nq queries run as one probe pass.
                qs0 = sample_queries(rng, t, args.n_past, args.probe_current == "on")
                if qs0:
                    nq = len(qs0)
                    rms, xss, xws, vs, rays = [], [], [], [], []
                    for c, s0 in zip(sel, starts):
                        qs = sample_queries(rng, t, args.n_past,
                                            args.probe_current == "on")
                        aq = [s0 + q for q in qs]
                        rm, xs, xw, valid = c.probe_inputs(
                            aq, args.raymap_convention, anchor=s0)
                        rms.append(rm); xss.append(xs); xws.append(xw); vs.append(valid)
                        if args.head in ("raydepth", "lingbot", "smallread"):
                            rays.append(c.probe_rays(
                                aq, anchor=s0, unit=args.head != "lingbot"))
                    rm = torch.cat(rms); xs = torch.cat(xss)
                    xw = torch.cat(xws); valid = torch.cat(vs)
                    ray3 = tuple(torch.cat(z) for z in zip(*rays)) if rays else None
                    st = torch.zeros_like(state) if args.zero_state else state
                    with amp:
                        out = run_probe(model, st.repeat_interleave(nq, 0),
                                        spos[:1].expand(B * nq, -1, -1), rm, hw,
                                        ray3)
                    loss, p = probe_loss(out, xs, xw, valid)
                    window = window + loss
                    for k in parts:
                        parts[k] += p[k]
                    nprobe += 1

            # Cut only AT a probe stop, at the first stop >= K frames since
            # the last cut. Cutting mid-stride orphans the writes after the
            # window's last stop: with stops at t%4==0 and cuts at t%8==7,
            # frames 8w+5..8w+7 provably received zero gradient (measured:
            # dead writes [5,6,7] per window). Ending every window on a stop
            # keeps all writes supervised and the stop schedule identical to
            # the full-BPTT baseline.
            frames_since_cut += 1
            if (args.tbptt and t % args.probe_every == 0
                    and frames_since_cut >= args.tbptt and t + 1 < N):
                if torch.is_tensor(window):
                    (window / max(1, n_stops)).backward()
                    loss_sum += float(window.detach())
                    window = 0.0
                state = state.detach()
                frames_since_cut = 0

        if torch.is_tensor(window):
            (window / max(1, n_stops)).backward()
            loss_sum += float(window.detach())
        gnorm = float(torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0))
        opt.step()
        sched.step()

        rec = {"step": step, "loss": loss_sum / max(1, n_stops),
               "l21_self": parts["l21_self"] / max(1, nprobe),
               "l21_world": parts["l21_world"] / max(1, nprobe),
               "grad_norm": gnorm, "state_norm": float(np.mean(state_norms)),
               "peak_gb": (torch.cuda.max_memory_allocated() / 1e9
                           if device.type == "cuda" else 0.0),
               "sec": time.time() - t0}
        if val_clips and step % args.val_every == 0:
            rec.update(evaluate(model, val_clips, N, args, device))
            print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                      if k.startswith("val")), flush=True)
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
            dump_viz(model, clips[0], args, device, step, tag="train")
            if val_clips:
                dump_viz(model, val_clips[0], args, device, step, tag="val")

    torch.save({"model": model.state_dict(), "step": args.updates,
                "args": vars(args)}, args.out / "last.pt")
    (args.out / "history.json").write_text(json.dumps(hist))
    dump_viz(model, clips[0], args, device, args.updates, tag="train")
    if val_clips:
        dump_viz(model, val_clips[0], args, device, args.updates, tag="val")
    if run is not None:
        run.finish()
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


@torch.no_grad()
def evaluate(model, vclips, frames, args, device):
    """Recall on UNSEEN scenes -- the only metric weights cannot memorise.

    Two stream lengths: matched to training (`valm_*`) and a fixed 96-frame
    extrapolation (`valx_*`). Fixed windows (start 0) and a fixed lag ladder,
    so the number is comparable across steps and runs."""
    model.eval()
    res = {}
    for tag, n in (("m", frames), ("x", 96)):
        es, ew = [], []
        for c in vclips:
            n_c = min(n, len(c))
            state, spos = model.init_state(1, device)
            for t in range(n_c):
                if not args.no_write:
                    state = model.write(state, spos, c.tap(t), c.patch_hw)
            T = n_c - 1
            lags = sorted({0, 1, min(2, T), min(4, T), T // 2, T})
            qs = [T - l for l in lags]
            rm, xs, xw, valid = c.probe_inputs(qs, args.raymap_convention, anchor=0)
            rays = (c.probe_rays(qs, anchor=0, unit=args.head != "lingbot")
                    if args.head in ("raydepth", "lingbot", "smallread") else None)
            out = run_probe(model, state.expand(len(qs), -1, -1),
                            spos.expand(len(qs), -1, -1), rm, (c.h, c.w), rays)
            es.append(float((out["pts3d_in_self_view"] - xs)
                            .norm(dim=-1)[valid].mean()))
            ew.append(float((out["pts3d_in_other_view"] - xw)
                            .norm(dim=-1)[valid].mean()))
        res[f"val{tag}_self"] = float(np.mean(es))
        res[f"val{tag}_world"] = float(np.mean(ew))
    model.train()
    return res


@torch.no_grad()
def dump_viz(model, clip, args, device, step, tag="train"):
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
    rm, xs, xw, valid = clip.probe_inputs(qs, args.raymap_convention, anchor=0)
    rays = (clip.probe_rays(qs, anchor=0, unit=args.head != "lingbot")
            if args.head in ("raydepth", "lingbot", "smallread") else None)
    st = torch.zeros_like(state) if args.zero_state else state
    out = run_probe(model, st.expand(len(qs), -1, -1),
                    spos.expand(len(qs), -1, -1), rm, (clip.h, clip.w), rays)
    d = args.out / "viz"
    d.mkdir(exist_ok=True)
    np.savez_compressed(
        d / f"probe_{tag}_{step:06d}.npz",
        lags=np.array(lags), qs=np.array(qs), t=T,
        pred_world=out["pts3d_in_other_view"].float().cpu().numpy(),
        pred_self=out["pts3d_in_self_view"].float().cpu().numpy(),
        conf_world=out["conf"].float().cpu().numpy(),
        conf_self=out["conf_self"].float().cpu().numpy(),
        gt_world=xw.float().cpu().numpy(), gt_self=xs.float().cpu().numpy(),
        valid=valid.cpu().numpy(), ray_o=rm[:, :3].float().cpu().numpy(),
        c2w_rel=relative_c2w(clip.gt_c2w[torch.tensor(qs, device=clip.device)],
                             clip.gt_c2w[0][None].expand(len(qs), -1, -1)
                             ).cpu().numpy(),
        scene=clip.meta.scene, frame_ids=np.array(clip.idx),
    )
    model.train()


if __name__ == "__main__":
    raise SystemExit(main())
