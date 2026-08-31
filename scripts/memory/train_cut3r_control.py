#!/usr/bin/env python3
"""CONTROL arm: fine-tune the ACTUAL CUT3R on the spatial-memory recall objective.

Same objective, probe protocol, optimizer and schedule as train_state.py --
`probe_loss` and `sample_queries` are imported from there, not copied -- but the
model is the vendored, unmodified CUT3R (`ARCroco3DStereo`), warm-started from
the released 512-DPT checkpoint, so the comparison against our StateMemory arms
is 1:1 on everything except the architecture.

Design decisions (also in the report to the main session):

* **Frozen image encoder, injected tokens.** `patch_embed`/`enc_blocks`/
  `enc_norm` are frozen and never run: write steps consume precomputed encoder
  tokens from build_cut3r_enc_cache.py and add the (trainable)
  `masked_ray_map_token`, exactly what `_encode_views` produces for an
  img-only view. Everything else trains: both decoders, state/register tokens,
  pose token + pose_retriever, raymap encoder, DPT heads.
* **Streaming write WITH gradients.** The per-frame step is re-composed from
  the model's own pieces (`_recurrent_rollout`, `pose_retriever.inquire`/
  `update_mem`), mirroring `_forward_impl`'s step body minus the head call
  (recall-only objective: no loss on write steps). The recurrent state a step
  carries is (state_feat, mem); BOTH are detached at tbptt cuts.
* **Probe = raymap-only view.** `_encode_ray_map` + `masked_img_token`, pose
  queried from mem, one `_decoder` pass, `DPTPts3dPose` head. The probe's
  outputs are never written back, so the state cannot be contaminated (the
  in-graph equivalent of the model's own `update_mask = img_mask & update`
  gate, which zeroes the state update for img_mask=False views).
* **Units.** Raymaps are built at CUT3R's operating point: metric scale
  (canonical x meta.gt_scale), poses relative to the window's first frame.
  Predictions are divided by gt_scale before loss/metrics, and GT supervision
  pointmaps are built in canonical units -- so every logged number is directly
  comparable to the StateMemory fleet's.
"""
from __future__ import annotations

import argparse
import json
import re
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(REPO / "CUT3R" / "src"))
sys.path.insert(0, str(REPO / "CUT3R" / "src" / "croco"))

from lingbot_map.memory.cache_format import ClipCache            # noqa: E402
from lingbot_map.memory.probe_data import build_raymap, gt_pointmaps  # noqa: E402
from lingbot_map.memory.recall_loss import probe_loss            # noqa: E402
from scripts.memory.train_state import expand_paths, sample_queries  # noqa: E402


class ClipC:
    """One tap-cache clip's GT plus its CUT3R encoder-token cache, on device.

    GT (depth/poses/K) comes from the tap cache like train_state.Clip; the
    write inputs come from the enc cache instead of the aggregator taps.
    Encoder tokens stay mmap'd on the CPU (fp16) and move per frame.
    """

    def __init__(self, path: Path, enc_root: Path, subsample: int, device,
                 max_frames: int | None):
        c = ClipCache(path)
        idx = list(range(0, len(c), subsample))
        if max_frames:
            idx = idx[:max_frames]
        self.idx = idx
        self.meta = c.meta
        self.gt_scale = float(c.meta.gt_scale)
        self.device = device

        enc_dir = Path(enc_root) / path.name
        em = json.loads((enc_dir / "meta.json").read_text())
        assert em["cache_indices"] == idx, \
            f"{path.name}: enc cache frames {em['cache_indices'][:3]}.. != {idx[:3]}.."
        assert em["scene"] == c.meta.scene
        self.h, self.w = em["hw"]
        self.patch_hw = tuple(em["patch_hw"])
        self.enc = np.load(enc_dir / "enc_tokens.npy", mmap_mode="r")
        assert self.enc.shape[0] == len(idx)

        self.K = torch.from_numpy(
            np.load(enc_dir / "K_cut3r.npy")).to(device).float()
        # GT depth: canonical units, nearest-resized from the tap-cache grid to
        # CUT3R's 384x512 (decimation, no invented values; valid = depth>0
        # survives exactly).
        d = torch.from_numpy(np.ascontiguousarray(c.gt_depth[idx])).float()
        self.gt_depth = F.interpolate(d[:, None], size=(self.h, self.w),
                                      mode="nearest")[:, 0].to(device)
        self.gt_c2w = torch.from_numpy(c.gt_c2w[idx]).to(device).float()
        self.c2w_metric = self.gt_c2w.clone()
        self.c2w_metric[:, :3, 3] *= self.gt_scale

    def __len__(self):
        return len(self.idx)

    def enc_tok(self, t: int) -> torch.Tensor:
        return torch.from_numpy(np.asarray(self.enc[t]))

    def probe_inputs(self, qs, anchor: int = 0):
        """Raymap in METRIC scale (CUT3R's operating point), GT pointmaps in
        CANONICAL units, both relative to `anchor` (the stream's first frame).
        qs are absolute frame indices, as in train_state.Clip.probe_inputs."""
        q = torch.tensor(qs, device=self.device)
        rm = build_raymap(self.K[q], self.c2w_metric[q],
                          self.c2w_metric[anchor][None], self.h, self.w, "cut3r")
        xs, xw, valid = gt_pointmaps(self.gt_depth[q], self.K[q],
                                     self.gt_c2w[q], self.gt_c2w[anchor][None])
        return rm, xs, xw, valid


# --- CUT3R streaming, re-composed from the model's own pieces ---------------

def cut3r_init(model, b: int, device):
    """Fresh (state_feat, state_pos, mem) -- what _forward_encoder derives
    before the first view. Gradients flow into register_tokens,
    decoder_embed_state and pose_retriever.mem, all trainable."""
    dummy_tok = torch.zeros(b, 1, model.enc_embed_dim, device=device)
    dummy_pos = torch.zeros(b, 1, 2, dtype=torch.long, device=device)
    state_feat, state_pos = model._init_state(dummy_tok, dummy_pos)
    mem = model.pose_retriever.mem.expand(b, -1, -1)
    return state_feat, state_pos, mem


def _pose_query(model, feat_i, pos_i, mem, first: bool):
    if not model.pose_head_flag:
        return None, None, None
    global_i = model._get_img_level_feat(feat_i)
    if first:
        pose_feat = model.pose_token.expand(feat_i.shape[0], -1, -1)
    else:
        pose_feat = model.pose_retriever.inquire(global_i, mem)
    pose_pos = -torch.ones(feat_i.shape[0], 1, 2,
                           device=feat_i.device, dtype=pos_i.dtype)
    return global_i, pose_feat, pose_pos


def cut3r_write(model, state_feat, state_pos, mem, enc_tok, pos_i, first: bool):
    """One img-view step of _forward_impl, head call omitted (no write loss).

    enc_tok is the cached frozen-encoder output; adding masked_ray_map_token
    reproduces _encode_views' feat for an img_mask=True, ray_mask=False view.
    update/reset semantics: a write view always has img_mask=update=True and
    reset=False, so state and mem are replaced unconditionally.
    """
    feat_i = enc_tok + model.masked_ray_map_token
    global_i, pose_feat, pose_pos = _pose_query(model, feat_i, pos_i, mem, first)
    new_state, dec = model._recurrent_rollout(
        state_feat, state_pos, feat_i, pos_i, pose_feat, pose_pos, None)
    if model.pose_head_flag:
        mem = model.pose_retriever.update_mem(mem, global_i, dec[-1][:, 0:1])
    return new_state, mem


def cut3r_probe(model, state_feat, state_pos, mem, raymap, true_shape):
    """One raymap-only view (img_mask=False, ray_mask=True, update=False),
    WITH grad. Returns the DPT head's dict (pts3d_in_self_view, conf_self,
    pts3d_in_other_view, conf, camera_pose, ...). Nothing is written back:
    the model's own gate for this view type is update_mask = img_mask & update
    = False, and here the new state/mem are simply never taken."""
    feat_ls, pos, _ = model._encode_ray_map(raymap, true_shape)
    feat_i = feat_ls[-1] + model.masked_img_token
    _, pose_feat, pose_pos = _pose_query(model, feat_i, pos, mem, first=False)
    _, dec = model._recurrent_rollout(
        state_feat, state_pos, feat_i, pos, pose_feat, pose_pos, None)
    d = model.dec_depth
    head_in = [dec[0].float(), dec[d * 2 // 4][:, 1:].float(),
               dec[d * 3 // 4][:, 1:].float(), dec[d].float()]
    return model._downstream_head(head_in, true_shape, pos=pos)


def to_canonical(res: dict, scales: torch.Tensor) -> dict:
    """CUT3R predicts at its metric operating point; divide by gt_scale so the
    loss and every logged L21 are in the fleet's canonical units."""
    s = scales.view(-1, 1, 1, 1)
    return {"pts3d_in_self_view": res["pts3d_in_self_view"].float() / s,
            "conf_self": res["conf_self"].float(),
            "pts3d_in_other_view": res["pts3d_in_other_view"].float() / s,
            "conf": res["conf"].float()}


def load_cut3r(ckpt: str, device):
    from dust3r.model import ARCroco3DStereo
    model = ARCroco3DStereo.from_pretrained(ckpt).to(device)
    return model


def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--clips", nargs="+", required=True, type=Path)
    ap.add_argument("--val-clips", nargs="*", type=Path, default=[])
    ap.add_argument("--val-every", type=int, default=100)
    ap.add_argument("--out", required=True, type=Path)
    ap.add_argument("--enc-cache", required=True, type=Path)
    ap.add_argument("--cut3r-ckpt",
                    default="/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth")
    ap.add_argument("--updates", type=int, default=4000)
    ap.add_argument("--lr", type=float, default=1e-4)
    ap.add_argument("--wd", type=float, default=0.05)
    ap.add_argument("--warmup", type=int, default=100)
    ap.add_argument("--subsample", type=int, default=2)
    ap.add_argument("--max-frames", type=int, default=160)
    ap.add_argument("--frames", type=int, default=16)
    ap.add_argument("--batch", type=int, default=4)
    ap.add_argument("--n-past", type=int, default=4)
    ap.add_argument("--probe-every", type=int, default=1)
    ap.add_argument("--probe-current", default="on", choices=["on", "off"])
    ap.add_argument("--tbptt", type=int, default=8)
    ap.add_argument("--no-grad-ckpt", action="store_true")
    ap.add_argument("--amp", default="bf16", choices=["bf16", "off"])
    ap.add_argument("--wandb", default="online", choices=["online", "offline", "off"])
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--log-every", type=int, default=10)
    ap.add_argument("--save-every", type=int, default=250)
    ap.add_argument("--patience", type=int, default=0)
    ap.add_argument("--min-delta", type=float, default=0.002)
    ap.add_argument("--resume", default="off", choices=["off", "auto"])
    # Ablation of the pretrained decoder: keep the (frozen, cached) encoder
    # and optionally the frozen heads, but re-initialise the interconnected
    # decoder system -- the CUT3R analogue of reinit-write + frozen-head arms.
    ap.add_argument("--random-decoder", action="store_true")
    ap.add_argument("--freeze-head", action="store_true")
    ap.add_argument("--equiv-check", action="store_true",
                    help="compare injected-cache vs full-RGB forward on one "
                         "frame pair of the first clip, then exit")
    ap.add_argument("--video-root", default="/data/ScanNetpp/data")
    args = ap.parse_args()

    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

    if args.equiv_check:
        return equiv_check(args, device)

    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(
        json.dumps({k: str(v) for k, v in vars(args).items()}, indent=2))

    run = None
    if args.wandb != "off":
        try:
            import wandb
            wdir = Path("/group/compact-3dmem/wandb")
            if not (wdir.is_dir() and os.access(wdir, os.W_OK)):
                wdir = args.out
            # Deterministic id + resume: a requeued/preempted job appends to
            # the SAME wandb run instead of fragmenting one training into many.
            rid = re.sub(r"[^a-z0-9_-]", "-", args.out.name.lower())[:120]
            run = wandb.init(project="spatial_memory", name=args.out.name,
                             id=rid, resume="allow",
                             mode=args.wandb, config=vars(args), dir=str(wdir))
        except Exception as e:
            print(f"[wandb] disabled: {e}", flush=True)

    clips = [ClipC(p, args.enc_cache, args.subsample, device, args.max_frames)
             for p in expand_paths(args.clips)]
    val_clips = [ClipC(p, args.enc_cache, args.subsample, device, args.max_frames)
                 for p in expand_paths(args.val_clips)]
    assert len({(c.h, c.w, c.patch_hw) for c in clips + val_clips}) == 1
    print(f"[data] {len(clips)} train / {len(val_clips)} val clip(s), "
          f"{len(clips[0])} frames each, {clips[0].h}x{clips[0].w}, "
          f"patch grid {clips[0].patch_hw}", flush=True)

    model = load_cut3r(args.cut3r_ckpt, device)
    # The released config carries freeze='encoder', which set_freeze() applies
    # in __init__ -- and that list also freezes the RAYMAP encoder and the
    # masked tokens. Our contract is "image encoder frozen, everything else
    # trains" (the StateMemory arms train their raymap path), so unfreeze all,
    # then freeze exactly the ViT image encoder (which never runs anyway --
    # its tokens are cached).
    model.requires_grad_(True)
    for m in (model.patch_embed, model.enc_blocks, model.enc_norm):
        m.requires_grad_(False)
    if args.random_decoder:
        torch.manual_seed(args.seed)
        # Construct an un-loaded twin the same way their load_model does
        # (eval of the ckpt's own args string, incl. its two compat fixes) --
        # model.config does not round-trip through the constructor.
        import dust3r.model as _dm
        _raw = torch.load(args.cut3r_ckpt, map_location="cpu",
                          weights_only=False)
        _a = _raw["args"].model.replace("ManyAR_PatchEmbed", "PatchEmbedDust3R")
        if "landscape_only" not in _a:
            _a = _a[:-2] + ", landscape_only=False))"
        else:
            _a = _a.replace(" ", "").replace("landscape_only=True",
                                             "landscape_only=False")
        del _raw
        fresh = eval(_a, vars(_dm))   # CPU, fully random init
        roots = ("decoder_embed", "dec_blocks", "dec_norm",
                 "decoder_embed_state", "dec_blocks_state", "dec_norm_state",
                 "register_tokens")
        sd = {k: v for k, v in fresh.state_dict().items()
              if k.split(".")[0] in roots}
        model.load_state_dict(sd, strict=False)
        # pose_retriever/pose_token stay pretrained: they feed the (frozen)
        # pose branch and are auxiliary memory, not the decoder under test.
        print(f"[random-decoder] re-initialised {len(sd)} tensors under "
              f"{roots}", flush=True)
        del fresh
    if args.freeze_head:
        model.downstream_head.requires_grad_(False)
        print("[freeze-head] downstream_head frozen", flush=True)
    model.gradient_checkpointing = not args.no_grad_ckpt
    model.train()
    n_train = sum(p.numel() for p in model.parameters() if p.requires_grad)
    n_all = sum(p.numel() for p in model.parameters())
    print(f"[model] {n_train/1e6:.1f} M trainable / {n_all/1e6:.1f} M total; "
          f"grad_ckpt={model.gradient_checkpointing}", flush=True)

    opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=args.wd)
    sched = torch.optim.lr_scheduler.LambdaLR(
        opt, lambda s: min(1.0, (s + 1) / max(1, args.warmup)))

    hist = []
    start_step, best_val, evals_since_best = 0, float("inf"), 0
    ckpt_path = args.out / "last.pt"
    if args.resume == "auto" and ckpt_path.exists():
        sd = torch.load(ckpt_path, map_location="cpu", weights_only=False)
        model.load_state_dict(sd["model"])
        if "opt" in sd:
            opt.load_state_dict(sd["opt"])
            sched.load_state_dict(sd["sched"])
        start_step = int(sd.get("step", 0)) + 1
        best_val = float(sd.get("best_val", float("inf")))
        evals_since_best = int(sd.get("evals_since_best", 0))
        hp = args.out / "history.json"
        if hp.exists():
            hist = [r for r in json.loads(hp.read_text())
                    if r.get("step", 0) < start_step]
        print(f"[resume] {ckpt_path} -> step {start_step} "
              f"(best_val {best_val:.4f})", flush=True)

    B = min(args.batch, len(clips))
    N = args.frames
    hw = (clips[0].h, clips[0].w)
    ph, pw = clips[0].patch_hw
    t0 = time.time()

    def save_ckpt(step):
        torch.save({"model": model.state_dict(), "opt": opt.state_dict(),
                    "sched": sched.state_dict(), "step": step,
                    "best_val": best_val, "evals_since_best": evals_since_best,
                    "args": vars(args)}, args.out / "last.pt")

    step = max(0, start_step - 1)
    for step in range(start_step, args.updates):
        sel = [clips[i] for i in rng.choice(len(clips), size=B, replace=False)]
        starts = [int(rng.integers(0, max(1, len(c) - N + 1))) for c in sel]
        state, spos, mem = cut3r_init(model, B, device)
        pos_img = model.patch_embed.position_getter(B, ph, pw, device)
        shape1 = torch.tensor([[hw[0], hw[1]]], device=device)
        parts, nprobe = {"l21_self": 0.0, "l21_world": 0.0}, 0
        window, loss_sum = 0.0, 0.0
        frames_since_cut = 0
        # Same normalisation as train_state.py: stop count known upfront, so
        # tbptt and full-BPTT optimise the same objective.
        n_stops = sum(1 for t in range(N)
                      if t % args.probe_every == 0
                      and (t > 0 or args.probe_current == "on"))
        scales = torch.tensor([c.gt_scale for c in sel], device=device)
        state_norms = []
        opt.zero_grad(set_to_none=True)

        amp = torch.autocast("cuda", dtype=torch.bfloat16,
                             enabled=(args.amp == "bf16" and device.type == "cuda"))
        for t in range(N):
            tok = torch.stack([c.enc_tok(s0 + t) for c, s0 in zip(sel, starts)]
                              ).to(device).float()
            with amp:
                state, mem = cut3r_write(model, state, spos, mem, tok, pos_img,
                                         first=(t == 0))
            state_norms.append(float(state.detach().norm(dim=(1, 2)).mean()))

            if t % args.probe_every == 0:
                qs0 = sample_queries(rng, t, args.n_past, args.probe_current == "on")
                if qs0:
                    nq = len(qs0)
                    rms, xss, xws, vs = [], [], [], []
                    for c, s0 in zip(sel, starts):
                        qs = sample_queries(rng, t, args.n_past,
                                            args.probe_current == "on")
                        aq = [s0 + q for q in qs]
                        rm, xs, xw, valid = c.probe_inputs(aq, anchor=s0)
                        rms.append(rm); xss.append(xs); xws.append(xw); vs.append(valid)  # noqa: E702
                    rm = torch.cat(rms); xs = torch.cat(xss)                              # noqa: E702
                    xw = torch.cat(xws); valid = torch.cat(vs)                            # noqa: E702
                    with amp:
                        res = cut3r_probe(model,
                                          state.repeat_interleave(nq, 0),
                                          spos[:1].expand(B * nq, -1, -1),
                                          mem.repeat_interleave(nq, 0),
                                          rm, shape1.expand(B * nq, -1))
                    out = to_canonical(res, scales.repeat_interleave(nq))
                    loss, p = probe_loss(out, xs, xw, valid)
                    window = window + loss
                    for k in parts:
                        parts[k] += p[k]
                    nprobe += 1

            # Cut only AT a probe stop, first stop >= K frames since the last
            # cut -- train_state.py's rule, verbatim, so every write stays
            # supervised and the stop schedule matches the fleet's.
            frames_since_cut += 1
            if (args.tbptt and t % args.probe_every == 0
                    and frames_since_cut >= args.tbptt and t + 1 < N):
                if torch.is_tensor(window):
                    (window / max(1, n_stops)).backward()
                    loss_sum += float(window.detach())
                    window = 0.0
                # The recurrent state is (state_feat, mem); detach BOTH.
                state = state.detach()
                mem = mem.detach()
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
        stop = False
        if val_clips and step % args.val_every == 0:
            rec.update(evaluate(model, val_clips, N, args, device))
            print("[val] " + " ".join(f"{k}={v:.4f}" for k, v in rec.items()
                                      if k.startswith("val")), flush=True)
            if rec["valm_self"] < best_val - args.min_delta:
                best_val, evals_since_best = rec["valm_self"], 0
            else:
                evals_since_best += 1
            if args.patience and evals_since_best >= args.patience:
                print(f"[early-stop] no valm_self improvement > "
                      f"{args.min_delta} in {args.patience} evals "
                      f"(best {best_val:.4f}); stopping at step {step}",
                      flush=True)
                stop = True
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
            save_ckpt(step)
        if stop:
            break

    save_ckpt(step)
    (args.out / "history.json").write_text(json.dumps(hist))
    if run is not None:
        run.finish()
    print(f"[done] {time.time()-t0:.0f}s", flush=True)
    return 0


@torch.no_grad()
def evaluate(model, vclips, frames, args, device):
    """train_state.evaluate, on the CUT3R stream: fixed windows from frame 0,
    fixed lag ladder, unweighted L21 means in CANONICAL units."""
    model.eval()
    res = {}
    for tag, n in (("m", frames), ("x", 96)):
        es, ew = [], []
        for c in vclips:
            n_c = min(n, len(c))
            ph, pw = c.patch_hw
            state, spos, mem = cut3r_init(model, 1, device)
            pos_img = model.patch_embed.position_getter(1, ph, pw, device)
            for t in range(n_c):
                tok = c.enc_tok(t)[None].to(device).float()
                state, mem = cut3r_write(model, state, spos, mem, tok, pos_img,
                                         first=(t == 0))
            T = n_c - 1
            lags = sorted({0, 1, min(2, T), min(4, T), T // 2, T})
            qs = [T - l for l in lags]
            rm, xs, xw, valid = c.probe_inputs(qs, anchor=0)
            shape = torch.tensor([[c.h, c.w]], device=device).expand(len(qs), -1)
            r = cut3r_probe(model, state.expand(len(qs), -1, -1),
                            spos.expand(len(qs), -1, -1),
                            mem.expand(len(qs), -1, -1), rm, shape)
            out = to_canonical(r, torch.full((len(qs),), c.gt_scale, device=device))
            es.append(float((out["pts3d_in_self_view"] - xs)
                            .norm(dim=-1)[valid].mean()))
            ew.append(float((out["pts3d_in_other_view"] - xw)
                            .norm(dim=-1)[valid].mean()))
        res[f"val{tag}_self"] = float(np.mean(es))
        res[f"val{tag}_world"] = float(np.mean(ew))
    model.train()
    return res


@torch.no_grad()
def equiv_check(args, device) -> int:
    """Injected-cache vs full-RGB forward on one clip: write frames 0..1, probe
    frame 0. The two paths must agree to fp16-cache rounding."""
    from scripts.memory.build_cut3r_enc_cache import preprocess_like_demo
    from scripts.memory.build_cache import decode_frames, probe_video

    path = expand_paths(args.clips)[0]
    c = ClipC(path, args.enc_cache, args.subsample, device, args.max_frames)
    model = load_cut3r(args.cut3r_ckpt, device)
    model.eval()

    fids = [c.meta.frame_ids[c.idx[t]] for t in (0, 1)]
    video = Path(args.video_root) / c.meta.scene / "iphone" / "rgb.mkv"
    nw, nh, _ = probe_video(video)
    frames = decode_frames(video, fids, nw, nh)
    K_cache = np.load(path / "gt_intrinsics.npy")[[c.idx[0], c.idx[1]]].astype(np.float64)
    K_native = K_cache.copy()
    K_native[:, 0, :] *= nw / c.meta.width
    K_native[:, 1, :] *= nh / c.meta.height
    imgs, K_pre, H, W = preprocess_like_demo(frames, K_native)
    assert (H, W) == (c.h, c.w)
    assert np.abs(K_pre - c.K[:2].cpu().numpy()).max() < 1e-3

    shape1 = torch.tensor([[H, W]], device=device)
    ph, pw = c.patch_hw
    pos_img = model.patch_embed.position_getter(1, ph, pw, device)
    rm, _, _, _ = c.probe_inputs([0], anchor=0)

    outs, toks = [], []
    for src in ("rgb", "cache"):
        state, spos, mem = cut3r_init(model, 1, device)
        stream_toks = []
        for t in (0, 1):
            if src == "rgb":
                (x,), _, _ = model._encode_image(
                    imgs[t:t + 1].to(device), shape1)
            else:
                x = c.enc_tok(t)[None].to(device).float()
            stream_toks.append(x)
            state, mem = cut3r_write(model, state, spos, mem, x, pos_img,
                                     first=(t == 0))
        toks.append(torch.cat(stream_toks))
        res = cut3r_probe(model, state, spos, mem, rm, shape1)
        outs.append((state, res))

    dtok = (toks[0] - toks[1]).abs()
    dstate = (outs[0][0] - outs[1][0]).abs()
    ds = (outs[0][1]["pts3d_in_self_view"] - outs[1][1]["pts3d_in_self_view"]).abs()
    dw = (outs[0][1]["pts3d_in_other_view"] - outs[1][1]["pts3d_in_other_view"]).abs()
    dc = (outs[0][1]["conf_self"] - outs[1][1]["conf_self"]).abs()
    scale = float((outs[0][1]["pts3d_in_self_view"]).abs().mean())
    print(f"[equiv] enc tokens   max {dtok.max():.2e} mean {dtok.mean():.2e} "
          f"(fp16 cache rounding)")
    print(f"[equiv] state        max {dstate.max():.2e} mean {dstate.mean():.2e}")
    print(f"[equiv] pts3d self   max {ds.max():.2e} mean {ds.mean():.2e} m "
          f"(pred |x| mean {scale:.2f} m)")
    print(f"[equiv] pts3d world  max {dw.max():.2e} mean {dw.mean():.2e} m")
    print(f"[equiv] conf_self    max {dc.max():.2e} mean {dc.mean():.2e}")
    ok = float(ds.mean()) < 1e-2 and float(dw.mean()) < 1e-2
    print(f"[equiv] {'PASS' if ok else 'FAIL'} (mean pointmap deviation < 1 cm "
          f"metric)")
    return 0 if ok else 1


if __name__ == "__main__":
    raise SystemExit(main())
