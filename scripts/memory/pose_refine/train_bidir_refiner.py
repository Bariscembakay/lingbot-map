#!/usr/bin/env python3
"""Bidirectional transformer on top of lingbot's aggregator, decoded by lingbot's frozen head.

Input at prefix length t (frames 0..t-1), all from lingbot's last layer (tap 23, 2048-d):
  specials   camera + 4 registers + scale token of every frame          -> 6t tokens
  --patches  also the patch tokens of the 8 anchor frames and of the last 64 frames,
             the frames whose full tokens lingbot's KV cache holds at time t
Four pre-norm self-attention blocks at width 2048 (no in/out projection), bidirectional,
1-D RoPE on frame index, learned token-type (and patch-position) embeddings. The outputs at
the camera-token positions ARE the new camera tokens (no added residual), decoded by
lingbot's camera head, frozen and causal as lingbot runs it.
--anchors fixed: frames 0..7 keep lingbot's own camera tokens and poses and are left out of
the loss; free: they are refined and supervised like every other frame.
Loss, lingbot paper Sec. 3.3 (training code unreleased; lambdas not given, so 1): absolute pose
as VGGT on the camera-to-world encoding over the head iterations, plus a relative-pose loss
over all frame pairs within a 64-frame window (geodesic rotation + L1 translation).
GT is the caches' anchor-scale-normalised pose.
Usage: train_bidir_refiner.py --out DIR [--patches] [--anchors fixed|free]
"""
from __future__ import annotations

import argparse
import json
import math
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.checkpoint import checkpoint

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from lingbot_map.memory.cache_format import ClipCache  # noqa: E402

D, HEADS, A, WINDOW, NPATCH = 2048, 16, 8, 64, 999
INPUTS = Path("/group/compact-3dmem/datasets/lingbot-poserefine-v1")


class Clip:
    """Tap-23 specials (in RAM), lingbot + GT poses, and a lazy memmap of tap-23 patch tokens."""

    def __init__(self, f: Path, device, patches: bool):
        z = np.load(f)
        self.name, self.src = f.stem, str(z["source"])
        self.spec = torch.from_numpy(np.ascontiguousarray(z["specials"][:, tp.TAP23])).to(device)   # [T,6,D]
        self.T = len(self.spec)
        self.hw = tuple(int(x) for x in z["hw"])
        self.scale = float(z["gt_scale"])
        self.lingbot = torch.from_numpy(z["pose_enc"]).to(device)
        self.gt_c2w = torch.from_numpy(z["gt_c2w"]).to(device)
        K = torch.from_numpy(z["K"]).to(device)
        self.gt_enc = tp.extri_intri_to_pose_encoding(self.gt_c2w[None, :, :3], K[None], self.hw)[0]
        self._taps = None
        self.patches = patches

    def patch_tokens(self, frames):
        if self._taps is None:
            c = ClipCache(self.src)
            self._taps, self._ti, self._psi = c.taps, c.meta.tap_layers.index(23), c.meta.patch_start_idx
        return torch.from_numpy(np.asarray(self._taps[frames, self._ti, self._psi:]))      # [F,999,D] fp16


def patch_frames(t):
    return sorted(set(range(min(A, t))) | set(range(max(0, t - WINDOW), t)))


def rope1d(x, pos, theta=10000.0):
    d = x.shape[-1] // 2
    freq = theta ** (-torch.arange(d, device=x.device, dtype=torch.float32) / d)
    ang = pos[:, None, :, None].float() * freq
    cos, sin = ang.cos().to(x.dtype), ang.sin().to(x.dtype)
    a, b = x[..., :d], x[..., d:]
    return torch.cat([a * cos - b * sin, a * sin + b * cos], -1)


class Block(nn.Module):
    def __init__(self):
        super().__init__()
        self.n1, self.qkv, self.o = nn.LayerNorm(D), nn.Linear(D, 3 * D), nn.Linear(D, D)
        self.n2, self.mlp = nn.LayerNorm(D), nn.Sequential(nn.Linear(D, 4 * D), nn.GELU(), nn.Linear(4 * D, D))

    def forward(self, x, pos):
        B, N, _ = x.shape
        q, k, v = self.qkv(self.n1(x)).view(B, N, 3, HEADS, D // HEADS).permute(2, 0, 3, 1, 4)
        q, k = rope1d(q, pos), rope1d(k, pos)
        x = x + self.o(F.scaled_dot_product_attention(q, k, v).transpose(1, 2).reshape(B, N, D))
        return x + self.mlp(self.n2(x))


class BidirRefiner(nn.Module):
    def __init__(self, head, anchors_fixed, depth=4, ckpt=False):
        super().__init__()
        self.type_emb = nn.Parameter(torch.zeros(7, D))          # 6 special kinds + patch
        self.patch_pos = nn.Parameter(torch.zeros(NPATCH, D))
        self.blocks = nn.ModuleList(Block() for _ in range(depth))
        self.head, self.anchors_fixed, self.ckpt = head.requires_grad_(False), anchors_fixed, ckpt

    def forward(self, spec, lingbot_pose, patches=None, pframes=None):
        """spec [B,t,6,D]; patches [B,F,999,D] of frames pframes [F] -> list of pose encodings [B,t,9]."""
        B, t = spec.shape[:2]
        dev = spec.device
        x = (spec.float() + self.type_emb[:6]).reshape(B, t * 6, D)
        pos = torch.arange(t, device=dev).repeat_interleave(6)
        if patches is not None:
            p = (patches.float() + self.type_emb[6] + self.patch_pos).reshape(B, -1, D)
            x = torch.cat([x, p], 1)
            pos = torch.cat([pos, torch.as_tensor(pframes, device=dev).repeat_interleave(NPATCH)])
        pos = pos[None].expand(B, -1)
        for blk in self.blocks:
            x = checkpoint(blk, x, pos, use_reentrant=False) if self.ckpt and self.training else blk(x, pos)
        cam = x[:, : t * 6].reshape(B, t, 6, D)[:, :, 0]
        if self.anchors_fixed:
            cam = torch.cat([spec[:, :A, 0].to(cam.dtype), cam[:, A:]], 1)
        # lingbot's head on lingbot's camera head inputs: frozen, causal, 8 scale frames.
        # In eval, the equivalent dense mask (bit-identical, see train_pose_refine.py decode):
        # the FlexAttention one recompiles for every new length.
        nfs, restore = 8, None
        if not torch.is_grad_enabled():
            restore = [b._prepare_blockwise_causal_attn_mask for b in self.head.trunk]
            for b in self.head.trunk:
                b._prepare_blockwise_causal_attn_mask = (
                    lambda device, num_frames, frame_seqlen, num_frame_per_block:
                    torch.ones(num_frames, num_frames, dtype=torch.bool, device=device).tril()
                    | tp.anchor_block(num_frames, A, device))
            nfs = -1
        preds = self.head.trunk_fn(self.head.token_norm(cam), None, self.head.num_iterations,
                                   num_frame_per_block=1, num_frame_for_scale=nfs, sliding_window_size=-1)
        if restore is not None:
            for b, f in zip(self.head.trunk, restore):
                b._prepare_blockwise_causal_attn_mask = f
        if self.anchors_fixed:
            preds = [torch.cat([lingbot_pose[:, :A].to(q.dtype), q[:, A:]], 1) for q in preds]
        return preds


def geodesic(Ra, Rb):
    c = ((Ra.transpose(-1, -2) @ Rb).diagonal(dim1=-2, dim2=-1).sum(-1) - 1) / 2
    return torch.acos(c.clamp(-1 + 1e-6, 1 - 1e-6))


def loss_fn(preds, gt, hw, skip):
    """lingbot Sec. 3.3: VGGT absolute-pose loss (over head iterations) + relative-pose loss over
    all pairs within a 64-frame window (final iteration), frames < skip excluded."""
    l_abs = tp.pose_loss(preds, gt, skip=skip)
    P, G = tp.pose_enc_to_c2w(preds[-1], hw), tp.pose_enc_to_c2w(gt, hw)
    t = gt.shape[1]
    i, j = torch.meshgrid(torch.arange(t, device=gt.device), torch.arange(t, device=gt.device), indexing="ij")
    m = (i != j) & ((i - j).abs() < WINDOW) & (i >= skip) & (j >= skip)
    i, j = i[m], j[m]
    Rp_i, Rp_j, Rg_i, Rg_j = P[:, i, :3, :3], P[:, j, :3, :3], G[:, i, :3, :3], G[:, j, :3, :3]
    rel_p = Rp_i.transpose(-1, -2) @ Rp_j
    rel_g = Rg_i.transpose(-1, -2) @ Rg_j
    tp_ = (Rp_i.transpose(-1, -2) @ (P[:, j, :3, 3] - P[:, i, :3, 3])[..., None])[..., 0]
    tg_ = (Rg_i.transpose(-1, -2) @ (G[:, j, :3, 3] - G[:, i, :3, 3])[..., None])[..., 0]
    l_rel = geodesic(rel_p, rel_g).mean() + (tp_ - tg_).abs().sum(-1).mean()
    return l_abs + l_rel, l_abs.detach(), l_rel.detach()


def batch_cpu(clips, idx, t, patches):
    spec = torch.stack([clips[i].spec[:t] for i in idx])
    gt = torch.stack([clips[i].gt_enc[:t] for i in idx])
    lp = torch.stack([clips[i].lingbot[:t] for i in idx])
    pf = patch_frames(t) if patches else None
    pt = torch.stack([clips[i].patch_tokens(pf) for i in idx]).pin_memory() if patches else None
    return spec, gt, lp, pt, pf


def to_dev(b, dev):
    spec, gt, lp, pt, pf = b
    return (spec.to(dev, non_blocking=True), gt.to(dev), lp.to(dev),
            pt.to(dev, non_blocking=True) if pt is not None else None, pf)


def batch(clips, idx, t, dev, patches):
    return to_dev(batch_cpu(clips, idx, t, patches), dev)


@torch.no_grad()
def evaluate(model, sets, dev, patches):
    model.eval()
    out = {}
    for name, clips, ts in sets:
        res = {}
        for c in clips:
            for t in ts:
                t = min(t, c.T)
                spec, gt, lp, pt, pf = batch([c], [0], t, dev, patches)
                with torch.autocast("cuda", dtype=torch.bfloat16):
                    p = model(spec, lp, pt, pf)[-1][0].float()
                r = tp.traj_metrics(tp.pose_enc_to_c2w(p, c.hw), c.gt_c2w[:t], c.scale)
                r["lingbot"] = tp.traj_metrics(tp.pose_enc_to_c2w(c.lingbot[:t].float(), c.hw), c.gt_c2w[:t], c.scale)
                res[f"{c.name}@{t}"] = r
        summ = {}
        for t in ts:
            rows = [r for k, r in res.items() if k.endswith(f"@{min(t, max(c.T for c in clips))}")]
            summ[str(t)] = {m: float(np.mean([r[m] for r in rows])) for m in tp.METRICS}
            summ[str(t)].update({f"lingbot_{m}": float(np.mean([r["lingbot"][m] for r in rows])) for m in tp.METRICS})
        out[name] = {"summary": summ, "per_clip": res}
    model.train()
    return out


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--inputs", type=Path, default=INPUTS)
    ap.add_argument("--patches", action="store_true")
    ap.add_argument("--anchors", choices=["fixed", "free"], required=True)
    ap.add_argument("--steps", type=int, default=6000)
    ap.add_argument("--batch", type=int, default=8)
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--t-min", type=int, default=16)
    ap.add_argument("--t-max", type=int, default=320)
    ap.add_argument("--t-step", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=250)
    ap.add_argument("--nrgbd-lengths", default="64,128,256,320,500")
    ap.add_argument("--grad-ckpt", action="store_true")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--max-train", type=int, default=0, help="smoke: first N train clips per dir")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    head = tp.build_head(tp.REPO / "ckpt/lingbot-map.pt", rope_on=False)
    model = BidirRefiner(head, args.anchors == "fixed", ckpt=args.grad_ckpt).to(dev)

    def load(d, store):
        fs = sorted(f for f in (args.inputs / d / "specials").glob("*.npz") if ".tmp" not in f.stem)
        fs = fs[: args.max_train] if args.max_train and d.startswith("train") else fs
        return [Clip(f, store, args.patches) for f in fs]

    train = load("train", "cpu") + load("train_v5", "cpu")
    lens = [int(x) for x in args.nrgbd_lengths.split(",")]
    sets = [("val24", load("val24", dev), [320]), ("nrgbd", load("nrgbd", dev), lens)]
    print(f"[data] train {len(train)} clips; val24 {len(sets[0][1])}; nrgbd {len(sets[1][1])} at t={lens}; "
          f"patches={args.patches} anchors={args.anchors}", flush=True)
    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] refiner {sum(p.numel() for p in params)/1e6:.1f} M (head frozen)", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    skip = A if args.anchors == "fixed" else 0
    tchoices = np.arange(args.t_min, args.t_max + 1, args.t_step)

    def summarise(r):
        row = {}
        for s, v in r.items():
            for t, m in v["summary"].items():
                for k in ("ate", "auc3", "auc30", "lingbot_ate", "lingbot_auc3"):
                    row[f"{s}@{t}/{k}"] = m[k]
        return row

    hist, best = [], float("inf")
    r = evaluate(model, sets, dev, args.patches)
    hist.append({"step": 0, **summarise(r)})
    print(f"[val 0] {hist[-1]}", flush=True)
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(2)   # patch tokens are ~150 MB per sample of /data memmap reads

    def submit():
        t = int(rng.choice(tchoices))
        return t, pool.submit(batch_cpu, train, rng.choice(len(train), args.batch, replace=False), t, args.patches)

    queue = [submit(), submit()]
    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        t, fut = queue.pop(0)
        queue.append(submit())
        spec, gt, lp, pt, pf = to_dev(fut.result(), dev)
        with torch.autocast("cuda", dtype=torch.bfloat16):
            preds = model(spec, lp, pt, pf)
        loss, la, lr_ = loss_fn([p.float() for p in preds], gt, train[0].hw, skip)
        opt.zero_grad(set_to_none=True)
        loss.backward()
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            print(f"[{step:6d}] loss {loss.item():.4f} abs {la.item():.4f} rel {lr_.item():.4f} t {t} "
                  f"|g| {gn.item():.3f} {(time.time()-t0)/step:.2f}s/it", flush=True)
        if step % args.val_every == 0 or step == args.steps:
            r = evaluate(model, sets, dev, args.patches)
            row = {"step": step, **summarise(r)}
            hist.append(row)
            print(f"[val {step}] {row}", flush=True)
            ck = {"model": {k: v for k, v in model.state_dict().items() if not k.startswith("head.")},
                  "args": vars(args), "step": step, "val": row}
            torch.save(ck, args.out / "last.pt")
            if row["val24@320/ate"] < best:
                best = row["val24@320/ate"]
                torch.save(ck, args.out / "best.pt")
                (args.out / "best_eval.json").write_text(json.dumps(r, indent=1))
            (args.out / "history.json").write_text(json.dumps(hist, indent=1))


if __name__ == "__main__":
    main()
