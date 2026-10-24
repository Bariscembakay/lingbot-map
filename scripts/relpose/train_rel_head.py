#!/usr/bin/env python3
"""Relative-pose camera head on frozen lingbot tokens: CUT3R-rel's moving frame in lingbot (option A).

lingbot's camera head, initialised from lingbot and trained, reads the tap-23 camera token of
every frame and predicts that frame's pose relative to the previous one (camera-to-world of frame
i in camera i-1; frame 0 is the identity) plus its field of view. The trajectory is the chain of
these steps from frame 0, so no output grows with the distance travelled.
--mode rel (default):
  head  causal over a --window of frames, temporal RoPE on: lingbot's head has no position
        signal, so it could not tell which earlier token is frame i-1; the bounded window keeps
        RoPE distances inside what training saw at any stream length. --prev-token also feeds
        frame i-1's token through a zero-initialised projection.
  loss  every head iteration (VGGT's gamma decay): per-step L1 translation x --trans-w,
        geodesic rotation x --rot-w (atan2 form, finite gradient at zero; a quaternion loss let
        CUT3R-rel keep a constant 0.1-0.2 deg/frame bias), L1 FoV x 0.5; final iteration: the
        same terms on the chained trajectory's relative pose for every pair less than
        --pair-window frames apart (lingbot's relative-pose term) x --pair-w.
  --residual: each step is lingbot's own step (from its absolute poses) composed with the head's
        output, which starts at no motion: step 0 is lingbot's trajectory exactly.
--mode abs: the matched control -- the same head, data and schedule on lingbot's own setup (full
  causal attention, no RoPE, absolute pose + window pairs: train_bidir_refiner.loss_fn).
Data: the refiner's packs (tap-23 specials of lingbot-cache1048, every frame a keyframe), its
hold-outs and bad-GT list; camera tokens are extracted once per node to /scratch. Validation:
val24 @320, NRGBD, held-out long clips @320/640/1048, KITTI 00-10 and VBR at full length.
Usage: train_rel_head.py --out DIR [--mode rel|abs] [--prev-token]
"""
from __future__ import annotations

import argparse
import fcntl
import json
import math
import os
import sys
import time
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn

REPO = Path(__file__).resolve().parents[2]
sys.path.insert(0, str(REPO / "scripts/memory/pose_refine"))
import train_bidir_refiner as br  # noqa: E402
import train_pose_refine as tp  # noqa: E402

A, D = br.A, br.D
INPUTS = br.INPUTS
TOKENS = Path(os.environ.get("RELHEAD_TOKENS", f"/scratch/{os.environ.get('USER')}/relhead_cam23"))
PACKS = "tartanair-v1,tartanair-v2,tartanground,matrixcity,scannetpp,scannet-v2"


# -- data ----------------------------------------------------------------------
def pack_cam_tokens(pack_root: Path, ds: str) -> np.ndarray:
    """[N, D] fp16 camera tokens of a pack, extracted once per node from its specials memmap."""
    out = TOKENS / f"{ds}.npy"
    TOKENS.mkdir(parents=True, exist_ok=True)
    with open(TOKENS / f"{ds}.lock", "w") as lk:
        fcntl.flock(lk, fcntl.LOCK_EX)
        if not out.exists():
            mm = np.load(pack_root / ds / "spec23.npy", mmap_mode="r")
            tmp = TOKENS / f"{ds}.{os.getpid()}.tmp.npy"
            dst = np.lib.format.open_memmap(tmp, mode="w+", dtype=mm.dtype, shape=(len(mm), D))
            for s in range(0, len(mm), 1 << 16):
                dst[s:s + (1 << 16)] = mm[s:s + (1 << 16), tp.CAM]
            dst.flush()
            del dst
            tmp.rename(out)
    return np.load(out)


class TrainClip:
    def __init__(self, ds, tok, idx, i):
        o, n = int(idx["offset"][i]), int(idx["length"][i])
        self.ds, self.name, self.seq = ds, str(idx["name"][i]), str(idx["seq"][i])
        self.T, self.hw = n, tuple(int(x) for x in idx["hw"][i])
        self.scale = float(idx["gt_scale"][i])
        self.tok = tok[o:o + n]                                   # view into the dataset's array
        self.gt_c2w = torch.from_numpy(idx["gt_c2w"][o:o + n])
        self.K = torch.from_numpy(idx["K"][o:o + n])
        self.gt_enc = tp.extri_intri_to_pose_encoding(self.gt_c2w[None, :, :3], self.K[None], self.hw)[0]
        self.lingbot = idx["pose_enc"][o:o + n]


class EvalClip:
    """Camera tokens, GT, lingbot's poses on the GPU; from a pack clip or a poserefine-v1 npz."""

    def __init__(self, name, tok, gt_c2w, K, hw, scale, lingbot, dev):
        self.name, self.T, self.hw, self.scale = name, len(tok), hw, scale
        self.tok = torch.as_tensor(np.ascontiguousarray(tok)).to(dev)
        self.gt_c2w = torch.as_tensor(gt_c2w).to(dev)
        self.K = torch.as_tensor(K).to(dev)
        self.lingbot = torch.as_tensor(lingbot).to(dev)

    @classmethod
    def from_npz(cls, f, dev):
        z = np.load(f)
        return cls(f.stem, z["specials"][:, tp.TAP23, tp.CAM], z["gt_c2w"], z["K"],
                   tuple(int(x) for x in z["hw"]), float(z["gt_scale"]), z["pose_enc"], dev)


def ref_index(t, K, f0=0, device=None):
    """Reference frame of frames f0..f0+t-1: the latest keyframe (every K-th frame from 0) before
    each, frame 0 itself; K=1 is the previous frame."""
    i = torch.arange(f0, f0 + t, device=device)
    return torch.where(i > 0, K * ((i - 1) // K), 0)


def to_steps(G, K=1):
    """[..., t, 4, 4] camera-to-world -> each frame in its reference camera; frame 0 the identity."""
    r = ref_index(G.shape[-3], K, device=G.device)
    return rigid_inv(G[..., r, :, :]) @ G


def rigid_inv(M):
    R, t = M[..., :3, :3], M[..., :3, 3:]
    out = torch.zeros_like(M)
    out[..., :3, :3] = R.transpose(-1, -2)
    out[..., :3, 3:] = -R.transpose(-1, -2) @ t
    out[..., 3, 3] = 1
    return out


def scan(S):
    """[B, n, 4, 4] -> running products S_0 S_1 ... S_i: an inclusive scan with log2(n) batched
    products instead of n sequential ones."""
    k = 1
    while k < S.shape[1]:
        S = torch.cat([S[:, :k], S[:, :-k] @ S[:, k:]], 1)
        k *= 2
    return S


def chain(S, K=1):
    """[B, t, 4, 4] steps (each frame in its reference, to_steps) -> poses in frame 0's camera
    (S[:, 0] is ignored): keyframes chain through each other, every frame hangs off its reference."""
    S = torch.cat([torch.eye(4, dtype=S.dtype, device=S.device).expand(S.shape[0], 1, 4, 4), S[:, 1:]], 1)
    if K == 1:
        return scan(S)
    Pk = scan(S[:, ::K])
    return Pk[:, ref_index(S.shape[1], K, device=S.device) // K] @ S


# -- model ---------------------------------------------------------------------
class RelHead(nn.Module):
    def __init__(self, head, mode, window, prev_token, init_identity=False, residual=False, ref_every=1):
        super().__init__()
        self.head, self.mode, self.window, self.residual, self.K = head, mode, window, residual, ref_every
        if init_identity:
            # lingbot's output layer maps to absolute poses; restart translation and the quaternion's
            # vector part at zero and its scalar part at a positive constant (summed over the head
            # iterations, never 0), so every step starts as no motion. FoV rows stay: FoV is per frame.
            fc2 = head.pose_branch.fc2
            with torch.no_grad():
                fc2.weight[:7].zero_()
                fc2.bias[:6].zero_()
                fc2.bias[6] = 0.25
        self.prev = None
        if prev_token:
            self.prev = nn.Linear(D, D)
            nn.init.zeros_(self.prev.weight)
            nn.init.zeros_(self.prev.bias)

    def mask(self, t, dev):
        i = torch.arange(t, device=dev)
        causal = i[None, :] <= i[:, None]
        if self.mode == "rel":
            return causal & (i[:, None] - i[None, :] < self.window)
        return causal | tp.anchor_block(t, A, dev)          # lingbot's: causal, 8 anchors both ways

    def forward(self, cam, f0=0):
        """cam [B, t, D] of frames f0.. -> pose encodings [B, t, 9] per head iteration (rel mode:
        each frame in its reference). --prev-token adds the reference frame's token (a context
        frame whose reference precedes f0 gets the first frame's; predict_c2w discards those)."""
        x = cam
        if self.prev is not None:
            r = (ref_index(x.shape[1], self.K, f0, x.device) - f0).clamp(min=0)
            x = x + self.prev(x[:, r])
        m = self.mask(x.shape[1], x.device)
        # the dense frame mask in place of the head's own (as train_bidir_refiner's decode does)
        restore = [b._prepare_blockwise_causal_attn_mask for b in self.head.trunk]
        for b in self.head.trunk:
            b._prepare_blockwise_causal_attn_mask = lambda device, num_frames, frame_seqlen, num_frame_per_block: m
        try:
            return self.head.trunk_fn(self.head.token_norm(x), None, self.head.num_iterations,
                                      num_frame_per_block=1, num_frame_for_scale=-1, sliding_window_size=-1)
        finally:
            for b, f in zip(self.head.trunk, restore):
                b._prepare_blockwise_causal_attn_mask = f


# -- loss ----------------------------------------------------------------------
def angle(Ra, Rb):
    """Geodesic angle (rad), atan2 form: finite gradient at zero, where acos's is not."""
    E = Ra.transpose(-1, -2) @ Rb
    v = torch.stack([E[..., 2, 1] - E[..., 1, 2], E[..., 0, 2] - E[..., 2, 0], E[..., 1, 0] - E[..., 0, 1]], -1)
    return torch.atan2(torch.sqrt((v * v).sum(-1) + 1e-12), E.diagonal(dim1=-2, dim2=-1).sum(-1) - 1)


def step_matrices(p, hw, lingbot_steps=None):  # lingbot_steps: to_steps of lingbot's poses
    """Step encodings [..., t, 9] -> step camera-to-world [..., t, 4, 4], after lingbot's step if given."""
    S = tp.pose_enc_to_c2w(p, hw)
    return S if lingbot_steps is None else lingbot_steps @ S


def rel_loss(preds, G, fov, hw, args, lingbot_steps=None, gamma=0.6):
    """preds: per-iteration step encodings [B,t,9]; G [B,t,4,4] GT camera-to-world; fov [B,t,2].
    --long-w: pairs at gaps 64..1024 too, translation error relative to the GT distance (+1 anchor
    unit): a constant per-step bias grows linearly with the gap, so these terms are where it shows."""
    K = args.ref_every
    Sg = to_steps(G, K)
    n, total, parts = len(preds), 0.0, {}
    for k, p in enumerate(preds):
        P = step_matrices(p, hw, lingbot_steps)[:, 1:]
        lt = (P[..., :3, 3] - Sg[:, 1:, :3, 3]).abs().sum(-1)
        lr = angle(P[..., :3, :3], Sg[:, 1:, :3, :3])
        lf = (p[:, 1:, 7:] - fov[:, 1:]).abs().sum(-1)
        total = total + gamma ** (n - 1 - k) * (args.trans_w * lt + args.rot_w * lr + 0.5 * lf).mean()
    parts.update(step_t=lt.mean().detach(), step_r_deg=torch.rad2deg(lr).mean().detach())
    C = chain(step_matrices(preds[-1], hw, lingbot_steps), K)
    lp = []
    for d in range(1, min(args.pair_window, C.shape[1])):
        rp, rg = rigid_inv(C[:, :-d]) @ C[:, d:], rigid_inv(G[:, :-d]) @ G[:, d:]
        lp.append((args.trans_w * (rp[..., :3, 3] - rg[..., :3, 3]).abs().sum(-1)
                   + args.rot_w * angle(rp[..., :3, :3], rg[..., :3, :3])).mean())
    lpair = torch.stack(lp).mean()
    parts["pair"] = lpair.detach()
    total = total + args.pair_w * lpair
    ll = []
    for d in (64, 128, 256, 512, 1024):
        if not args.long_w or d >= C.shape[1]:
            break
        rp, rg = rigid_inv(C[:, :-d]) @ C[:, d:], rigid_inv(G[:, :-d]) @ G[:, d:]
        et = (rp[..., :3, 3] - rg[..., :3, 3]).norm(dim=-1) / (rg[..., :3, 3].norm(dim=-1) + 1)
        ll.append((args.trans_w * et + args.rot_w * angle(rp[..., :3, :3], rg[..., :3, :3])).mean())
    if ll:
        llong = torch.stack(ll).mean()
        parts["long"] = llong.detach()
        total = total + args.long_w * llong
    return total, parts


# -- evaluation ----------------------------------------------------------------
@torch.no_grad()
def predict_c2w(model, c, t=None, chunk=2048, with_enc=False):
    """Final-iteration poses of the first t frames as camera-to-world in frame 0's camera (float64).
    The windowed head runs in chunks with (window-1) x iterations frames of context: every output
    then sees exactly what one pass over the whole stream gives it (RoPE is relative).
    with_enc: also return the head's per-frame encodings [t, 9] (field of view at 7:9)."""
    cam = c.tok[:t]
    T = len(cam)
    if model.mode == "abs" or T <= chunk:
        spans = [(0, 0, T)]
    else:
        # + K: frames whose reference precedes the chunk get a stand-in token; keep them outside
        # every real frame's receptive field
        ctx = (model.window - 1) * model.head.num_iterations + (model.K if model.prev is not None else 0)
        spans = [(max(0, s - ctx), s, min(T, s + chunk)) for s in range(0, T, chunk)]
    enc = []
    for a, s, e in spans:
        with torch.autocast("cuda", dtype=torch.bfloat16):
            p = model(cam[None, a:e], f0=a)[-1][0].float()
        enc.append(p[s - a:])
    enc = torch.cat(enc)
    if model.mode == "abs":
        P = tp.pose_enc_to_c2w(enc, c.hw).double()
    else:
        M = step_matrices(enc, c.hw, to_steps(tp.pose_enc_to_c2w(c.lingbot[:T].float(), c.hw), model.K)
                          if model.residual else None).double()
        P = chain(M[None], model.K)[0].cpu()
    return (P, enc) if with_enc else P


def evaluate(model, sets, base):
    """{set@t: mean metrics}; base (lingbot's, computed once when empty) is filled in place."""
    model.eval()
    out = {}
    for name, clips, ts in sets:
        for t_req in ts:
            key = f"{name}@{t_req}" if t_req else name
            rows, lrows = [], []
            for c in clips:
                t = min(t_req, c.T) if t_req else c.T
                rows.append(tp.traj_metrics(predict_c2w(model, c, t), c.gt_c2w[:t], c.scale))
                if key not in base:
                    lrows.append(tp.traj_metrics(tp.pose_enc_to_c2w(c.lingbot[:t].float(), c.hw),
                                                 c.gt_c2w[:t], c.scale))
            out[key] = {m: float(np.mean([r[m] for r in rows])) for m in tp.METRICS}
            if key not in base:
                base[key] = {m: float(np.mean([r[m] for r in lrows])) for m in tp.METRICS}
    model.train()
    return out


# -- main ----------------------------------------------------------------------
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--mode", choices=["rel", "abs"], default="rel")
    ap.add_argument("--window", type=int, default=64, help="rel: frames the head attends to, itself included")
    ap.add_argument("--no-rope", action="store_true", help="rel: lingbot's head as is, without temporal RoPE")
    ap.add_argument("--prev-token", action="store_true")
    ap.add_argument("--residual", action="store_true", help="rel: correct lingbot's own steps (starts at lingbot)")
    ap.add_argument("--ref-every", type=int, default=1,
                    help="rel: each frame relative to the latest keyframe (every K-th frame); 1 = previous frame")
    ap.add_argument("--long-w", type=float, default=0.0, help="rel: weight of the gap 64..1024 pair terms")
    ap.add_argument("--no-init-identity", action="store_true",
                    help="rel: keep lingbot's absolute-pose output layer instead of starting every step at no motion")
    ap.add_argument("--rot-w", type=float, default=10.0)
    ap.add_argument("--trans-w", type=float, default=10.0)
    ap.add_argument("--pair-w", type=float, default=1.0)
    ap.add_argument("--pair-window", type=int, default=64)
    ap.add_argument("--norm", choices=["none", "extent"], default="extent", help="abs: as train_bidir_refiner")
    ap.add_argument("--trans-floor", type=float, default=1.0)
    ap.add_argument("--steps", type=int, default=8000)
    ap.add_argument("--batch", type=int, default=16)
    ap.add_argument("--accum", type=int, default=1, help="split each batch into this many sequential micro-batches")
    ap.add_argument("--lr", type=float, default=5e-5)
    ap.add_argument("--warmup", type=int, default=500)
    ap.add_argument("--t-min", type=int, default=64)
    ap.add_argument("--t-max", type=int, default=1048)
    ap.add_argument("--t-step", type=int, default=16)
    ap.add_argument("--val-every", type=int, default=500)
    ap.add_argument("--packs", default=PACKS)
    ap.add_argument("--pack-root", type=Path, default=br.PACK_ROOT)
    ap.add_argument("--inputs", type=Path, default=INPUTS, help="poserefine-v1 root (val24, nrgbd, kitti, vbr)")
    ap.add_argument("--longval-per-ds", type=int, default=8)
    ap.add_argument("--longval-lengths", default="320,640,1048")
    ap.add_argument("--nrgbd-lengths", default="64,128,256,320,500")
    ap.add_argument("--no-drive-val", action="store_true", help="skip KITTI / VBR (smoke runs)")
    ap.add_argument("--max-train", type=int, default=0, help="smoke: first N train clips per dataset")
    ap.add_argument("--seed", type=int, default=0)
    ap.add_argument("--no-wandb", action="store_true")
    args = ap.parse_args()
    torch.manual_seed(args.seed)
    rng = np.random.default_rng(args.seed)
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    (args.out / "config.json").write_text(json.dumps({k: str(v) for k, v in vars(args).items()}, indent=1))

    rope = args.mode == "rel" and not args.no_rope
    head = tp.build_head(tp.REPO / "ckpt/lingbot-map.pt", rope_on=rope, max_frames=4096 if rope else 1024)
    if args.mode == "rel" and args.window <= args.ref_every:
        print(f"[model] window {args.window} -> {args.ref_every + 1}: the head must see each frame's reference", flush=True)
        args.window = args.ref_every + 1
    model = RelHead(head, args.mode, args.window, args.prev_token,
                    init_identity=args.mode == "rel" and not args.no_init_identity,
                    residual=args.mode == "rel" and args.residual, ref_every=args.ref_every).to(dev)

    val_scenes = {l.strip().split("/")[-1].split("_c")[0] for l in br.VAL24.read_text().split()}
    jumps = json.loads(br.GT_JUMPS.read_text())
    train, longval = [], {}
    for ds in args.packs.split(","):
        if ds == "kitti360":
            print("[data] kitti360 skipped: metric GT, not in lingbot's anchor scale", flush=True)
            continue
        idx = dict(np.load(args.pack_root / ds / "index.npz"))
        t0 = time.time()
        tok = pack_cam_tokens(args.pack_root, ds)
        bad = {b[0].rsplit("__f", 1)[0] for b in jumps.get(ds, [])}
        mine, lv = [], []
        for i in range(len(idx["name"])):
            name = str(idx["name"][i])
            if name.rsplit("__f", 1)[0] in bad:
                continue
            c = TrainClip(ds, tok, idx, i)
            if (ds == "scannetpp" and name.split("__")[0] in val_scenes) or br.longval_split(ds, c.seq):
                lv.append(c)
            else:
                mine.append(c)
        mine = mine[: args.max_train] if args.max_train else mine
        train += mine
        lv.sort(key=lambda c: (-c.T, c.name))
        longval[ds] = [EvalClip(c.name, c.tok, c.gt_c2w, c.K, c.hw, c.scale, c.lingbot, dev)
                       for c in lv[: args.longval_per_ds]]
        print(f"[data] {ds}: {len(mine)} train clips, {len(lv)} held out, tokens {tok.shape} "
              f"({time.time() - t0:.0f}s)", flush=True)

    def load(d):
        return [EvalClip.from_npz(f, dev) for f in sorted((args.inputs / d / "specials").glob("*.npz"))
                if ".tmp" not in f.stem]

    sets = [("val24", load("val24"), [320]),
            ("nrgbd", load("nrgbd"), [int(x) for x in args.nrgbd_lengths.split(",")])]
    sets += [(f"longval_{ds}", cl, [int(x) for x in args.longval_lengths.split(",")])
             for ds, cl in longval.items() if cl]
    if not args.no_drive_val:
        sets += [("kitti", load("kitti"), [0]), ("vbr", load("vbr"), [0])]
    print("[data] eval: " + ", ".join(f"{n} {len(c)}" for n, c, _ in sets), flush=True)

    params = [p for p in model.parameters() if p.requires_grad]
    print(f"[model] mode={args.mode} rope={rope} window={args.window} prev_token={args.prev_token} "
          f"init_identity={args.mode == 'rel' and not args.no_init_identity} residual={args.residual} "
          f"ref_every={args.ref_every} long_w={args.long_w} "
          f"trainable {sum(p.numel() for p in params) / 1e6:.1f} M", flush=True)
    opt = torch.optim.AdamW(params, lr=args.lr, weight_decay=0.05, betas=(0.9, 0.95))
    tchoices = np.arange(args.t_min, args.t_max + 1, args.t_step)
    lengths = np.array([c.T for c in train])
    group = np.array([c.ds for c in train])
    gnames = sorted(set(group))

    run = None
    if not args.no_wandb:
        # never ./wandb in the repo root: that directory shadows the wandb package
        os.environ.setdefault("WANDB_DIR", "/group/compact-3dmem/wandb" if Path("/group/compact-3dmem/wandb").is_dir()
                              else str(args.out))
        try:
            import wandb
            run = wandb.init(project="lingbot-rel", group="relhead", name=args.out.name,
                             config={k: str(v) for k, v in vars(args).items()})
        except Exception as e:  # no egress / no login on this node
            print(f"[wandb] disabled: {e}", flush=True)

    base, hist, best = {}, [], float("inf")

    def validate(step):
        nonlocal best
        r = evaluate(model, sets, base)
        ratios = {k: v["ate"] / base[k]["ate"] for k, v in r.items()}
        # selection: geometric mean ATE ratio to lingbot over each set's longest stream
        longest = [k for k in r if "@" not in k] + [
            max((k for k in r if k.startswith(n + "@")), key=lambda k: int(k.split("@")[1]))
            for n in {k.split("@")[0] for k in r if "@" in k}]
        score = float(np.exp(np.mean([np.log(ratios[k]) for k in longest])))
        row = {"step": step, "select": score, **{f"{k}/{m}": v[m] for k, v in r.items() for m in v},
               **{f"ratio/{k}": v for k, v in ratios.items()}}
        hist.append(row)
        print(f"[val {step}] select {score:.4f} | " + " ".join(f"{k} {ratios[k]:.3f}" for k in longest), flush=True)
        if run is not None:
            run.log({("val/" + k if not k.startswith("ratio/") else k): v for k, v in row.items() if k != "step"},
                    step=step)
        ck = {"model": model.state_dict(), "args": vars(args), "step": step, "val": row}
        torch.save(ck, args.out / "last.pt")
        if step > 0 and score < best:   # step 0 of a residual arm is lingbot itself (score 1)
            best = score
            torch.save(ck, args.out / "best.pt")
        (args.out / "history.json").write_text(json.dumps({"lingbot": base, "hist": hist}, indent=1))

    validate(0)
    print("[lingbot] " + json.dumps({k: round(v["ate"], 4) for k, v in base.items()}), flush=True)

    t0 = time.time()
    for step in range(1, args.steps + 1):
        lr = args.lr * min(1.0, step / args.warmup) * 0.5 * (1 + math.cos(math.pi * step / args.steps))
        for g in opt.param_groups:
            g["lr"] = lr
        g = gnames[rng.integers(len(gnames))]                 # dataset first, then t, then clips >= t
        t = int(rng.choice(tchoices))
        pool_idx = np.flatnonzero((group == g) & (lengths >= t))
        if len(pool_idx) == 0:
            t = int(lengths[group == g].max())
            pool_idx = np.flatnonzero((group == g) & (lengths == t))
        idx = rng.choice(pool_idx, args.batch, replace=len(pool_idx) < args.batch)
        hw = train[idx[0]].hw          # unused by pose_enc_to_c2w (no intrinsics built)
        opt.zero_grad(set_to_none=True)
        loss, parts = 0.0, {}
        for mb in np.array_split(idx, args.accum):
            cam = torch.from_numpy(np.stack([train[i].tok[:t] for i in mb])).pin_memory().to(dev, non_blocking=True)
            G = torch.stack([train[i].gt_c2w[:t] for i in mb]).to(dev)
            enc = torch.stack([train[i].gt_enc[:t] for i in mb]).to(dev)
            with torch.autocast("cuda", dtype=torch.bfloat16):
                preds = model(cam)
            preds = [p.float() for p in preds]
            if args.mode == "rel":
                ls = None
                if args.residual:
                    L = torch.from_numpy(np.stack([train[i].lingbot[:t] for i in mb])).to(dev)
                    ls = to_steps(tp.pose_enc_to_c2w(L, hw), args.ref_every)
                l, pt = rel_loss(preds, G, enc[..., 7:], hw, args, lingbot_steps=ls)
            else:
                l, la, lrel = br.loss_fn(preds, enc, hw, 0, norm=args.norm == "extent", floor=args.trans_floor)
                pt = {"abs": la, "pair": lrel}
            (l / args.accum).backward()
            loss = loss + l.detach() / args.accum
            parts = {k: parts.get(k, 0) + v / args.accum for k, v in pt.items()}
        gn = torch.nn.utils.clip_grad_norm_(params, 1.0)
        opt.step()
        if step % 50 == 0:
            msg = " ".join(f"{k} {v.item():.4f}" for k, v in parts.items())
            print(f"[{step:6d}] loss {loss.item():.4f} {msg} t {t} {g} |g| {gn.item():.3f} "
                  f"{(time.time() - t0) / step:.2f}s/it {torch.cuda.max_memory_allocated() / 2**30:.1f}GiB", flush=True)
            if run is not None:
                run.log({"train/loss": loss.item(), **{f"train/{k}": v.item() for k, v in parts.items()},
                         "train/grad_norm": gn.item(), "train/lr": lr}, step=step)
        if step % args.val_every == 0 or step == args.steps:
            validate(step)


if __name__ == "__main__":
    main()
