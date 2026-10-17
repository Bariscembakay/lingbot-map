#!/usr/bin/env python3
"""Correction-precision probe: how well does a small MLP regress the relative pose of a true
revisit pair from lingbot's per-frame special tokens alone?

Pairs (i, j): GT revisits, frame j driven >= --min-path m before i and |c_i - c_j| < --radius,
at most --pairs-per-query random j per query. Target: pose of i in j's camera frame (metric).
Train on KITTI-360 runs (lingbot saw KITTI-360), drive --val-drive held out for model
selection; test on KITTI 00-10 and VBR (seen by neither).
Model: per-frame encoder (shared) -> [a, b, a-b, a*b] -> MLP -> 6D rotation + translation;
loss L1 on translation (m) + L1 on rotation-matrix entries; pairs are also fed swapped
with the inverted target.
Baselines: identity; lingbot's own relative pose from its drifted trajectory, translation
put in metres by the scene's Sim(3) scale against GT (an oracle scale, favours the baseline).
Usage: revisit_pose_probe.py --features t23_all6,t23_cam,alltaps_cam --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402

FEATURES = {  # name -> (tap indices into the specials stack, token indices)
    "t23_cam": ([3], [0]),
    "t23_all6": ([3], list(range(6))),
    "alltaps_cam": ([0, 1, 2, 3], [0]),
    "alltaps_all6": ([0, 1, 2, 3], list(range(6))),
}


def path_len(c):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])


def revisit_pairs(gt_c, radius, min_path, per_query, rng):
    s = path_len(gt_c)
    out = []
    for i in range(len(gt_c)):
        db = np.nonzero(s[i] - s >= min_path)[0]
        if not len(db):
            continue
        j = db[np.linalg.norm(gt_c[db] - gt_c[i], axis=1) < radius]
        if len(j):
            out += [(i, jj) for jj in rng.choice(j, min(per_query, len(j)), replace=False)]
    return np.array(out, dtype=np.int64).reshape(-1, 2)


def rel(T, p):
    """pose of p[:,0] in the camera frame of p[:,1]"""
    return np.linalg.inv(T[p[:, 1]]) @ T[p[:, 0]]


def load(root, files, args, rng):
    """-> dict(feature -> frame table [F, D] fp16), pair rows into it [P, 2], target [P, 4, 4],
    lingbot rel [P, 4, 4], scene id [P], scene names"""
    feats = {k: [] for k in args.features}
    rows, tgt, lb, sid, names, nf = [], [], [], [], [], 0
    for f in files:
        z = np.load(f)
        G = z["gt_c2w"].astype(np.float64)
        p = revisit_pairs(G[:, :3, 3], args.radius, args.min_path, args.pairs_per_query, rng)
        if not len(p):
            continue
        L = tp.pose_enc_to_c2w(torch.from_numpy(z["pose_enc"]).float(), tuple(z["hw"])).double().numpy()
        s, _, _ = tp.umeyama(L[:, :3, 3], G[:, :3, 3])
        L[:, :3, 3] *= s
        frames = np.unique(p)
        spec = z["specials"][frames]
        for k in args.features:
            taps, toks = FEATURES[k]
            feats[k].append(spec[:, taps][:, :, toks].reshape(len(frames), -1))
        rows.append(np.searchsorted(frames, p) + nf); nf += len(frames)
        tgt.append(rel(G, p)); lb.append(rel(L, p))
        sid += [len(names)] * len(p); names.append(f.stem)
    return ({k: np.concatenate(v) for k, v in feats.items()}, np.concatenate(rows), np.concatenate(tgt),
            np.concatenate(lb), np.array(sid), names)


def rot6d_to_R(x):
    a, b = F.normalize(x[:, :3], dim=-1), x[:, 3:6]
    b = F.normalize(b - (a * b).sum(-1, keepdim=True) * a, dim=-1)
    return torch.stack([a, b, torch.cross(a, b, dim=-1)], -1)


class PairMLP(nn.Module):
    def __init__(self, d, h=512):
        super().__init__()
        self.enc = nn.Sequential(nn.LayerNorm(d), nn.Linear(d, h), nn.GELU(), nn.Linear(h, h))
        self.head = nn.Sequential(nn.Linear(4 * h, 1024), nn.GELU(), nn.Linear(1024, 512), nn.GELU(), nn.Linear(512, 9))

    def forward(self, x):
        a, b = self.enc(x[:, 0]), self.enc(x[:, 1])
        o = self.head(torch.cat([a, b, a - b, a * b], -1))
        return rot6d_to_R(o[:, :6]), o[:, 6:]


def errors(R, t, T):
    """rotation (deg) and translation (m) error against target T [P,4,4]"""
    Rg, tg = T[:, :3, :3], T[:, :3, 3]
    c = ((np.einsum("pij,pij->p", R, Rg)) - 1) / 2
    return np.degrees(np.arccos(np.clip(c, -1, 1))), np.linalg.norm(t - tg, axis=1)


def summary(re, te, sid=None, names=None):
    d = {"n": int(len(re)), "rot_med": float(np.median(re)), "rot_mean": float(re.mean()),
         "t_med": float(np.median(te)), "t_mean": float(te.mean()),
         "ok_1m_2deg": float(((te < 1) & (re < 2)).mean()), "ok_2m_5deg": float(((te < 2) & (re < 5)).mean())}
    if sid is not None:
        d["per_scene"] = {names[s]: summary(re[sid == s], te[sid == s]) for s in np.unique(sid)}
    return d


def predict(model, table, rows, norm, dev, bs=4096):
    R, t = [], []
    with torch.no_grad():
        for a in range(0, len(rows), bs):
            r, tt = model(norm(table[rows[a:a + bs]]))
            R.append(r.double().cpu().numpy()); t.append(tt.double().cpu().numpy())
    return np.concatenate(R), np.concatenate(t)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("/data/lingbot-poserefine-v1"))
    ap.add_argument("--features", default="t23_cam,t23_all6,alltaps_cam,alltaps_all6")
    ap.add_argument("--val-drive", default="drive_0009")
    ap.add_argument("--test-sets", default="kitti,vbr")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--min-path", type=float, default=200.0)
    ap.add_argument("--pairs-per-query", type=int, default=5)
    ap.add_argument("--epochs", type=int, default=40)
    ap.add_argument("--lr", type=float, default=1e-3)
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    args.features = args.features.split(",")
    dev = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    rng = np.random.default_rng(0)
    torch.manual_seed(0)
    k360 = sorted(f for f in (args.inputs / "kitti360/specials").glob("*.npz") if ".tmp" not in f.stem)
    tr = load(args.inputs, [f for f in k360 if args.val_drive not in f.stem], args, rng)
    va = load(args.inputs, [f for f in k360 if args.val_drive in f.stem], args, rng)
    te = {s: load(args.inputs, sorted(f for f in (args.inputs / s / "specials").glob("*.npz") if ".tmp" not in f.stem),
                  args, rng) for s in args.test_sets.split(",")}
    print(f"pairs: train {len(tr[1])} ({len(tr[5])} runs), val {len(va[1])}, "
          + ", ".join(f"{s} {len(v[1])}" for s, v in te.items()), flush=True)
    res = {"args": {k: (str(v) if isinstance(v, Path) else v) for k, v in vars(args).items()}, "baselines": {}, "mlp": {}}
    for name, d in [("val_kitti360", va)] + list(te.items()):
        T, L = d[2], d[3]
        I = np.tile(np.eye(4), (len(T), 1, 1))
        res["baselines"][name] = {"identity": summary(*errors(I[:, :3, :3], I[:, :3, 3], T), d[4], d[5]),
                                  "lingbot_drifted_sim3scale": summary(*errors(L[:, :3, :3], L[:, :3, 3], T), d[4], d[5])}
    for feat in args.features:
        table = tr[0][feat]
        mu = torch.from_numpy(table.astype(np.float32).mean(0)).to(dev)
        sd = torch.from_numpy(table.astype(np.float32).std(0)).to(dev) + 1e-3
        norm = lambda x: (torch.from_numpy(x).to(dev).float() - mu) / sd  # noqa: E731
        X = norm(table)
        rows = torch.from_numpy(tr[1]).to(dev)
        Rg = torch.from_numpy(tr[2][:, :3, :3]).float().to(dev)
        tg = torch.from_numpy(tr[2][:, :3, 3]).float().to(dev)
        Ti = np.linalg.inv(tr[2])
        Rgi, tgi = torch.from_numpy(Ti[:, :3, :3]).float().to(dev), torch.from_numpy(Ti[:, :3, 3]).float().to(dev)
        model = PairMLP(X.shape[-1]).to(dev)
        opt = torch.optim.AdamW(model.parameters(), lr=args.lr, weight_decay=0.05)
        steps = args.epochs * ((len(rows) + 255) // 256)
        sched = torch.optim.lr_scheduler.OneCycleLR(opt, args.lr, total_steps=steps)
        best, best_state, hist = float("inf"), None, []
        for ep in range(args.epochs):
            model.train()
            for b in torch.randperm(len(rows), device=dev).split(256):
                swap = torch.rand(len(b), device=dev) < 0.5
                x = X[torch.where(swap[:, None], rows[b].flip(1), rows[b])]
                R_t = torch.where(swap[:, None, None], Rgi[b], Rg[b])
                t_t = torch.where(swap[:, None], tgi[b], tg[b])
                R, t = model(x)
                loss = (t - t_t).abs().mean() + (R - R_t).abs().mean() * 10
                opt.zero_grad(); loss.backward(); opt.step(); sched.step()
            model.eval()
            v = summary(*errors(*predict(model, va[0][feat], va[1], norm, dev), va[2]))
            score = v["t_med"] + v["rot_med"] / 5
            hist.append({"epoch": ep, "loss": float(loss), **v})
            if score < best:
                best, best_state = score, {k: w.clone() for k, w in model.state_dict().items()}
        model.load_state_dict(best_state)
        r = {"dim": int(X.shape[-1]), "history": hist,
             "val_kitti360": summary(*errors(*predict(model, va[0][feat], va[1], norm, dev), va[2]))}
        for s, d in te.items():
            r[s] = summary(*errors(*predict(model, d[0][feat], d[1], norm, dev), d[2]), d[4], d[5])
        res["mlp"][feat] = r
        print(f"[{feat}] dim {X.shape[-1]}: " + "; ".join(
            f"{s} t_med {r[s]['t_med']:.2f} m rot_med {r[s]['rot_med']:.2f} deg ok(2m,5deg) {r[s]['ok_2m_5deg']:.2f}"
            for s in ["val_kitti360"] + list(te)), flush=True)
    for s, b in res["baselines"].items():
        print(f"[baseline {s}] " + "; ".join(f"{k} t_med {v['t_med']:.2f} rot_med {v['rot_med']:.2f} ok(2m,5deg) {v['ok_2m_5deg']:.2f}"
                                             for k, v in b.items()), flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
