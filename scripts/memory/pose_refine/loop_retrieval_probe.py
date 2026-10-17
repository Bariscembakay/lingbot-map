#!/usr/bin/env python3
"""Loop-closure detection probe: does nearest-neighbour search over a per-frame descriptor
find the true revisited frame?

Revisit (GT): query i has a frame j driven >= --min-path m earlier (GT path length) with
|c_i - c_j| < --radius. Database of query i = all frames >= --min-path m of driving earlier.
Per descriptor and scene, cosine search over the database gives
  R@1/5/10   fraction of revisit queries whose top-K holds a frame within --radius
  PR         over every query with a non-empty database, thresholding its top-1 score:
             a detection is correct when the top-1 frame is within --radius, positives
             are the revisit queries -> average precision and max recall at 100% precision.
  pair AUC   separability: ROC-AUC of descriptor similarity between same-place pairs (within
             --radius, >= --min-path apart) and different-place pairs (> --neg-dist apart),
             sampled per scene; plus the mean similarity of each
Descriptors: lingbot special tokens (tap x token group x frame/global/both halves),
DINOv2 CLS / mean / GeM, lingbot's own (drifted) camera centres as a reference, and with
--desc the extract_loop_descriptors.py set (lingbot's DINO, pooled tap-23 patches, pooled
global-attention K/V, camera-token K/V, MegaLoc).
"centered" variants subtract the scene's mean descriptor (transductive, probe only).
Usage: loop_retrieval_probe.py --sets kitti,vbr --out x.json
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402

TAPS = (4, 11, 17, 23)
GROUPS = {"cam": [0], "regs": [1, 2, 3, 4], "all6": list(range(6))}
DEV = torch.device("cuda" if torch.cuda.is_available() else "cpu")
HALVES = {"frame": slice(0, 1024), "global": slice(1024, 2048), "both": slice(0, 2048)}


def path_len(c):
    return np.concatenate([[0.0], np.cumsum(np.linalg.norm(np.diff(c, axis=0), axis=1))])


def descriptors(spec, dino, lingbot_c, center, extra=None):
    d = {}
    for ti, tap in enumerate(TAPS):
        for g, toks in GROUPS.items():
            for h, sl in HALVES.items():
                d[f"lingbot_t{tap}_{g}_{h}"] = spec[:, ti, toks, sl]
    if dino is not None:
        for k in ("cls", "mean", "gem"):
            d[f"dino_{k}"] = dino[k]
    if extra is not None:
        for k in extra.files:
            if k not in ("gem_p", "layers"):
                d[k] = extra[k]
    for k, v in d.items():
        x = torch.from_numpy(np.asarray(v)).to(DEV).float().flatten(1)
        if center:
            x = x - x.mean(0, keepdim=True)
        yield k, F.normalize(x, dim=-1)
    if not center:
        yield "lingbot_traj", torch.from_numpy(lingbot_c).to(DEV).float()


def search(x, s, min_path, k=10, chunk=2048):
    """Top-k database frames per query (score high = similar); queries with no database get -inf."""
    n = len(x)
    s = torch.from_numpy(s).to(DEV)
    top_s, top_i = torch.full((n, k), -float("inf"), device=DEV), torch.zeros((n, k), dtype=torch.long, device=DEV)
    traj = x.shape[1] == 3
    for a in range(0, n, chunk):
        q = x[a:a + chunk]
        sim = -torch.cdist(q, x) if traj else q @ x.T
        ok = s[a:a + chunk, None] - s[None, :] >= min_path
        sim = sim.masked_fill(~ok, -float("inf"))
        v, i = sim.topk(k, dim=1)
        top_s[a:a + chunk], top_i[a:a + chunk] = v, i
    return top_s.cpu().numpy(), top_i.cpu().numpy()


def pair_auc(x, gt_c, s, radius, min_path, neg_dist, rng, n_pos=20000, n_neg=100000):
    """ROC-AUC of similarity, same-place pairs vs pairs > neg_dist apart, plus their mean similarities."""
    n = len(gt_c)
    G = torch.from_numpy(gt_c).to(DEV)
    S = torch.from_numpy(s).to(DEV)
    pos = []
    for a in range(0, n, 2048):
        d = torch.cdist(G[a:a + 2048], G)
        m = (d < radius) & (S[a:a + 2048, None] - S[None] >= min_path)
        i, j = m.nonzero(as_tuple=True)
        pos.append(torch.stack([i + a, j], 1))
    pos = torch.cat(pos)
    if len(pos) == 0:
        return None
    pos = pos[torch.from_numpy(rng.choice(len(pos), min(n_pos, len(pos)), replace=False)).to(DEV)]
    cand = torch.from_numpy(rng.integers(0, n, (4 * n_neg, 2))).to(DEV)
    far = (G[cand[:, 0]] - G[cand[:, 1]]).norm(dim=1) > neg_dist
    neg = cand[far][:n_neg]
    traj = x.shape[1] == 3
    sim = (lambda p: -(x[p[:, 0]] - x[p[:, 1]]).norm(dim=1)) if traj else (lambda p: (x[p[:, 0]] * x[p[:, 1]]).sum(1))
    sp, sn = sim(pos).float(), sim(neg).float()
    allv = torch.cat([sp, sn])
    rank = allv.argsort().argsort().float() + 1
    auc = (rank[:len(sp)].sum() - len(sp) * (len(sp) + 1) / 2) / (len(sp) * len(sn))
    return {"pair_auc": float(auc), "sim_pos": float(sp.mean()), "sim_neg": float(sn.mean()), "n_pos_pairs": len(sp)}


def pr_stats(score, correct, positive):
    """score/correct over queries with a database; positive = query is a GT revisit."""
    o = np.argsort(-score)
    tp_ = np.cumsum(correct[o])
    prec = tp_ / np.arange(1, len(o) + 1)
    rec = tp_ / max(positive.sum(), 1)
    ap = float(np.sum(np.diff(np.concatenate([[0], rec])) * prec))
    full = prec >= 1.0 - 1e-12
    return {"ap": ap, "r_at_p100": float(rec[full].max()) if full.any() else 0.0}


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--inputs", type=Path, default=Path("/data/lingbot-poserefine-v1"))
    ap.add_argument("--sets", default="kitti,vbr")
    ap.add_argument("--radius", type=float, default=10.0)
    ap.add_argument("--min-path", type=float, default=200.0)
    ap.add_argument("--neg-dist", type=float, default=50.0)
    ap.add_argument("--desc", type=Path, default=None, help="extract_loop_descriptors.py output dir (<set>/<scene>.npz)")
    ap.add_argument("--no-centered", action="store_true")
    ap.add_argument("--out", type=Path, required=True)
    args = ap.parse_args()
    res = {"radius": args.radius, "min_path": args.min_path}
    for name in args.sets.split(","):
        per = {}
        for f in sorted((args.inputs / name / "specials").glob("*.npz")):
            if ".tmp" in f.stem:
                continue
            z = np.load(f)
            dz = args.inputs / name / "dino" / f.name
            dino = np.load(dz) if dz.exists() else None
            ez = args.desc / name / f.name if args.desc else None
            extra = np.load(ez) if ez is not None and ez.exists() else None
            rng = np.random.default_rng(0)
            gt_c = z["gt_c2w"][:, :3, 3].astype(np.float64)
            lb_c = tp.pose_enc_to_c2w(torch.from_numpy(z["pose_enc"]).float(), tuple(z["hw"]))[:, :3, 3].numpy()
            s = path_len(gt_c)
            n = len(gt_c)
            G = torch.from_numpy(gt_c).to(DEV)
            has_db = s - s[0] >= args.min_path
            positive = np.zeros(n, bool)
            for a in range(0, n, 2048):
                dist = torch.cdist(G[a:a + 2048], G)
                ok = torch.from_numpy(s[a:a + 2048, None] - s[None, :] >= args.min_path).to(DEV)
                positive[a:a + 2048] = ((dist < args.radius) & ok).any(1).cpu().numpy()
            r = {"frames": n, "revisit_queries": int(positive.sum()), "queries_with_db": int(has_db.sum())}
            if positive.sum() == 0:
                per[f.stem] = r
                print(f"[{name}] {f.stem}: no revisits", flush=True)
                continue
            for center in ((False,) if args.no_centered else (False, True)):
                for k, x in descriptors(z["specials"], dino, lb_c, center, extra):
                    key = k + ("_centered" if center else "")
                    ts, ti = search(x, s, args.min_path)
                    hit = np.linalg.norm(gt_c[ti] - gt_c[:, None], axis=-1) < args.radius      # [n, 10]
                    m = {f"r@{K}": float(hit[positive, :K].any(1).mean()) for K in (1, 5, 10)}
                    m.update(pr_stats(ts[has_db, 0], hit[has_db, 0], positive[has_db]))
                    pa = pair_auc(x, gt_c, s, args.radius, args.min_path, args.neg_dist, rng)
                    if pa:
                        m.update(pa)
                    r[key] = m
            per[f.stem] = r
            best = sorted(((v["r@1"], k) for k, v in r.items() if isinstance(v, dict)), reverse=True)[:5]
            print(f"[{name}] {f.stem}: {n} frames, {r['revisit_queries']} revisit queries; best R@1 "
                  + ", ".join(f"{k} {v:.3f}" for v, k in best)
                  + f"; dino_cls {r.get('dino_cls', {}).get('r@1', float('nan')):.3f}"
                  + f"; lingbot_traj {r['lingbot_traj']['r@1']:.3f}", flush=True)
        # pooled over all revisit queries of the set (each scene weighted by its query count)
        keys = sorted({k for v in per.values() for k, m in v.items() if isinstance(m, dict)})
        pooled = {}
        for k in keys:
            w = [(v["revisit_queries"], v[k]) for v in per.values() if k in v]
            W = sum(q for q, _ in w)
            mks = set.intersection(*(set(m) for _, m in w))
            pooled[k] = {mk: sum(q * m[mk] for q, m in w) / W for mk in sorted(mks)}
        res[name] = {"pooled": pooled, "per_scene": per}
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
