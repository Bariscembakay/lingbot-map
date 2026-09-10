#!/usr/bin/env python3
"""nrgbd_recall_s2 benchmark runner for OUR StateMemory checkpoint.

Ingest the tier's frames into the 768-token state, then query that final state
with every ingested camera's raymap (images withheld) and score the decoded
geometry with the same metric code the baselines use.

Scale/frame convention -- the reason this runner needs no Umeyama alignment:
the baselines reconstruct in their own frame-0-anchored, up-to-scale worlds, so
their fused clouds are Sim(3)-aligned before scoring. Ours is a *posed* system
trained on caches whose GT is (frame-0 relative, OpenCV) divided by a canonical
per-scene scale s recorded in the cache meta. Multiplying the prediction by that
same s therefore lands exactly in the frame the benchmark's GT already lives in.
s is an input-side constant, not a fitted parameter -- hence align={"scale": s}
and no rotation/translation fit. This is the "scale-normalised, not scale-blind"
property, and it is why our column is declared separately in the table.
"""
from __future__ import annotations

import argparse
import json
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))


def load_poses(p: Path) -> np.ndarray:
    lines = [l for l in Path(p).read_text().splitlines() if l.strip()]
    n = len(lines) // 4
    return np.array([[list(map(float, lines[4 * i + j].split()))
                      for j in range(4)] for i in range(n)], np.float64)


@torch.no_grad()
def main() -> int:
    ap = argparse.ArgumentParser()
    ap.add_argument("--scene", required=True)
    ap.add_argument("--ckpt", required=True, type=Path)
    ap.add_argument("--cache", required=True, type=Path,
                    help="v4 tap cache built at the benchmark protocol "
                         "(stride 2, 500 frames)")
    ap.add_argument("--manifest-dir", type=Path,
                    default=REPO / "scripts/memory/recall_bench/manifests")
    ap.add_argument("--tiers", type=int, nargs="+", default=[100, 300, 500])
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--method", default="ours")
    args = ap.parse_args()

    import cv2
    from scipy.spatial import cKDTree
    from lingbot_map.memory.cut3r_state import StateMemory
    from scripts.memory.train_state import Clip, run_probe
    from scripts.memory.render_state import write_ply, heat

    man = json.loads((args.manifest_dir / f"{args.scene}.json").read_text())
    root = Path(man["root"])
    ids = man["ingest_frames"]
    focal = float(Path(root / man["focal"]).read_text().split()[0])
    # Same gl->cv flip as the baseline runner; the cache builder applies the
    # identical one, so cache GT and benchmark GT agree by construction.
    poses = load_poses(root / man["poses"]) @ np.diag([1.0, -1.0, -1.0, 1.0])
    c2w_gt = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]

    meta = json.loads((args.cache / "meta.json").read_text())
    s = float(meta["gt_scale"])
    cached_ids = list(meta["frame_ids"])
    if cached_ids[:len(ids)] != ids[:len(cached_ids)]:
        print(f"[fatal] cache frames do not match the manifest ingest list\n"
              f"  cache[:5]={cached_ids[:5]} manifest[:5]={ids[:5]}",
              file=sys.stderr)
        return 2
    print(f"[cache] {args.cache.name} frames={len(cached_ids)} gt_scale={s:.6f}")

    Hn, Wn = cv2.imread(str(root / f"images/img{ids[0]}.png")).shape[:2]
    K = np.array([[focal, 0, Wn / 2], [0, focal, Hn / 2], [0, 0, 1]], np.float64)
    dmin, dmax = man["depth_valid_range_m"]

    def gt_points(i, c2w_rel):
        d = cv2.imread(str(root / f"depth/depth{i}.png"), cv2.IMREAD_UNCHANGED)
        d = np.nan_to_num(d.astype(np.float64)) / man["depth_scale_mm"]
        valid = (d > dmin) & (d < dmax)
        jj, ii = np.meshgrid(np.arange(Wn), np.arange(Hn), indexing="xy")
        pts_c = np.stack([(jj - K[0, 2]) / K[0, 0] * d,
                          (ii - K[1, 2]) / K[1, 1] * d, d], -1)
        pw = pts_c @ c2w_rel[:3, :3].T + c2w_rel[:3, 3]
        rgb = cv2.cvtColor(cv2.imread(str(root / f"images/img{i}.png")),
                           cv2.COLOR_BGR2RGB)
        return pw.reshape(-1, 3), valid.reshape(-1), rgb.reshape(-1, 3), d

    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    ck = torch.load(args.ckpt, map_location="cpu", weights_only=False)
    t = ck["args"]
    head = t.get("head", "dpt")
    ns = SimpleNamespace(head=head,
                         raymap_convention=t.get("raymap_convention", "cut3r"))
    model = None

    for tier in args.tiers:
        od = args.out / args.method / f"{args.scene}_n{tier}"
        if (od / "metrics.json").exists():
            print(f"[skip] {args.scene}_n{tier} already done", flush=True)
            continue
        if tier > len(cached_ids):
            print(f"[skip] n{tier} exceeds cache ({len(cached_ids)})")
            continue

        # subsample=1: the cache is already at the protocol's stride 2, so the
        # checkpoint's training-time subsample must not be applied again.
        clip = Clip(args.cache, 1, device, tier, t.get("taps", "23"))
        if model is None:
            # read_depth/write_oneway must come from the checkpoint: relying on
            # the constructor defaults silently builds the wrong architecture
            # for any arm that varied them, and load_state_dict would then
            # either throw or (worse) match a subset.
            model = StateMemory(patch_size=14, tap_dim=clip.taps.shape[-1],
                                state_tokens=int(t.get("state_tokens", 768)),
                                dec_depth=int(t.get("dec_depth", 12)),
                                read_depth=int(t.get("read_depth", 2)),
                                write_oneway=bool(t.get("write_oneway", False)),
                                head_type=head, grad_ckpt=False)
            missing, unexpected = model.load_state_dict(ck["model"], strict=False)
            if missing or unexpected:
                print(f"[fatal] state_dict mismatch: {len(missing)} missing, "
                      f"{len(unexpected)} unexpected\n  missing[:5]={missing[:5]}"
                      f"\n  unexpected[:5]={unexpected[:5]}", file=sys.stderr)
                return 3
            model.to(device).eval()
            print(f"[ckpt] step {ck.get('step')} head={head} "
                  f"dec_depth={t.get('dec_depth')} read={t.get('read_depth')} "
                  f"oneway={t.get('write_oneway')}")

        t0 = time.time()
        state, spos = model.init_state(1, device)
        for i in range(len(clip)):
            state = model.write(state, spos, clip.tap(i), clip.patch_hw)
        print(f"[ours|{args.scene}|n{tier}] ingest {time.time()-t0:.0f}s",
              flush=True)

        qs = list(range(len(clip)))
        rm, _, _, _ = clip.probe_inputs(qs, ns.raymap_convention, anchor=0)
        rays = (clip.probe_rays(qs, anchor=0,
                               unit=head not in ("lingbot", "smallread_lingbot"))
                if head in ("raydepth", "lingbot", "smallread",
                            "smallread_lingbot") else None)
        chunks = []
        for a in range(0, len(qs), 16):
            b = min(a + 16, len(qs))
            r = tuple(x[a:b] for x in rays) if rays else None
            chunks.append(run_probe(model, state.expand(b - a, -1, -1),
                                    spos.expand(b - a, -1, -1), rm[a:b],
                                    (clip.h, clip.w), r))
        out = {k: torch.cat([c[k] for c in chunks], 0) for k in chunks[0]}
        # canonical -> metric, in the benchmark's own frame-0-relative frame
        pw_all = out["pts3d_in_other_view"].float().cpu().numpy() * s
        z_all = out["pts3d_in_self_view"].float().cpu().numpy()[..., 2] * s
        H, W = clip.h, clip.w

        preds, gts, cols, gcols, per_q = [], [], [], [], []
        depth_absrel, depth_d125 = [], []
        for q in range(tier):
            gp, gv, grgb, gdep = gt_points(ids[q], c2w_gt[q])
            gd = cv2.resize(gdep, (W, H), interpolation=cv2.INTER_NEAREST)
            m = (gd > dmin) & (gd < dmax)
            zs = z_all[q]
            if m.sum() > 100:
                r = np.abs(zs[m] - gd[m]) / gd[m]
                depth_absrel.append(float(r.mean()))
                depth_d125.append(float(
                    (np.maximum(zs[m] / gd[m], gd[m] / zs[m]) < 1.25).mean()))
            pw = pw_all[q].reshape(-1, 3)
            img = cv2.cvtColor(cv2.imread(str(root / f"images/img{ids[q]}.png")),
                               cv2.COLOR_BGR2RGB)
            rgb_small = cv2.resize(img, (W, H), interpolation=cv2.INTER_AREA)
            preds.append(pw[::3])
            cols.append(rgb_small.reshape(-1, 3)[::3])
            gts.append(gp[gv][::4])
            gcols.append(grgb[gv][::4])
            per_q.append(pw[::9])

        P = np.concatenate(preds); C = np.concatenate(cols)
        G = np.concatenate(gts); GC = np.concatenate(gcols)
        tree = cKDTree(G[::2])
        d_acc, _ = tree.query(P, workers=-1)
        tree_p = cKDTree(P[::2])
        d_comp, _ = tree_p.query(G[::4], workers=-1)
        lag = []
        for q, pq in enumerate(per_q):
            dq, _ = tree.query(pq, workers=-1)
            lag.append({"q": q, "age": tier - 1 - q, "acc_mean": float(dq.mean())})
        res = {
            "acc_mean": float(d_acc.mean()), "acc_med": float(np.median(d_acc)),
            "comp_mean": float(d_comp.mean()),
            "comp_med": float(np.median(d_comp)),
            "chamfer": float((d_acc.mean() + d_comp.mean()) / 2),
            "depth_absrel": float(np.mean(depth_absrel)),
            "depth_d125": float(np.mean(depth_d125)),
            "n_queries": tier, "lag": lag,
            "align": {"scale": s, "fitted": False},
        }
        print(f"[ours|{args.scene}|n{tier}] acc {d_acc.mean():.4f} "
              f"comp {d_comp.mean():.4f} absrel {np.mean(depth_absrel):.4f}",
              flush=True)

        od.mkdir(parents=True, exist_ok=True)
        (od / "metrics.json").write_text(json.dumps(
            {"method": args.method, "scene": args.scene, "tier": tier,
             "query_all_past": True, "stride": man["stride"],
             "ckpt": str(args.ckpt), "ckpt_step": int(ck.get("step", -1)),
             "posed": True, "gtpose": res, "selfpose": res}, indent=1))
        write_ply(od / "pred_cloud_rgb.ply", P, C.astype(np.uint8))
        lo, hi = np.percentile(d_acc, 5), np.percentile(d_acc, 95)
        write_ply(od / "pred_cloud_err.ply", P, heat(d_acc, float(lo), float(hi)))
        write_ply(od / "gt_cloud_rgb.ply", G, GC.astype(np.uint8))
        print(f"[viz] -> {od}", flush=True)
        del clip
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
