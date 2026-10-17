#!/usr/bin/env python3
"""Do KITTI odometry (benchmark) and KITTI-360 (training) share places?

No GPS is on disk for either, so the test is visual: MegaLoc descriptors of every KITTI
odometry frame (extract_loop_descriptors.py output) are matched against MegaLoc of KITTI-360
image_00 frames (every --stride-th, cover-fit to 504x280 as build_seq_inputs.py feeds lingbot).
Null control: the same KITTI frames matched against VBR (Rome, cannot overlap); a KITTI frame
counts as a shared place when its best KITTI-360 similarity exceeds the null's 99.9th
percentile, and a shared street is a run of >= --min-run consecutive such frames whose matches
are also consecutive in one KITTI-360 drive. Writes per-sequence stats and side-by-side jpgs
of the strongest matches.
Usage: cross_dataset_overlap.py --desc DIR --out DIR [--stride 2]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

import cv2
import numpy as np
import torch
import torch.nn.functional as F

sys.path.insert(0, str(Path(__file__).resolve().parent))
import build_seq_inputs as bs  # noqa: E402

K360 = Path("/data/kitti360")
KITTI_RGB = Path("/group/compact-3dmem/campaigns/lingbot_map/default/kitti")


def load_k360(path):
    rgb, _ = bs.cover_fit(cv2.cvtColor(cv2.imread(str(path)), cv2.COLOR_BGR2RGB), np.eye(3))
    return torch.from_numpy(np.ascontiguousarray(rgb)).permute(2, 0, 1)


class Frames(torch.utils.data.Dataset):
    def __init__(self, paths):
        self.paths = paths

    def __len__(self):
        return len(self.paths)

    def __getitem__(self, i):
        return load_k360(self.paths[i])


@torch.no_grad()
def megaloc_k360(paths, dev, bs_=64):
    m = torch.hub.load("gmberton/MegaLoc", "get_trained_model").to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    out = []
    for x in torch.utils.data.DataLoader(Frames(paths), batch_size=bs_, num_workers=8):
        x = (x.to(dev).float() / 255 - mean) / std
        out.append(F.normalize(m(x).float(), dim=-1).cpu())
    return torch.cat(out)


def best_match(q, db, chunk=4096):
    s, i = [], []
    for a in range(0, len(q), chunk):
        v, j = (q[a:a + chunk] @ db.T).max(1)
        s.append(v)
        i.append(j)
    return torch.cat(s), torch.cat(i)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--desc", type=Path, required=True, help="extract_loop_descriptors.py output (kitti/, vbr/)")
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--stride", type=int, default=2)
    ap.add_argument("--min-run", type=int, default=10)
    args = ap.parse_args()
    dev = torch.device("cuda")
    args.out.mkdir(parents=True, exist_ok=True)
    cache = args.out / f"k360_megaloc_s{args.stride}.pt"
    if cache.exists():
        k = torch.load(cache)
    else:
        paths, ids = [], []
        for d in sorted(K360.glob("*_sync")):
            fs = sorted((d / "image_00/data_rect").glob("*.png"))[::args.stride]
            paths += fs
            ids += [(d.name, int(f.stem)) for f in fs]
        k = {"desc": megaloc_k360(paths, dev), "ids": ids}
        torch.save(k, cache)
    db, ids = k["desc"].to(dev), k["ids"]
    print(f"[k360] {len(ids)} frames from {len({d for d, _ in ids})} drives", flush=True)
    vbr = F.normalize(torch.cat([torch.from_numpy(np.load(f)["megaloc"]).float()
                                 for f in sorted((args.desc / "vbr").glob("*.npz"))]).to(dev), dim=-1)
    res, null_all = {}, []
    kitti = {f.stem: F.normalize(torch.from_numpy(np.load(f)["megaloc"]).float().to(dev), dim=-1)
             for f in sorted((args.desc / "kitti").glob("*.npz"))}
    for name, q in kitti.items():
        null_all.append(best_match(q, vbr)[0].cpu())
    null = torch.cat(null_all)
    tau = float(torch.quantile(null, 0.999))
    print(f"[null] KITTI->VBR best-match similarity: median {null.median():.3f}, p99.9 {tau:.3f}, max {null.max():.3f}", flush=True)
    top = []
    for name, q in kitti.items():
        s, j = best_match(q, db)
        s, j = s.cpu().numpy(), j.cpu().numpy()
        hit = s > tau
        runs, a = [], None
        for t in range(len(s) + 1):
            ok = t < len(s) and hit[t] and (a is None or (ids[j[t]][0] == ids[j[t - 1]][0]
                                                          and abs(ids[j[t]][1] - ids[j[t - 1]][1]) <= 20 * args.stride))
            if ok and a is None:
                a = t
            elif not ok and a is not None:
                if t - a >= args.min_run:
                    runs.append((a, t - 1, ids[j[a]][0], ids[j[a]][1], ids[j[t - 1]][1]))
                a = t if (t < len(s) and hit[t]) else None
        res[name] = {"frames": len(s), "frac_above_null_p999": float(hit.mean()), "best_sim": float(s.max()),
                     "median_sim": float(np.median(s)), "runs": [list(map(lambda v: v if isinstance(v, str) else int(v), r)) for r in runs],
                     "frames_in_runs": int(sum(r[1] - r[0] + 1 for r in runs))}
        print(f"[{name}] {len(s)} frames: best {s.max():.3f}, median {np.median(s):.3f}, above null p99.9 {hit.mean():.3%}, "
              f"{len(runs)} consecutive runs covering {res[name]['frames_in_runs']} frames"
              + (f"; e.g. frames {runs[0][0]}-{runs[0][1]} ~ {runs[0][2]} {runs[0][3]}-{runs[0][4]}" if runs else ""), flush=True)
        top += [(float(s[t]), name, t, ids[j[t]]) for t in np.argsort(-s)[:4]]
    for r, (sv, name, t, (drive, fid)) in enumerate(sorted(top, reverse=True)[:12]):
        a = cv2.imread(str(KITTI_RGB / name.split("_")[1] / "gt/rgb" / f"{t:06d}.png"))
        b = cv2.imread(str(K360 / drive / "image_00/data_rect" / f"{fid:010d}.png"))
        if a is None or b is None:
            continue
        h = 280
        a = cv2.resize(a, (int(a.shape[1] * h / a.shape[0]), h))
        b = cv2.resize(b, (int(b.shape[1] * h / b.shape[0]), h))
        img = np.concatenate([a, b], 1)
        cv2.putText(img, f"{name} f{t}  vs  {drive} f{fid}  sim {sv:.3f}", (10, 25), cv2.FONT_HERSHEY_SIMPLEX, 0.7, (0, 255, 255), 2)
        cv2.imwrite(str(args.out / f"top{r:02d}_{name}_{t}_{drive}_{fid}.jpg"), img)
    (args.out / "overlap.json").write_text(json.dumps({"tau_null_p999": tau, "null_median": float(null.median()),
                                                      "stride": args.stride, "per_seq": res}, indent=1))


if __name__ == "__main__":
    main()
