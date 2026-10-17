#!/usr/bin/env python3
"""Plan the 1048-frame cache clips of a dataset: per sequence, the frame stride and the clip frame lists.

The stride is the densest one that fits the whole sequence in one --length-frame clip,
ceil(N / length), as long as consecutive cached frames still overlap: it is capped by the
largest s <= --max-stride for which at least --keep of the pairs (i, i+s) rotate by <= --max-rot
degrees and translate by <= --max-trans x the sequence's median scene depth (median valid GT depth
of --depth-samples frames; --metric-depth metres without depth; larger strides are tried only
while every smaller one passes). When the overlap cap is below ceil(N / length), the sequence is
cut into consecutive --length-frame clips at the capped stride; a remainder shorter than
--min-tail frames is dropped.
Runs of invalid poses split a sequence (ScanNet v2). Sequences shorter than --min-len are skipped.
Usage: plan_clips.py --dataset tartanair-v2 --out plan.json
"""
from __future__ import annotations

import argparse
import json
from concurrent.futures import ProcessPoolExecutor
from pathlib import Path

import numpy as np

from clip_sources import source

A = None


def runs_of_valid(T, min_len):
    ok = np.isfinite(T).all((1, 2))
    out, a = [], None
    for i, v in enumerate(list(ok) + [False]):
        if v and a is None:
            a = i
        elif not v and a is not None:
            if i - a >= min_len:
                out.append((a, i))
            a = None
    return out


def choose_stride(T, depth_med):
    tmax = A.max_trans * depth_med
    best = 1
    for s in range(1, A.max_stride + 1):
        if len(T) <= s:
            break
        dt = np.linalg.norm(T[s:, :3, 3] - T[:-s, :3, 3], axis=1)
        m = np.einsum("nji,njk->nik", T[:-s, :3, :3], T[s:, :3, :3])
        ang = np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1)))
        if ((dt <= tmax) & (ang <= A.max_rot)).mean() < A.keep:
            break
        best = s
    return best


def plan_seq(seq):
    src = source(A.dataset)
    try:
        T = src.poses(seq)
    except Exception as e:  # a broken sequence is reported, not fatal
        return seq, [], {"error": f"{type(e).__name__}: {e}"}
    clips, info = [], {"frames": len(T), "runs": []}
    for a, b in runs_of_valid(T, A.min_len):
        if src.metric:
            dmed = A.metric_depth
        else:
            ids = np.linspace(a, b - 1, A.depth_samples).round().astype(int).tolist()
            try:
                d, _ = src.depth(seq, ids)
                v = d[d > 0]
                dmed = float(np.median(v)) if v.size else A.metric_depth
            except Exception as e:
                info.setdefault("depth_errors", []).append(str(e)[:120])
                dmed = A.metric_depth
        s_max = choose_stride(T[a:b], dmed)
        s = min(s_max, max(1, -(-(b - a) // A.length)))     # densest stride that fits the run in one clip, if overlap allows
        ids = list(range(a, b, s))
        chunks = [ids[k:k + A.length] for k in range(0, len(ids), A.length)]
        if len(chunks) > 1 and len(chunks[-1]) < A.min_tail:
            chunks = chunks[:-1]
        chunks = [c for c in chunks if len(c) >= A.min_len]
        clips += [{"seq": seq, "frames": c, "stride": s, "depth_med": dmed} for c in chunks]
        info["runs"].append({"range": [a, b], "stride": s, "stride_overlap_max": s_max, "depth_med": dmed, "clips": len(chunks)})
    return seq, clips, info


def init(args):
    global A
    A = args


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--length", type=int, default=1048)
    ap.add_argument("--min-len", type=int, default=64)
    ap.add_argument("--min-tail", type=int, default=256)
    ap.add_argument("--max-stride", type=int, default=32)
    ap.add_argument("--max-rot", type=float, default=20.0)
    ap.add_argument("--max-trans", type=float, default=0.25)
    ap.add_argument("--keep", type=float, default=0.95)
    ap.add_argument("--depth-samples", type=int, default=5)
    ap.add_argument("--metric-depth", type=float, default=15.0, help="scene depth (m) when there is no GT depth")
    ap.add_argument("--workers", type=int, default=16)
    args = ap.parse_args()
    seqs = source(args.dataset).seqs()
    with ProcessPoolExecutor(args.workers, initializer=init, initargs=(args,)) as ex:
        res = list(ex.map(plan_seq, seqs, chunksize=2))
    clips = [c for _, cs, _ in res for c in cs]
    info = {s: i for s, _, i in res}
    n = np.array([len(c["frames"]) for c in clips]) if clips else np.zeros(1, int)
    st = np.array([c["stride"] for c in clips]) if clips else np.zeros(1, int)
    summ = {"dataset": args.dataset, "sequences": len(seqs), "errors": sum("error" in i for i in info.values()),
            "clips": len(clips), "frames": int(n.sum()), "full_length_clips": int((n == args.length).sum()),
            "clip_frames_median": float(np.median(n)), "stride_median": float(np.median(st)), "stride_max": int(st.max()),
            "est_GB_specials_desc": round(float(n.sum()) * 125e3 / 1e9, 1)}
    print("[plan] " + ", ".join(f"{k} {v}" for k, v in summ.items()), flush=True)
    for s, i in info.items():
        if "error" in i:
            print(f"   error {s}: {i['error']}", flush=True)
    args.out.parent.mkdir(parents=True, exist_ok=True)
    args.out.write_text(json.dumps({"summary": summ, "args": vars(args) | {"out": str(args.out)}, "clips": clips, "per_seq": info}))


if __name__ == "__main__":
    main()
