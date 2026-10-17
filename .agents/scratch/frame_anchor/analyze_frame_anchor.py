#!/usr/bin/env python
"""Scores the frame-anchor diagnostic from cut3r_anchor_dump.py records.

Two hypotheses for CUT3R's long-sequence degradation, and what separates them:
  origin  -- frame-0 outputs (X_world, pose) lose precision with distance from
             the origin; X_self, which is egocentric, does not.
  state   -- degradation follows stream index (state saturation), so it hits
             X_self too and does not recover when the camera returns.
The palindrome return trip re-shows each image at its forward distance but a
later index; note it is also a revisit, which CUT3R's Table 5 says can help.
"""
import argparse
import glob
import json
import os
import os.path as osp

import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402
import numpy as np  # noqa: E402
from scipy.stats import rankdata  # noqa: E402

DIST_BINS = {
    "nrgbd": [0, 0.5, 1, 1.5, 2, 3, 5, np.inf],
    "kitti": [0, 25, 50, 100, 200, 400, 800, 1600, np.inf],
    "kitti_depthval": [0, 10, 25, 50, 100, 200, 400, np.inf],
}
POS_BINS = [0, 50, 100, 200, 400, 800, 1600, 3200, np.inf]
METRICS = ["world_shape_med", "self_shape_med", "ws_ratio", "world_err_per_m", "self_absrel",
           "dist_ratio", "dist_pred", "dist_gt",
           "sw_res_med", "pw_res_med",
           "rpe1_rot", "rpe1_trans", "rpe10_rot", "rpe10_trans", "rpe10_dir",
           "wrpe10_rot", "wrpe10_trans"]
BLUE, ORANGE, INK2 = "#2a78d6", "#eb6834", "#52514e"


def rot_deg(R):
    return np.degrees(np.arccos(np.clip((np.trace(R, axis1=-2, axis2=-1) - 1) / 2, -1, 1)))


def rpe(poses, gt, lag, min_t=0.05):
    """Relative-pose error between stream positions p-lag and p. Returns
    rotation (deg), translation (m), translation direction (deg)."""
    P = len(poses)
    out = np.full((P, 3), np.nan)
    if P <= lag:
        return out
    Ep = np.linalg.inv(poses[:-lag]) @ poses[lag:]
    Eg = np.linalg.inv(gt[:-lag]) @ gt[lag:]
    out[lag:, 0] = rot_deg(np.linalg.inv(Eg[:, :3, :3]) @ Ep[:, :3, :3])
    tp, tg = Ep[:, :3, 3], Eg[:, :3, 3]
    out[lag:, 1] = np.linalg.norm(tp - tg, axis=1)
    ng, npd = np.linalg.norm(tg, axis=1), np.linalg.norm(tp, axis=1)
    cos = (tp * tg).sum(1) / np.maximum(ng * npd, 1e-12)
    out[lag:, 2] = np.where(ng > min_t, np.degrees(np.arccos(np.clip(cos, -1, 1))), np.nan)
    return out


def load_scene(d):
    r = dict(np.load(f"{d}/frames.npz"))
    meta = json.load(open(f"{d}/meta.json"))
    N, P = meta["n_frames"], len(r["pos"])
    t = {"pos": r["pos"].astype(int), "fid": r["fid"].astype(int)}
    t["ret"] = t["pos"] >= N
    c = r["c2w_pred"]
    t["dist_pred"] = np.linalg.norm(c[:, :3, 3], axis=1)
    g = r["c2w_gt"]
    has_gt = bool(np.isfinite(g).all(axis=(1, 2)).any())
    if has_gt:
        g0 = g[np.isfinite(g).all(axis=(1, 2))][0]
        grel = np.linalg.inv(g0) @ g
        t["dist_gt"] = np.linalg.norm(grel[:, :3, 3], axis=1)
        for lag in (1, 10):
            e = rpe(c, grel, lag)
            t[f"rpe{lag}_rot"], t[f"rpe{lag}_trans"], t[f"rpe{lag}_dir"] = e.T
        # World head's implied pose: the Sim(3) taking X_self onto X_world.
        T = r["T_sw"].copy()
        s = np.cbrt(np.linalg.det(T[:, :3, :3]))
        T[:, :3, :3] /= s[:, None, None]
        e = rpe(T, grel, 10)
        t["wrpe10_rot"], t["wrpe10_trans"] = e[:, 0], e[:, 1]
    else:
        t["dist_gt"] = np.full(P, np.nan)
    for k in ("world_shape_med", "self_shape_med", "self_absrel", "sw_res_med", "pw_res_med",
              "conf_mean", "conf_self_mean", "xw_norm_med"):
        t[k] = r[k]
    t["ws_ratio"] = r["world_shape_med"] / r["self_shape_med"]
    # World-head error in metres per metre of distance of the points from the
    # origin. Constant across distance <=> precision is relative to the origin.
    t["world_err_per_m"] = r["world_shape_med"] * r["xs_depth_med"] / r["xw_norm_med"]
    t["dist_ratio"] = t["dist_pred"] / np.where(t["dist_gt"] > 1e-6, t["dist_gt"], np.nan)
    for k in METRICS:
        t.setdefault(k, np.full(P, np.nan))
    return t, meta


def partial_spearman(y, x, z, max_collinear=0.95):
    """Spearman correlation of y with x after removing z (rank residuals).
    Undefined (nan) when x and z are themselves nearly rank-collinear, as on a
    sequence that drives monotonically away from the origin."""
    m = np.isfinite(y) & np.isfinite(x) & np.isfinite(z)
    if m.sum() < 20:
        return np.nan
    ry, rx, rz = (rankdata(a[m]) for a in (y, x, z))
    if abs(np.corrcoef(rx, rz)[0, 1]) > max_collinear:
        return np.nan
    A = np.c_[np.ones(m.sum()), rz]
    res = lambda v: v - A @ np.linalg.lstsq(A, v, rcond=None)[0]  # noqa: E731
    a, b = res(ry), res(rx)
    return float((a * b).sum() / np.sqrt((a * a).sum() * (b * b).sum()))


def binned(tables, key, edges, metrics, mask_fn):
    rows = []
    x = np.concatenate([t[key][mask_fn(t)] for t in tables])
    for lo, hi in zip(edges[:-1], edges[1:]):
        sel = (x >= lo) & (x < hi)
        row = {"bin": f"[{lo:g}, {hi:g})", "n": int(sel.sum())}
        for mk in metrics:
            v = np.concatenate([t[mk][mask_fn(t)] for t in tables])[sel]
            v = v[np.isfinite(v)]
            row[mk] = float(np.median(v)) if len(v) else None
        rows.append(row)
    return rows


def palindrome_pairs(t, mk, n_frames):
    """Per image (or image pair, for RPE): (key fid, forward value, return
    value, forward index, return index). An RPE at stream position p covers
    frames (p-lag, p); keyed by the pair's smaller fid, which is fid-lag on the
    way out and fid on the way back. Return-trip pairs that straddle the
    turnaround are not lag-apart in fid and are dropped."""
    lag = int(mk.split("_")[0].lstrip("wrpe")) if mk.startswith(("rpe", "wrpe")) else 0
    f, r = ~t["ret"], t["ret"]
    fwd = dict(zip(t["fid"][f] - lag, zip(t[mk][f], t["pos"][f])))
    out = []
    for fid, v, p in zip(t["fid"][r], t[mk][r], t["pos"][r]):
        if p - lag < n_frames - 1 or fid not in fwd:
            continue
        fv, fp = fwd[fid]
        if np.isfinite(v) and np.isfinite(fv) and v > 0 and fv > 0:
            out.append((fid, fv, v, fp, p))
    return np.array(out).reshape(-1, 5)


def plot_scene(t, meta, path):
    has_shape = np.isfinite(t["world_shape_med"]).any()
    has_pose = np.isfinite(t["rpe10_rot"]).any()
    panels = ["dist"] + (["shape", "ratio"] if has_shape else []) + (["rpe"] if has_pose else []) \
        + ([] if has_shape else ["swres"])
    fig, axes = plt.subplots(len(panels), 1, figsize=(9, 2.1 * len(panels)), sharex=True)
    x = t["pos"]
    turn = meta["n_frames"] - 1 if meta["order"] == "palindrome" else None
    for ax, p in zip(np.atleast_1d(axes), panels):
        if p == "dist":
            ax.plot(x, t["dist_pred"], color=BLUE, lw=1.5, label="predicted")
            if np.isfinite(t["dist_gt"]).any():
                ax.plot(x, t["dist_gt"], color=ORANGE, lw=1.5, label="GT")
            ax.set_ylabel("dist. from\norigin (m)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        elif p == "shape":
            ax.semilogy(x, t["world_shape_med"], color=BLUE, lw=1, label="X_world")
            ax.semilogy(x, t["self_shape_med"], color=ORANGE, lw=1, label="X_self")
            ax.set_ylabel("shape err\n(rel. depth)")
            ax.legend(frameon=False, fontsize=8, loc="upper left")
        elif p == "ratio":
            ax.plot(x, t["ws_ratio"], color=BLUE, lw=1)
            ax.axhline(1, color=INK2, lw=0.8, ls=":")
            ax.set_ylabel("world / self")
        elif p == "rpe":
            ax.semilogy(x, t["rpe10_rot"], color=BLUE, lw=1)
            ax.set_ylabel("RPE@10 rot\n(deg)")
        elif p == "swres":
            ax.semilogy(x, t["sw_res_med"], color=BLUE, lw=1)
            ax.set_ylabel("self→world\nresidual (rel.)")
        if turn is not None:
            ax.axvline(turn, color=INK2, lw=0.8, ls="--")
        ax.grid(alpha=0.25, lw=0.5)
        for s in ("top", "right"):
            ax.spines[s].set_visible(False)
    np.atleast_1d(axes)[-1].set_xlabel("stream position (dashed: turnaround)")
    fig.suptitle(f"CUT3R {meta['benchmark']}/{meta['scene']}", fontsize=10)
    fig.tight_layout()
    fig.savefig(path, dpi=110)
    plt.close(fig)


HEAD_BINS = np.array([0.05, 0.1, 0.2, 0.5, 1, 2, 5, 10, 20, 50, 100, 200, 500, 1000, 2000, 5000])
BENCH_COLOR = {"nrgbd": BLUE, "kitti_depthval": ORANGE, "kitti": "#1baf7a"}
BENCH_LABEL = {"nrgbd": "NRGBD", "kitti_depthval": "KITTI depth-val", "kitti": "KITTI odometry"}


def plot_headline(per_bench, path, min_n=20):
    """Binned medians against predicted distance from the origin, all benchmarks;
    solid = forward pass, dashed = return trip."""
    panels = [("ws_ratio", "X_world / X_self\nshape error"), ("self_shape_med", "X_self shape error\n(rel. depth)"),
              ("sw_res_med", "X_self→X_world\nresidual (rel.)"), ("rpe10_rot", "RPE@10 rotation\n(deg)")]
    fig, axes = plt.subplots(1, len(panels), figsize=(4.2 * len(panels), 3.4))
    for ax, (mk, lab) in zip(axes, panels):
        for bench, tables in per_bench.items():
            for passname, ls in (("fwd", "-"), ("ret", "--")):
                d = np.concatenate([t["dist_pred"][t["ret"] == (passname == "ret")] for t in tables])
                y = np.concatenate([t[mk][t["ret"] == (passname == "ret")] for t in tables])
                xs, ys = [], []
                for lo, hi in zip(HEAD_BINS[:-1], HEAD_BINS[1:]):
                    v = y[(d >= lo) & (d < hi) & np.isfinite(y)]
                    if len(v) >= min_n:
                        xs.append(np.sqrt(lo * hi))
                        ys.append(np.median(v))
                if xs:
                    ax.plot(xs, ys, ls, color=BENCH_COLOR[bench], lw=2, marker="o", ms=4,
                            label=f"{BENCH_LABEL[bench]} {'return' if passname == 'ret' else 'forward'}")
        ax.set_xscale("log")
        if mk != "ws_ratio":
            ax.set_yscale("log")
        else:
            ax.axhline(1, color=INK2, lw=0.8, ls=":")
        ax.set_xlabel("predicted distance from frame 0 (m)")
        ax.set_title(lab, fontsize=9)
        ax.grid(alpha=0.25, lw=0.5)
        for sp in ("top", "right"):
            ax.spines[sp].set_visible(False)
    seen = {}
    for ax in axes:
        for h, l in zip(*ax.get_legend_handles_labels()):
            seen.setdefault(l, h)
    fig.legend(seen.values(), seen.keys(), frameon=False, fontsize=8, ncol=len(seen), loc="lower center")
    fig.suptitle("CUT3R (published) — errors vs distance from the frame-0 origin; medians per log bin", fontsize=10)
    fig.tight_layout(rect=(0, 0.08, 1, 1))
    fig.savefig(path, dpi=120)
    plt.close(fig)


def fmt(v, nd=4):
    return "—" if v is None or (isinstance(v, float) and not np.isfinite(v)) else f"{v:.{nd}g}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--arm-dir", required=True)
    ap.add_argument("--out", required=True)
    args = ap.parse_args()
    os.makedirs(f"{args.out}/plots", exist_ok=True)

    summary, md, per_bench = {}, ["# Frame-anchor diagnostic — CUT3R (published checkpoint)\n"], {}
    for bdir in sorted(glob.glob(f"{args.arm_dir}/*/")):
        bench = osp.basename(bdir.rstrip("/"))
        scenes = sorted(d for d in glob.glob(f"{bdir}/*/") if osp.exists(f"{d}/frames.npz"))
        if not scenes or bench not in DIST_BINS:
            continue
        tables, metas = zip(*(load_scene(d.rstrip("/")) for d in scenes))
        per_bench[bench] = tables
        for t, m in zip(tables, metas):
            plot_scene(t, m, f"{args.out}/plots/{bench}_{m['scene']}.png")
        mets = [mk for mk in METRICS if any(np.isfinite(t[mk]).any() for t in tables)]
        fwd = lambda t: ~t["ret"]  # noqa: E731
        S = {"scenes": [m["scene"] for m in metas],
             "n_frames": [m["n_frames"] for m in metas],
             "fwd_by_dist_pred": binned(tables, "dist_pred", DIST_BINS[bench], mets, fwd),
             "fwd_by_pos": binned(tables, "pos", POS_BINS, mets, fwd),
             "ret_by_dist_pred": binned(tables, "dist_pred", DIST_BINS[bench], mets, lambda t: t["ret"])}
        if any(np.isfinite(t["dist_gt"]).any() for t in tables):
            S["fwd_by_dist_gt"] = binned(tables, "dist_gt", DIST_BINS[bench], mets, fwd)
        # Within-scene partial rank correlations. Forward pass: distance and index
        # mostly rise together. Full palindrome: the return trip decouples them.
        corr = {}
        for mk in mets:
            for scope, msk in (("fwd", fwd), ("all", lambda t: np.ones(len(t["pos"]), bool))):
                d_given_p = [partial_spearman(t[mk][msk(t)], t["dist_pred"][msk(t)], t["pos"][msk(t)].astype(float)) for t in tables]
                p_given_d = [partial_spearman(t[mk][msk(t)], t["pos"][msk(t)].astype(float), t["dist_pred"][msk(t)]) for t in tables]
                corr[f"{mk}|{scope}"] = dict(
                    dist_given_pos=d_given_p, pos_given_dist=p_given_d,
                    med_dist_given_pos=float(np.nanmedian(d_given_p)) if np.isfinite(d_given_p).any() else None,
                    med_pos_given_dist=float(np.nanmedian(p_given_d)) if np.isfinite(p_given_d).any() else None)
        S["partial_spearman"] = corr
        pal = {}
        if all(m["order"] == "palindrome" for m in metas):
            for mk in mets:
                pr = [palindrome_pairs(t, mk, m["n_frames"]) for t, m in zip(tables, metas)]
                allp = np.concatenate(pr)
                if not len(allp):
                    continue
                lr = np.log(allp[:, 2] / allp[:, 1])
                # first tenth of each sequence: forward = earliest & nearest the
                # origin, return = latest positions at the same images.
                early = np.concatenate([p[p[:, 0] < 0.1 * m["n_frames"]] for p, m in zip(pr, metas)])
                le = np.log(early[:, 2] / early[:, 1]) if len(early) else np.array([])
                dp = [palindrome_pairs(t, "dist_pred", m["n_frames"]) for t, m in zip(tables, metas)]
                dfwd = {(si, int(r[0])): r[1:3] for si, p in enumerate(dp) for r in p}
                ke = [(si, int(r[0])) for si, (p, m) in enumerate(zip(pr, metas)) for r in p
                      if r[0] < 0.1 * m["n_frames"]]
                de = np.array([dfwd[k] for k in ke if k in dfwd]).reshape(-1, 2)
                pal[mk] = dict(n=int(len(lr)), median_ret_over_fwd=float(np.exp(np.median(lr))),
                               early_dist_pred_fwd=float(np.median(de[:, 0])) if len(de) else None,
                               early_dist_pred_ret=float(np.median(de[:, 1])) if len(de) else None,
                               frac_ret_worse=float((lr > 0).mean()),
                               early_n=int(len(le)),
                               early_median_ret_over_fwd=float(np.exp(np.median(le))) if len(le) else None)
        S["palindrome"] = pal
        summary[bench] = S

        md.append(f"\n## {bench} — {len(scenes)} scenes, {sum(S['n_frames'])} frames\n")
        show = [mk for mk in ("world_shape_med", "self_shape_med", "ws_ratio", "world_err_per_m",
                              "self_absrel", "sw_res_med", "rpe10_rot", "rpe10_trans", "wrpe10_rot",
                              "dist_ratio") if mk in mets]
        for title, key in (("Forward pass, binned by predicted distance from origin (m)", "fwd_by_dist_pred"),
                           ("Forward pass, binned by GT distance from origin (m)", "fwd_by_dist_gt"),
                           ("Return trip, binned by predicted distance from origin (m)", "ret_by_dist_pred"),
                           ("Forward pass, binned by stream position", "fwd_by_pos")):
            if key not in S:
                continue
            md.append(f"\n**{title}** — medians\n\n| bin | n | " + " | ".join(show) + " |\n|" + "---|" * (len(show) + 2))
            for row in S[key]:
                if row["n"]:
                    md.append(f"| {row['bin']} | {row['n']} | " + " | ".join(fmt(row[mk]) for mk in show) + " |")
        md.append("\n**Within-scene partial Spearman** (median over scenes): ρ(metric, dist | pos) / ρ(metric, pos | dist)\n\n"
                  "| metric | forward pass | full palindrome |\n|---|---|---|")
        for mk in show:
            a, b = corr[f"{mk}|fwd"], corr[f"{mk}|all"]
            md.append(f"| {mk} | {fmt(a['med_dist_given_pos'], 2)} / {fmt(a['med_pos_given_dist'], 2)} | "
                      f"{fmt(b['med_dist_given_pos'], 2)} / {fmt(b['med_pos_given_dist'], 2)} |")
        if pal:
            md.append("\n**Palindrome, same image: return / forward** (median of per-image ratios)\n\n"
                      "| metric | pairs | median ratio | frac. return worse | first-10% pairs | first-10% median ratio | first-10% pred. dist fwd → ret (m) |\n|---|---|---|---|---|---|---|")
            for mk in show:
                if mk in pal:
                    p = pal[mk]
                    md.append(f"| {mk} | {p['n']} | {fmt(p['median_ret_over_fwd'], 3)} | {fmt(p['frac_ret_worse'], 2)} | "
                              f"{p['early_n']} | {fmt(p['early_median_ret_over_fwd'], 3)} | "
                              f"{fmt(p['early_dist_pred_fwd'], 3)} → {fmt(p['early_dist_pred_ret'], 3)} |")

    plot_headline(per_bench, f"{args.out}/headline.png")
    json.dump(summary, open(f"{args.out}/summary.json", "w"), indent=1)
    open(f"{args.out}/summary.md", "w").write("\n".join(md) + "\n")
    print("\n".join(md))


if __name__ == "__main__":
    main()
