#!/usr/bin/env python3
"""Refiners on lingbot's benchmark sequences, scored by the benchmark's own evaluators.

cache: lingbot runs each test scene exactly as benchmark/run.py does (LingbotMapMethod, config
       methods/lingbot_map.yaml (FlashInfer) or _sdpa: streaming, auto keyframe interval ceil(N/320) beyond 320 frames,
       the benchmark's own image loading and resizing), and every frame's special tokens (4 taps) and
       pose are saved. Non-keyframes still pass through the aggregator, so every frame has specials.
eval:  each refiner checkpoint re-predicts all poses of a scene in one pass (every frame, no
       subsampling); lingbot's cached poses are scored the same way. Trajectories are written as BSS
       artifacts and scored by benchmark TrajectoryEvaluator (evo, Sim3 ATE, RPE delta 1) and, for the
       short benchmarks lingbot reports AUC on, AUCEvaluator.
Usage: eval_refiner_bench.py cache --benches kitti,oxford_dense [--scenes 00,01]
       eval_refiner_bench.py eval --benches kitti --runs RUN_DIR[,RUN_DIR...]
"""
from __future__ import annotations

import argparse
import datetime
import json
import os
import sys
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import yaml

HERE = Path(__file__).resolve().parent
REPO = HERE.parents[2]
sys.path.insert(0, str(HERE))
sys.path.insert(0, str(REPO / "benchmark"))
import train_pose_refine as tp  # noqa: E402

# roots overridable so the same script runs on Leonardo against a mirror of the INSAIT trees
LB = Path(os.environ.get("BENCH_ROOT", "/group/compact-3dmem/campaigns/lingbot_map"))
BENCH = {"kitti": LB / "default/kitti", "oxford_dense": LB / "dense_s1/oxford_spires", "vbr": LB / "default/vbr",
         "eth3d": LB / "default/eth3d", "seven_scenes": LB / "default/seven_scenes", "tat": LB / "default/tat",
         "nrgbd": LB / "default/neural_rgbd", "tum_fr1": LB / "tum_fr1/tum"}
AUC_BENCH = {"eth3d", "seven_scenes", "tat"}      # lingbot Table 4
CACHE = Path(os.environ.get("BENCH_CACHE", "/group/compact-3dmem/datasets/_staging/lingbot-benchcache"))
OUT = Path(os.environ.get("BENCH_OUT", "/group/compact-3dmem/campaigns/spatial_memory/refiner_bench"))


def tag(args):
    """cache / output suffix of a non-default protocol, e.g. kitti@f1024_kf1"""
    return (f"@f{args.max_frames}" if args.max_frames else "") + (f"_kf{args.kf}" if args.kf else "")


def scenes(bench, only=None):
    out = sorted(d for d in BENCH[bench].iterdir() if (d / "gt/traj.txt").exists() and not d.name.startswith(("_", ".")))
    return [d for d in out if only is None or d.name in only]


def cache(args):
    from benchmark.core.loader import BSSLoader
    from benchmark.core.storage import BSSArtifact
    from methods.lingbot_map import LingbotMapMethod
    # flashinfer = lingbot's shipped config (paged KV, ~20 fps); sdpa re-concatenates the KV cache every
    # frame and ran at <1 fps on long scenes
    cfg = yaml.safe_load((REPO / f"benchmark/configs/methods/{'lingbot_map' if args.backend == 'flashinfer' else 'lingbot_map_sdpa'}.yaml").read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["checkpoint"] = str(REPO / "ckpt/lingbot-map.pt")   # the yaml hardcodes an INSAIT home path
    if args.max_frame_num:
        # frame-position table: every-frame keyframing on long scenes needs more than the yaml's 1024 rows
        kw["max_frame_num"] = args.max_frame_num
    method = LingbotMapMethod(**kw)
    if args.kf:
        method.keyframe_interval = args.kf      # e.g. 1 = every frame cached, as the training clips were
    model, psi = method.model, method.model.aggregator.patch_start_idx
    for bench in args.benches.split(","):
        for sc in scenes(bench, args.scenes and set(args.scenes.split(","))):
            f = CACHE / (bench + tag(args)) / f"{sc.name}.npz"
            if f.exists():
                continue
            f.parent.mkdir(parents=True, exist_ok=True)
            gt = BSSLoader(BSSArtifact(sc / "gt"), resize_context=method.resize_context)
            rgb = gt.load_rgb_list()
            images = method._prepare_images(rgb[: args.max_frames] if args.max_frames else rgb)
            chunks = []
            h = model.aggregator.register_forward_hook(
                lambda _m, _i, o: chunks.append(torch.stack(o[0], 2)[0][:, :, :psi].half().cpu()))
            try:
                pred = method._run_inference(images)
            finally:
                h.remove()
            spec = torch.cat(chunks).numpy()
            pose = pred["pose_enc"][0].float().numpy()
            assert len(spec) == len(pose) == len(images), (spec.shape, pose.shape, images.shape)
            tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
            np.savez(tmp, backend=args.backend, specials=spec, pose_enc=pose, hw=np.int32(images.shape[-2:]), bench=bench, scene=sc.name,
                     gt_dir=str(sc / "gt"), keyframe_interval=np.int32(args.kf or (-(-len(images) // 320) if len(images) > 320 else 1)))
            tmp.rename(f)
            print(f"[cache] {bench}/{sc.name}: {len(spec)} frames {tuple(images.shape[-2:])}", flush=True)
            del images, chunks, spec


def add_desc(args):
    """MegaLoc per frame, on the images lingbot was fed (benchmark loading and resizing), added to each
    cached scene as "megaloc" [T, 8448] fp16 (the refiners' --desc input)."""
    from benchmark.core.loader import BSSLoader
    from benchmark.core.storage import BSSArtifact
    from methods.lingbot_map import LingbotMapMethod
    from build_clip_cache import load_megaloc
    cfg = yaml.safe_load((REPO / "benchmark/configs/methods/lingbot_map_sdpa.yaml").read_text())
    kw = {k[1:]: v for k, v in cfg.items() if k.startswith("_")}
    kw["checkpoint"] = str(REPO / "ckpt/lingbot-map.pt")
    method = LingbotMapMethod(**kw)          # only for its image preparation
    del method.model
    torch.cuda.empty_cache()
    dev = torch.device("cuda")
    mega = load_megaloc().to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    for bench in args.benches.split(","):
        for f in sorted((CACHE / (bench + tag(args))).glob("*.npz")):
            z = dict(np.load(f))
            if "megaloc" in z:
                continue
            gt = BSSLoader(BSSArtifact(Path(str(z["gt_dir"]))), resize_context=method.resize_context)
            rgb = gt.load_rgb_list()
            n = len(z["pose_enc"])
            x = method._prepare_images(rgb[:n])
            out = []
            with torch.no_grad(), torch.autocast("cuda", dtype=torch.float16):
                for a in range(0, n, 64):
                    out.append(mega((x[a:a + 64].to(dev) - mean) / std).float().cpu())
            z["megaloc"] = torch.cat(out).half().numpy()
            tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
            np.savez(tmp, **z)
            tmp.rename(f)
            print(f"[desc] {bench}{tag(args)}/{f.stem}: {n} frames", flush=True)


def load_refiner(run, dev):
    import train_bidir_refiner as br
    ck = torch.load(Path(run) / "best.pt", map_location="cpu", weights_only=False)
    a = SimpleNamespace(**ck["args"])
    head = tp.build_head(tp.REPO / "ckpt/lingbot-map.pt", rope_on=False)
    dd = br.DESC_DIM if getattr(a, "desc", False) else 0
    if getattr(a, "arch", "tokens") == "posetok":
        m = br.PoseTokenRefiner(a.anchors == "fixed", width=a.width, tok_part=a.tok_part, pose_out=a.pose_out, desc_dim=dd)
    else:
        m = br.BidirRefiner(head, a.anchors == "fixed", width=a.width, head_mode=getattr(a, "head_mode", "causal"), desc_dim=dd)
    missing, unexpected = m.load_state_dict(ck["model"], strict=False)
    assert not unexpected and all(k.startswith("head.") for k in missing), (unexpected[:5], missing[:5])
    return m.to(dev).eval(), int(ck["step"])


def score(pred_c2w, gt_dir, out_dir, auc):
    """Write a BSS trajectory artifact and score it with the benchmark's evaluators."""
    from benchmark.core.loader import BSSLoader
    from benchmark.core.storage import BSSArtifact
    from benchmark.evaluation.auc import AUCEvaluator
    from benchmark.evaluation.trajectory import TrajectoryEvaluator
    from benchmark.io.trajectory import write_trajectory
    out_dir.mkdir(parents=True, exist_ok=True)
    write_trajectory(out_dir / "traj.txt", pred_c2w)
    gtm = json.loads((Path(gt_dir) / ".complete.json").read_text())["metadata"]
    meta = {"num_frames": len(pred_c2w), "image_width": gtm.get("image_width"), "image_height": gtm.get("image_height"),
            "frame_keys": ["pose"], "global_keys": []}
    (out_dir / ".complete.json").write_text(json.dumps({"completed_at": datetime.datetime.now().isoformat(), "metadata": meta}))
    g, p = BSSLoader(BSSArtifact(Path(gt_dir))), BSSLoader(BSSArtifact(out_dir))
    r = TrajectoryEvaluator().evaluate(g, p)
    r.pop("traj_transform")
    if auc:
        a = AUCEvaluator().evaluate(g, p)
        r.update({k: float(v) for k, v in a.items() if k.startswith("AUC_")})
    (out_dir / "eval.json").write_text(json.dumps(r, indent=1))
    return r


def plot_scenes(files):
    """One PNG per scene: GT and every arm's trajectory, Sim3-aligned to GT, seen from above GT's
    plane of motion (its two main axes). Written to OUT/_plots/<bench>/<scene>.png."""
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    from benchmark.io.trajectory import read_trajectory
    for b, fs in files.items():
        for f in fs:
            z = np.load(f)
            sc = str(z["scene"])
            n = len(z["pose_enc"])      # a scene whose last GT poses are missing reads shorter without n
            G = read_trajectory(Path(str(z["gt_dir"])) / "traj.txt", n)[:n, :3, 3]
            ok = np.isfinite(G).all(1)
            c = G[ok].mean(0)
            axes = np.linalg.svd(G[ok] - c, full_matrices=False)[2][:2]
            fig, ax = plt.subplots(figsize=(7, 7))
            ax.plot(*((G[ok] - c) @ axes.T).T, color="k", lw=1.6, label="GT")
            for arm in sorted(d.name for d in OUT.iterdir() if not d.name.startswith(("_", "."))):
                pf = OUT / arm / b / sc / "pred/traj.txt"
                if not pf.exists():
                    continue
                P = read_trajectory(pf, n)[:, :3, 3]
                m = ok & np.isfinite(P).all(1)
                s, R, t = tp.umeyama(P[m], G[m])
                Pa = s * P[m] @ R.T + t
                ate = json.loads((pf.parent / "eval.json").read_text())["ate"]
                ax.plot(*((Pa - c) @ axes.T).T, lw=1.0, label=f"{arm[:60]} (ATE {ate:.3g})")
            ax.set_aspect("equal")
            ax.set_title(f"{b} / {sc}  ({len(G)} frames)")
            ax.legend(fontsize=6, loc="best")
            o = OUT / "_plots" / b / f"{sc}.png"
            o.parent.mkdir(parents=True, exist_ok=True)
            fig.savefig(o, dpi=130, bbox_inches="tight")
            plt.close(fig)


@torch.no_grad()
def evaluate(args):
    dev = torch.device("cuda")
    files = {b + tag(args): sorted((CACHE / (b + tag(args))).glob("*.npz")) for b in args.benches.split(",")}
    # the published run covers whole sequences, so it is only comparable under the default protocol
    arms = ([("lingbot_official", "official")] if not tag(args) else []) + [("lingbot", None)] + [(Path(r).name, r) for r in args.runs.split(",") if r]
    table = {}
    for arm, run in arms:
        model, step = load_refiner(run, dev) if run and run != "official" else (None, None)
        for b, fs in files.items():
            rows = {}
            for f in fs:
                z = np.load(f)
                hw = tuple(int(x) for x in z["hw"])
                lp = torch.from_numpy(z["pose_enc"]).to(dev)
                if run == "official":    # the benchmark's own lingbot run of this scene, as published
                    from benchmark.io.trajectory import read_trajectory
                    of = Path(str(z["gt_dir"])).parent / "lingbot_map/traj.txt"
                    if not of.exists():
                        continue
                    c2w = read_trajectory(of, len(lp))
                elif model is None:
                    p = lp
                else:
                    sp = z["spec23"] if "spec23" in z.files else z["specials"][:, tp.TAP23]
                    spec = torch.from_numpy(np.ascontiguousarray(sp)).to(dev)
                    desc = (torch.from_numpy(z["megaloc"]).to(dev)[None]
                            if getattr(model, "desc_in", None) is not None else None)
                    with torch.autocast("cuda", dtype=torch.bfloat16):
                        p = model(spec[None], lp[None], desc=desc)[-1][0].float()
                if run != "official":
                    c2w = tp.pose_enc_to_c2w(p, hw).double().cpu().numpy()
                rows[str(z["scene"])] = score(c2w, str(z["gt_dir"]), OUT / arm / b / str(z["scene"]) / "pred",
                                              b.split("@")[0] in AUC_BENCH or args.auc_all)
                print(f"[eval] {arm} {b}/{z['scene']}: " + ", ".join(f"{k} {v:.4g}" for k, v in rows[str(z['scene'])].items()), flush=True)
            mean = {k: float(np.mean([r[k] for r in rows.values()])) for k in next(iter(rows.values()))} if rows else {}
            table.setdefault(arm, {"step": step})[b] = {"mean": mean, "scenes": rows}
            (OUT / arm / b).mkdir(parents=True, exist_ok=True)
            (OUT / arm / b / "summary.json").write_text(json.dumps(table[arm][b], indent=1))
        del model
        torch.cuda.empty_cache()
    plot_scenes(files)
    print("\n| arm | " + " | ".join(f"{b} ATE" for b in files) + " |")
    for arm in table:
        print(f"| {arm} | " + " | ".join(f"{table[arm][b]['mean'].get('ate', float('nan')):.3f}" for b in files) + " |")


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("task", choices=["cache", "eval", "both", "desc"])
    ap.add_argument("--benches", required=True)
    ap.add_argument("--scenes")
    ap.add_argument("--runs", default="")
    ap.add_argument("--backend", choices=["flashinfer", "sdpa"], default="flashinfer")
    ap.add_argument("--max-frames", type=int, default=0, help="only the first N frames of each scene (0 = all)")
    ap.add_argument("--kf", type=int, default=0, help="force lingbot's keyframe interval (0 = benchmark's auto)")
    ap.add_argument("--max-frame-num", type=int, default=0, help="lingbot's frame-position table size (0 = the yaml's)")
    ap.add_argument("--auc-all", action="store_true", help="AUC on every benchmark, not only lingbot's Table-4 ones")
    args = ap.parse_args()
    if args.task in ("cache", "both"):
        cache(args)
    if args.task == "desc":
        add_desc(args)
    if args.task in ("eval", "both"):
        evaluate(args)


if __name__ == "__main__":
    main()
