#!/usr/bin/env python3
"""Build the 1048-frame lingbot caches from a plan_clips.py plan (one npz per clip).

lingbot streams each clip as the NRGBD / ScanNet++ caches were built (GCTStream, SDPA attention,
sliding window 64 + 8 scale frames, every frame a keyframe, bf16 aggregator). Per clip:
  specials [T, 4, 6, 2048] fp16   camera / 4 registers / scale token at taps 4, 11, 17, 23
  pose_enc [T, 9]                 lingbot's causal camera-head output
  gt_c2w   [T, 4, 4]              OpenCV camera-to-world relative to the clip's first frame, divided
                                  by gt_scale (paper 3.2: mean anchor-cloud distance from the 8 scale
                                  frames' GT depth; 1 = metric for datasets without depth)
  K [T, 3, 3], hw                 at the size lingbot was fed
  lbdino_{cls,mean,gem} [T, 1024] lingbot's own DINOv2 (aggregator.patch_embed) per frame
  megaloc [T, 8448]               MegaLoc place descriptor per frame
  rel_rot_residual_deg            lingbot-vs-GT consecutive relative rotation (convention check)
Patch tokens are not stored: tap 23 alone is 4.1 MB per frame.
Usage: build_clip_cache.py --plan plan.json --out DIR [--shard i --nshards n] [--batch B]
"""
from __future__ import annotations

import argparse
import json
import os
import sys
import time
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import torch
import torch.nn.functional as F

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from clip_sources import source  # noqa: E402
from lingbot_map.memory.gt import canonical_scale  # noqa: E402
from scripts.memory.build_cache import build_model  # noqa: E402
from scripts.benchmark_gct_memory import compile_model  # noqa: E402

GEM_P = 3.0
A = 8


def load_megaloc():
    """From a stored copy (hub checkout + HF weights) under checkpoints/MegaLoc in each zone, offline:
    ~/.cache is node-local /scratch on sof1-side nodes and shared /home on msp3, and dozens of shards
    fetching github's zipball into one hub path collided."""
    root = Path(os.environ.get("MEGALOC_ROOT", "/group/compact-3dmem/checkpoints/MegaLoc"))
    if (root / "repo/hubconf.py").exists():
        os.environ["HF_HOME"] = str(root / "hf")
        os.environ["HF_HUB_OFFLINE"] = "1"
        return torch.hub.load(str(root / "repo"), "get_trained_model", source="local")
    return torch.hub.load("gmberton/MegaLoc", "get_trained_model")


def rel_rot_residual(P, G):
    ra = np.einsum("nji,njk->nik", P[:-1, :3, :3], P[1:, :3, :3])
    rb = np.einsum("nji,njk->nik", G[:-1, :3, :3], G[1:, :3, :3])
    m = np.einsum("nji,njk->nik", ra, rb)
    return float(np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean())


@torch.no_grad()
def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--plan", type=Path, required=True)
    ap.add_argument("--out", type=Path, required=True)
    ap.add_argument("--shard", type=int, default=0)
    ap.add_argument("--nshards", type=int, default=1)
    ap.add_argument("--no-megaloc", action="store_true")
    ap.add_argument("--batch", type=int, default=1, help="clips streamed together (B dimension of inference_streaming)")
    ap.add_argument("--skip-list", type=Path, help="file of clip npz names already built elsewhere (msp3 shards: sof1 staging)")
    ap.add_argument("--compile", action="store_true", help="CUDA-graph the per-frame blocks (lingbot's compile_model)")
    ap.add_argument("--flashinfer", action="store_true",
                    help="FlashInfer attention; its JIT build fails on msp3/gcp nodes (glibc math headers vs nvcc)")
    args = ap.parse_args()
    plan = json.loads(args.plan.read_text())
    ds = plan["summary"]["dataset"]
    clips = plan["clips"][args.shard::args.nshards]
    src = source(ds)
    dev = torch.device("cuda")
    # 3D RoPE frequencies are a fixed table indexed by frame: 2048 rows reproduce the first 1024 exactly and cover 1048-frame clips
    margs = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=2048, kv_cache_sliding_window=64, num_scale_frames=A,
                            use_sdpa=not args.flashinfer, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(margs, dev)
    # the DPT depth head is not cached and is most of the per-frame cost; specials and the
    # camera head do not depend on it, and GCTBase.forward skips a None head
    model.depth_head = None
    if args.compile:
        compile_model(model)
    agg = model.aggregator
    psi = agg.patch_start_idx
    mega = None if args.no_megaloc else load_megaloc().to(dev).eval()
    mean = torch.tensor([0.485, 0.456, 0.406], device=dev)[:, None, None]
    std = torch.tensor([0.229, 0.224, 0.225], device=dev)[:, None, None]
    args.out.mkdir(parents=True, exist_ok=True)
    print(f"[cache] {ds}: shard {args.shard}/{args.nshards}, {len(clips)} clips, attention {'flashinfer' if args.flashinfer else 'sdpa'}, compile {args.compile}", flush=True)

    def clip_name(c):
        return f"{c['seq'].replace('/', '__')}__f{c['frames'][0]:06d}_s{c['stride']}_n{len(c['frames'])}"

    done = set(args.skip_list.read_text().split()) if args.skip_list and args.skip_list.exists() else set()
    final = Path(f"/group/compact-3dmem/datasets/_staging/lingbot-cache1048-{ds}")
    if args.out.resolve() != final:   # msp3 shard on local scratch: clips already staged on sof1 are done
        import subprocess
        r = subprocess.run(["ssh", "sof1", f"ls {final}"], capture_output=True, text=True, timeout=120)
        done |= set(r.stdout.split())
        print(f"[cache] {len(done)} clips already staged on sof1", flush=True)
    todo = [c for c in clips if not (args.out / f"{clip_name(c)}.npz").exists()
            and not (args.out / f"{clip_name(c)}.failed").exists() and f"{clip_name(c)}.npz" not in done]
    # longest first so lengths inside a batch are close: a shorter clip is padded by repeating its last
    # frame, which cannot change its real frames (streaming is causal)
    todo.sort(key=lambda c: -len(c["frames"]))
    batches = [todo[k:k + args.batch] for k in range(0, len(todo), args.batch)]
    from concurrent.futures import ThreadPoolExecutor
    pool = ThreadPoolExecutor(1)      # decode the next batch's images while lingbot runs on this one

    def load(batch):
        return [src.frames(c["seq"], c["frames"]) for c in batch]

    def stream(xs):
        """xs: list of [T_i, 3, H, W] float CPU tensors of one size -> per clip (specials, pose, cls, mean, gem)"""
        B, S = len(xs), max(len(x) for x in xs)
        x = torch.stack([torch.cat([x, x[-1:].expand(S - len(x), -1, -1, -1)]) for x in xs])
        acc = {"spec": [], "cls": [], "mean": [], "gem": []}

        # captures stay on the GPU until the batch ends: a per-frame .cpu() synchronises every frame
        # and cost ~40% of the streaming throughput
        def on_agg(_m, _i, o):
            acc["spec"].append(torch.stack(o[0], 2)[:, :, :, :psi].half())

        def on_dino(_m, _i, o):
            p = o["x_norm_patchtokens"].float()
            acc["cls"].append(o["x_norm_clstoken"].half().reshape(B, -1, 1024))
            acc["mean"].append(p.mean(1).half().reshape(B, -1, 1024))
            acc["gem"].append(p.clamp(min=1e-6).pow(GEM_P).mean(1).pow(1 / GEM_P).half().reshape(B, -1, 1024))

        hs = [agg.register_forward_hook(on_agg), agg.patch_embed.register_forward_hook(on_dino)]
        try:
            with torch.amp.autocast("cuda", dtype=torch.bfloat16):
                pred = model.inference_streaming(x, num_scale_frames=A, keyframe_interval=1, output_device=None)
        finally:
            for h in hs:
                h.remove()
        out = {k: torch.cat(v, 1).cpu().numpy() for k, v in acc.items()}
        pose = pred["pose_enc"].float().cpu()
        assert out["spec"].shape[1] == pose.shape[1] == S, (out["spec"].shape, pose.shape, S)
        return x, [(out["spec"][b, :len(xs[b])], pose[b, :len(xs[b])], out["cls"][b, :len(xs[b])],
                 out["mean"][b, :len(xs[b])], out["gem"][b, :len(xs[b])]) for b in range(B)]

    def finish(c, imgs, K, x, cap):
        name = clip_name(c)
        f = args.out / f"{name}.npz"
        ids = c["frames"]
        spec, pose, cls, mn, gem = cap
        assert len(spec) == len(ids) == len(pose), (len(spec), len(ids), len(pose))
        out = {"specials": spec, "pose_enc": pose.numpy(), "lbdino_cls": cls, "lbdino_mean": mn, "lbdino_gem": gem}
        if mega is not None:
            m = []
            for a in range(0, len(x), 64):
                m.append(mega((x[a:a + 64].to(dev) - mean) / std).float().cpu())
            out["megaloc"] = torch.cat(m).half().numpy()
        T = src.poses(c["seq"])[ids]
        Trel = np.linalg.inv(T[0])[None] @ T
        scale = 1.0
        if not src.metric:
            d, Kd = src.depth(c["seq"], ids[:A])
            scale = canonical_scale(d.astype(np.float64), Kd.astype(np.float64), Trel[:A], A)
        G = Trel.copy()
        G[:, :3, 3] /= scale
        hw = imgs.shape[1:3]
        P = tp.pose_enc_to_c2w(pose, hw).double().numpy()
        res = rel_rot_residual(np.linalg.inv(P[0])[None] @ P, G)
        out.update(gt_c2w=G.astype(np.float32), K=K.astype(np.float32), hw=np.int32(hw), gt_scale=np.float64(scale),
                   dataset=ds, seq=c["seq"], frame_ids=np.int32(ids), stride=np.int32(c["stride"]),
                   rel_rot_residual_deg=np.float32(res), attn_backend="flashinfer" if args.flashinfer else "sdpa")
        tmp = f.with_suffix(f".tmp{os.getpid()}.npz")
        np.savez(tmp, **out)
        tmp.rename(f)
        return f"{name}: {len(ids)} frames {hw}, scale {scale:.3f}, rel-rot {res:.3f} deg/frame"

    def fail(c, e):  # one broken sequence (e.g. a ScanNet++ scene without depth.bin) must not end the shard
        (args.out / f"{clip_name(c)}.failed").write_text(f"{type(e).__name__}: {e}\n")
        print(f"[cache] {clip_name(c)}: FAILED {type(e).__name__}: {e}", flush=True)

    nxt = pool.submit(load, batches[0]) if batches else None
    done_n = 0
    for bi, batch in enumerate(batches):
        t0 = time.time()
        fut, nxt = nxt, (pool.submit(load, batches[bi + 1]) if bi + 1 < len(batches) else None)
        try:
            loaded = fut.result()
        except Exception:   # find the broken clip(s) one by one
            loaded = []
            for c in batch:
                try:
                    loaded.append(src.frames(c["seq"], c["frames"]))
                except Exception as e:
                    fail(c, e)
                    loaded.append(None)
        ok = [(c, im) for c, im in zip(batch, loaded) if im is not None]
        t_load = time.time() - t0
        xs = [torch.from_numpy(np.ascontiguousarray(im.transpose(0, 3, 1, 2))).float().div_(255) for _, (im, _) in ok]
        groups = {}
        for i, x in enumerate(xs):   # a batch whose clips differ in image size streams as several
            groups.setdefault(tuple(x.shape[-2:]), []).append(i)
        caps = [None] * len(ok)
        try:
            for idx in groups.values():
                xb, cb = stream([xs[i] for i in idx])
                for j, i in enumerate(idx):   # views into the padded batch, so the clip frames are not held twice
                    caps[i], xs[i] = cb[j], xb[j, :len(xs[i])]
        except Exception as e:
            for c, _ in ok:
                fail(c, e)
            torch.cuda.empty_cache()
            continue
        t_inf = time.time() - t0 - t_load
        nfr = sum(len(c["frames"]) for c, _ in ok)
        for (c, (im, K)), x, cap in zip(ok, xs, caps):
            try:
                msg = finish(c, im, K, x, cap)
                done_n += 1
                print(f"[cache] {done_n}/{len(todo)} {msg}", flush=True)
            except Exception as e:
                fail(c, e)
        print(f"[cache] batch {bi + 1}/{len(batches)}: {len(ok)} clips, {nfr} frames, {nfr / (time.time() - t0):.1f} clip-frames/s "
              f"(wait-for-images {t_load:.0f} s, lingbot {t_inf:.0f} s, rest {time.time() - t0 - t_load - t_inf:.0f} s)", flush=True)


if __name__ == "__main__":
    main()
