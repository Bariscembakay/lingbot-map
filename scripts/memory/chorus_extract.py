"""Chorus (chorus_3dgs) features for ScanNet++ scenes, averaged per voxel.

Per-Gaussian output would be several GB/scene, so features are averaged into
`--voxel` cubes on the GPU before anything is saved. Coordinates stay in the
ScanNet++ world frame (metres), the frame our clips' GT poses use.

Writes <out>/<scene>/{voxel_ijk.npy int32 [V,3], voxel_xyz.npy float32 [V,3]
(mean of member Gaussian centres), count.npy int32 [V], meta.json} plus one
fp16 [V,D] file per requested output:
  lang.npy   [V,1152]  SigLIP2 projector output (voxel mean of unit vectors)
  dino.npy   [V,1024]  DINOv3 projector output (same)
  upcast.npy [V,1232]  backbone_upcast, the shared embedding both projectors read;
                       lang/dino follow from it via the frozen projectors.
Runs with different --outputs share one voxel grid: the outlier filter and
grid are deterministic, and a later run asserts it reproduces voxel_ijk.
"""
import argparse, hashlib, json, subprocess, time
from pathlib import Path
import numpy as np
import torch
from plyfile import PlyData
import chorus

FILE = {"lang": "lang", "dino": "dino", "backbone_upcast": "upcast"}
ap = argparse.ArgumentParser()
ap.add_argument("--splats", default="/data/ScanNetppv2_gsplat/splats")
ap.add_argument("--checkpoint",
                default="/group/compact-3dmem/checkpoints/Chorus/lang-dino-enc-pretrain-scan-ppv2-mp3d-mcmc.pth")
ap.add_argument("--out", required=True)
ap.add_argument("--voxel", type=float, default=0.05)
ap.add_argument("--outputs", nargs="+", default=["lang", "dino"], choices=list(FILE))
ap.add_argument("--scenes", nargs="+", required=True)
args = ap.parse_args()

enc = chorus.load("chorus_3dgs", checkpoint=args.checkpoint, outputs=tuple(args.outputs),
                  return_numpy=False)
ck_sha = hashlib.sha256(open(args.checkpoint, "rb").read(1 << 24)).hexdigest()[:16]
git_sha = subprocess.run(["git", "-C", str(Path(chorus.__file__).parents[1]), "rev-parse", "--short", "HEAD"],
                         capture_output=True, text=True).stdout.strip()

for scene in args.scenes:
    od = Path(args.out) / scene
    if all((od / f"{FILE[k]}.npy").exists() for k in args.outputs):
        print(f"[skip] {scene}", flush=True); continue
    t0 = time.time()
    ply = Path(args.splats) / f"{scene}.ply"
    v = PlyData.read(ply)["vertex"]
    xyz_all = np.stack([v["x"], v["y"], v["z"]], 1).astype(np.float32)
    try:
        out = enc.encode(str(ply), save=False)
    except torch.OutOfMemoryError:
        # Dense scenes exceed an 80 GB card at Chorus's default chunk_size; they
        # are rerun on a bigger GPU rather than with smaller chunks, which would
        # change the features (each chunk only sees its own points).
        print(f"[oom] {scene} ({len(xyz_all)} Gaussians)", flush=True)
        torch.cuda.empty_cache(); torch.cuda.reset_peak_memory_stats()
        continue
    keep = np.asarray(out.metadata["source_keep_index"])
    xyz = torch.from_numpy(xyz_all[keep]).cuda()
    feats = {k: torch.as_tensor(out.features[k]) for k in args.outputs}  # CPU
    assert all(len(f) == len(xyz) for f in feats.values()), "features not aligned to kept rows"

    ijk = torch.floor(xyz / args.voxel).to(torch.int64)
    uniq, inv = torch.unique(ijk, dim=0, return_inverse=True)
    V = len(uniq)
    cnt = torch.zeros(V, device="cuda").index_add_(0, inv, torch.ones(len(inv), device="cuda"))
    def mean(x, step=1 << 20):
        # Chunked: a 10M-Gaussian scene's features (~90 GB fp32) never sit on the GPU at once.
        acc = torch.zeros(V, x.shape[1], device="cuda")
        for i in range(0, len(x), step):
            acc.index_add_(0, inv[i:i + step], x[i:i + step].to("cuda", torch.float32))
        return acc / cnt[:, None]
    od.mkdir(parents=True, exist_ok=True)
    ijk_np = uniq.cpu().numpy().astype(np.int32)
    if (od / "voxel_ijk.npy").exists():
        assert np.array_equal(np.load(od / "voxel_ijk.npy"), ijk_np), f"{scene}: voxel grid differs from earlier run"
    else:
        np.save(od / "voxel_ijk.npy", ijk_np)
        np.save(od / "voxel_xyz.npy", mean(xyz).cpu().numpy().astype(np.float32))
        np.save(od / "count.npy", cnt.cpu().numpy().astype(np.int32))
    for k, f in feats.items():
        np.save(od / f"{FILE[k]}.npy", mean(f).cpu().numpy().astype(np.float16))
    run = dict(outputs=args.outputs, n_gaussians=int(len(xyz_all)), n_kept=int(len(keep)), n_voxels=int(V),
               checkpoint=Path(args.checkpoint).name, checkpoint_sha256_first16MB=ck_sha, chorus_git=git_sha,
               outlier_filter=bool(out.metadata.get("outlier_filter_enabled")), sec=round(time.time() - t0, 1),
               peak_gpu_gb=round(torch.cuda.max_memory_allocated() / 1e9, 1))
    mp = od / "meta.json"
    meta = json.loads(mp.read_text()) if mp.exists() else dict(scene=scene, voxel=args.voxel,
                                                               frame="scannetpp_world_m")
    meta.setdefault("runs", []).append(run)
    mp.write_text(json.dumps(meta, indent=1))
    print(json.dumps(dict(scene=scene, **run)), flush=True)
    torch.cuda.reset_peak_memory_stats()
