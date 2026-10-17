#!/usr/bin/env python3
"""EMERGENCY viz: CUT3R state recall on one NRGBD scene.

Ingest the scene at stride 2; at 300 and 500 ingested frames, query PAST
cameras (stride 8) from the state; fuse each snapshot's recalled pointmaps
and emit viewer PLYs against GT-depth unprojection. Conventions copied from
CUT3R's own NRGBD eval loader (eval/mv_recon/data.py): fx=fy=554.2562584220408
cx=320 cy=240 @640x480; depth png /1000 m, valid (1e-3,10); poses.txt OpenGL
c2w -> pose[:,1:3]*=-1."""
import sys, time
from pathlib import Path
import numpy as np
import torch, cv2

sys.path.insert(0, "/home/baris_bakay/lingbot-map/.agents/scratch/memory_eval")
sys.path.insert(0, "/home/baris_bakay/lingbot-map")
from run_cut3r_recall import (get_ray_map, preprocess_like_demo, CKPT)  # noqa
from dust3r.model import ARCroco3DStereo                    # noqa: E402
from dust3r.inference import inference, inference_step      # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera     # noqa: E402
from scripts.memory.render_state import emit                 # noqa: E402
from scipy.spatial import cKDTree                            # noqa: E402

SCENE = "breakfast_room"
ROOT = Path("/data/NRGBD") / SCENE
STRIDE, SNAPSHOTS, QSTRIDE = 2, (300, 500), 8
OUTDIR = Path("/group/compact-3dmem/campaigns/spatial_memory/"
              "CUT3R_zeroshot_512dpt") / f"nrgbd_{SCENE}_recall_stride2"

def load_poses(p):
    lines = [l for l in Path(p).read_text().splitlines() if l.strip()]
    n = len(lines) // 4
    P = np.array([[list(map(float, lines[4*i+j].split())) for j in range(4)]
                  for i in range(n)], np.float64)
    P[:, :, 1:3] *= -1.0            # gl -> cv, CUT3R's own line
    return P

def gt_points(idx, c2w_rel, K):
    d = cv2.imread(str(ROOT / "depth" / f"depth{idx}.png"), cv2.IMREAD_UNCHANGED)
    d = np.nan_to_num(d.astype(np.float32)) / 1000.0
    d[(d > 10) | (d < 1e-3)] = 0
    h, w = d.shape
    v, u = np.mgrid[0:h, 0:w]
    z = d
    x = (u - K[0, 2]) * z / K[0, 0]
    y = (v - K[1, 2]) * z / K[1, 1]
    pts = np.stack([x, y, z], -1)
    valid = z > 0
    R, t = c2w_rel[:3, :3], c2w_rel[:3, 3]
    return pts @ R.T + t, valid

def main():
    K = np.array([[554.2562584220408, 0, 320], [0, 554.2562584220408, 240],
                  [0, 0, 1]], np.float64)
    n_imgs = len(list((ROOT / "images").glob("img*.png")))
    poses = load_poses(ROOT / "poses.txt")
    max_ingest = SNAPSHOTS[-1]
    ids = [i * STRIDE for i in range(max_ingest)]
    assert ids[-1] < min(n_imgs, len(poses)), (ids[-1], n_imgs, len(poses))
    c2w = np.linalg.inv(poses[ids[0]])[None] @ poses[ids]   # relative to frame 0

    frames = np.stack([cv2.cvtColor(cv2.imread(str(ROOT / "images" / f"img{i}.png")),
                                    cv2.COLOR_BGR2RGB) for i in ids])
    imgs, K_final, H, W = preprocess_like_demo(
        frames, np.repeat(K[None], len(ids), 0))
    print(f"[pre] {len(imgs)} frames -> {H}x{W}")
    model = ARCroco3DStereo.from_pretrained(CKPT).to("cuda").eval()

    for snap in SNAPSHOTS:
        views = [{
            "img": imgs[i:i+1], "ray_map": torch.full((1, 6, H, W), torch.nan),
            "true_shape": torch.from_numpy(np.int32([[H, W]])),
            "idx": i, "instance": str(i),
            "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
            "img_mask": torch.tensor(True)[None], "ray_mask": torch.tensor(False)[None],
            "update": torch.tensor(True)[None], "reset": torch.tensor(False)[None],
        } for i in range(snap)]
        t0 = time.time()
        _, state_args = inference(views, model, "cuda")
        state = state_args[-1]
        print(f"[state@{snap}] ingest {time.time()-t0:.0f}s")
        qs = list(range(0, snap, QSTRIDE))
        preds, gts, errs = [], [], []
        for q in qs:
            rm = get_ray_map(c2w[0], c2w[q], K_final[q], H, W)
            view = {
                "img": torch.full((1, 3, H, W), torch.nan),
                "ray_map": torch.from_numpy(rm)[None].float(),
                "true_shape": torch.from_numpy(np.int32([[H, W]])),
                "idx": snap+q, "instance": str(snap+q),
                "camera_pose": torch.from_numpy(np.eye(4, dtype=np.float32))[None],
                "img_mask": torch.tensor(False)[None], "ray_mask": torch.tensor(True)[None],
                "update": torch.tensor(False)[None], "reset": torch.tensor(False)[None],
            }
            out = inference_step(view, state, model, "cuda")["pred"]
            pw = out["pts3d_in_other_view"][0].numpy().reshape(-1, 3)
            gp, gv = gt_points(ids[q], c2w[q], K)
            # per-view metric-1: world L21 on the model-input grid vs GT grid --
            # resolutions differ, so score on nearest-GT distance instead.
            preds.append(pw[::3]); gts.append(gp[gv][::3])
        P, G = np.concatenate(preds), np.concatenate(gts)
        tree = cKDTree(G[::2])
        d, _ = tree.query(P, workers=-1)
        errs = d
        print(f"[state@{snap}] fused {len(P)} pred pts vs {len(G)} GT pts | "
              f"nn-dist mean {d.mean():.4f} med {np.median(d):.4f} m")
        from scripts.memory.render_state import write_ply, heat
        tag = f"state{snap}_recall"
        (OUTDIR / tag).mkdir(parents=True, exist_ok=True)
        lo, hi = np.percentile(d, 5), np.percentile(d, 95)
        write_ply(OUTDIR / tag / "pred_cloud.ply", P, heat(d, float(lo), float(hi)))
        write_ply(OUTDIR / f"{tag}-gt.ply", G, np.full((len(G), 3), 160, np.uint8))
        print(f"[viz] {tag} -> {OUTDIR}")
    print("done")

if __name__ == "__main__":
    main()
