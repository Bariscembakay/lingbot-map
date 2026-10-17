#!/usr/bin/env python3
"""Check a dataset's GT camera convention against lingbot before caching it.

lingbot runs (streaming, 8 scale frames, every frame a keyframe, SDPA) on the first --frames
frames of a few sequences, resized as the benchmark does (area budget 255000, align 14). Each
candidate convention turns the raw poses into OpenCV camera-to-world; per candidate:
  rel-rot     mean angle between consecutive relative rotations, lingbot vs GT (deg/frame):
              invariant to the world frame, so it isolates the camera-axis convention
  pitch/yaw   median camera-frame direction of motion, GT minus lingbot (deg): a constant
              offset here is an off-centre principal point (the KITTI-360 failure)
  ATE / AUC@3 Sim3-aligned, metres / %
plus lingbot's predicted horizontal FoV against the true one.
Candidates: tartan*  raw body axes (NED: x fwd, y right, z down) vs NED->OpenCV
            matrixcity  raw Blender/OpenGL vs OpenGL->OpenCV (flip y, z)
Usage: verify_gt_convention.py --dataset tartanair-v2 [--seqs 3] [--frames 400]
"""
from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path
from types import SimpleNamespace

import cv2
import numpy as np
import torch

REPO = Path(__file__).resolve().parents[3]
sys.path.insert(0, str(REPO))
sys.path.insert(0, str(Path(__file__).resolve().parent))
import train_pose_refine as tp  # noqa: E402
from check_training_data import quat_to_R  # noqa: E402
from scripts.memory.build_cache import benchmark_resize, build_model  # noqa: E402

NED_TO_CV = np.array([[0, 0, 1], [1, 0, 0], [0, 1, 0]], np.float64)   # columns: OpenCV x, y, z in body axes
GL_TO_CV = np.diag([1.0, -1.0, -1.0])
TARTAN = {  # root, trajectory glob, image dir / pattern, pose file, (W, H, f)
    "tartanair-v1": ("/data/tartanair-v1", "*/Easy/P*", "image_left/{:06d}_left.png", "pose_left.txt", (640, 480, 320.0)),
    "tartanair-v2": ("/data/tartanair-v2", "*/Data_easy/P*", "image_lcam_front/{:06d}_lcam_front.png", "pose_lcam_front.txt", (640, 640, 320.0)),
    "tartanground": ("/data/tartanground", "*/Data_diff/P*", "image_lcam_front/{:06d}_lcam_front.png", "pose_lcam_front.txt", (640, 640, 320.0)),
}
MC_ROUTES = Path("/group/compact-3dmem/campaigns/spatial_memory/_data_prep/matrixcity_street_routes_1048f_seed0.json")


def sequences(ds, n, frames):
    """-> [(name, image paths, raw c2w [F,4,4], {candidate: axis matrix}, (W, H, f))]"""
    if ds in TARTAN:
        root, pat, img, pose, cam = TARTAN[ds]
        out, envs = [], set()
        for t in sorted(Path(root).glob(pat)):
            env = t.relative_to(root).parts[0]
            if env in envs:                      # one trajectory per environment
                continue
            a = np.loadtxt(t / pose)
            if len(a) < frames:
                continue
            envs.add(env)
            T = np.tile(np.eye(4), (frames, 1, 1))
            T[:, :3, :3], T[:, :3, 3] = quat_to_R(a[:frames, 3:7]), a[:frames, :3]
            out.append((str(t).split("/data/")[1], [t / img.format(i) for i in range(frames)], T,
                        {"raw_body": np.eye(3), "ned_to_opencv": NED_TO_CV}, cam))
            if len(out) == n:
                break
        return out
    if ds == "matrixcity":
        R = json.loads(MC_ROUTES.read_text())["routes"][:n]
        root = Path("/data/matrix_city/small_city/street/train")
        return [(f"route{k}", [root / c / f"{i:04d}.png" for c, i in r["frames"][:frames]],
                 np.asarray(r["c2w"][:frames], np.float64), {"raw_opengl": np.eye(3), "opengl_to_opencv": GL_TO_CV},
                 (1000, 1000, 500.0)) for k, r in enumerate(R)]
    raise ValueError(ds)


def motion_dir(T, step=5):
    p, y = [], []
    for i in range(len(T) - step):
        v = T[i, :3, :3].T @ (T[i + step, :3, 3] - T[i, :3, 3])
        if np.linalg.norm(v) > 1e-6:
            p.append(np.degrees(np.arctan2(v[1], v[2])))
            y.append(np.degrees(np.arctan2(v[0], v[2])))
    return (float(np.median(p)), float(np.median(y))) if p else (float("nan"), float("nan"))


def rel_rot(A, B):
    ra = np.einsum("nji,njk->nik", A[:-1, :3, :3], A[1:, :3, :3])
    rb = np.einsum("nji,njk->nik", B[:-1, :3, :3], B[1:, :3, :3])
    m = np.einsum("nji,njk->nik", ra, rb)
    return float(np.degrees(np.arccos(np.clip((np.trace(m, axis1=1, axis2=2) - 1) / 2, -1, 1))).mean())


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--seqs", type=int, default=3)
    ap.add_argument("--frames", type=int, default=400)
    ap.add_argument("--out", type=Path)
    args = ap.parse_args()
    margs = SimpleNamespace(image_size=518, patch_size=14, max_frame_num=1024, kv_cache_sliding_window=64, num_scale_frames=8,
                            use_sdpa=True, camera_num_iterations=4, ckpt=str(REPO / "ckpt/lingbot-map.pt"))
    model = build_model(margs, torch.device("cuda"))
    res = {}
    for name, paths, Traw, cands, (W, H, f) in sequences(args.dataset, args.seqs, args.frames):
        w, h = benchmark_resize(W, H, 518, 14, 255000)
        imgs = np.stack([cv2.resize(cv2.cvtColor(cv2.imread(str(p)), cv2.COLOR_BGR2RGB), (w, h), interpolation=cv2.INTER_AREA) for p in paths])
        x = torch.from_numpy(np.ascontiguousarray(imgs.transpose(0, 3, 1, 2))).float().div_(255)
        with torch.no_grad(), torch.amp.autocast("cuda", dtype=torch.bfloat16):
            pred = model.inference_streaming(x, num_scale_frames=8, keyframe_interval=1, output_device=torch.device("cpu"))
        pe = pred["pose_enc"][0].float()
        L = tp.pose_enc_to_c2w(pe, (h, w)).double().numpy()
        L = np.linalg.inv(L[0])[None] @ L
        fov_l = float(np.degrees(pe[:, 8].median()))
        fov_gt = float(np.degrees(2 * np.arctan(W / 2 / f)))
        r = {"frames": len(paths), "hw": [h, w], "fov_x_lingbot": fov_l, "fov_x_gt": fov_gt}
        for cname, M in cands.items():
            G = Traw.copy()
            G[:, :3, :3] = G[:, :3, :3] @ M
            G = np.linalg.inv(G[0])[None] @ G
            pg, yg = motion_dir(G)
            pl, yl = motion_dir(L)
            m = tp.traj_metrics(torch.from_numpy(L), torch.from_numpy(G), 1.0)
            r[cname] = {"rel_rot_deg": rel_rot(L, G), "pitch_gt_minus_lb": pg - pl, "yaw_gt_minus_lb": yg - yl,
                        "ate_m": float(m["ate"]), "auc3": float(m["auc3"]),
                        "path_m": float(np.linalg.norm(np.diff(G[:, :3, 3], axis=0), axis=1).sum())}
            print(f"[{args.dataset}] {name} {cname:16s}: rel-rot {r[cname]['rel_rot_deg']:.3f} deg/frame, motion pitch/yaw diff "
                  f"{r[cname]['pitch_gt_minus_lb']:+.2f}/{r[cname]['yaw_gt_minus_lb']:+.2f} deg, ATE {m['ate']:.2f} m "
                  f"over {r[cname]['path_m']:.0f} m, AUC@3 {m['auc3']:.1f}", flush=True)
        print(f"[{args.dataset}] {name} FoV_x lingbot {fov_l:.1f} vs GT {fov_gt:.1f} deg", flush=True)
        res[name] = r
    if args.out:
        args.out.parent.mkdir(parents=True, exist_ok=True)
        args.out.write_text(json.dumps(res, indent=1))


if __name__ == "__main__":
    main()
