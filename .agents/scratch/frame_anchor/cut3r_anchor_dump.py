#!/usr/bin/env python
"""Does CUT3R degrade with distance from its frame-0 origin?

Streams one sequence through the published CUT3R checkpoint, one frame per
step through the model's own `_forward_decoder_step` (the body of
`_forward_impl`'s loop), and records per stream position:

  * the predicted camera-to-frame-0 pose,
  * shape error of the egocentric pointmap X_self and of the frame-0 pointmap
    X_world against GT (per-frame Sim(3) fit, so only non-rigid error remains),
  * GT-free agreement between X_self, X_world and the pose head,
  * the Sim(3) that maps X_self onto X_world (the world head's implied pose).

`--order palindrome` plays 0..N-1 and back to 0. CUT3R is causal, so the first
N positions are exactly the forward run; the return trip shows every image a
second time at the same distance from the origin but a later stream index.
"""
import argparse
import glob
import json
import os
import os.path as osp
import subprocess
import sys
import time

import cv2
import numpy as np
import torch

REPO = osp.abspath(osp.join(osp.dirname(__file__), "../../.."))
CUT3R = osp.join(REPO, "CUT3R")
sys.path[:0] = [CUT3R, osp.join(CUT3R, "src")]

from dust3r.model import ARCroco3DStereo  # noqa: E402
from dust3r.utils.camera import pose_encoding_to_camera  # noqa: E402
from dust3r.utils.image import load_images  # noqa: E402

CKPT = "/group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth"
NRGBD_K = np.array([[554.2562584220408, 0, 320], [0, 554.2562584220408, 240], [0, 0, 1]])
KITTI_SEL_INTR = "/group/compact-3dmem/datasets/kitti/depth_selection/val_selection_cropped/intrinsics"


# ---------------------------------------------------------------- datasets
def nrgbd_frames(root, scene):
    n = len(os.listdir(f"{root}/{scene}/images"))
    lines = open(f"{root}/{scene}/poses.txt").read().split("\n")
    frames = []
    for i in range(n):
        rows = lines[4 * i : 4 * i + 4]
        if "nan" in rows[0]:
            c2w = None
        else:
            c2w = np.array([[float(x) for x in r.split()] for r in rows])
            c2w[:, 1:3] *= -1.0  # OpenGL -> OpenCV, as CUT3R's own NRGBD loader
        frames.append(dict(fid=i, img=f"{root}/{scene}/images/img{i}.png",
                           depth=f"{root}/{scene}/depth/depth{i}.png", depth_kind="nrgbd",
                           K=NRGBD_K, c2w=c2w))
    return frames


def kitti_odo_frames(root, seq):
    imgs = sorted(glob.glob(f"{root}/sequences/{seq}/image_2/*.png"))
    calib = {l.split(":")[0]: np.array(l.split()[1:], float).reshape(3, 4)
             for l in open(f"{root}/sequences/{seq}/calib.txt") if l.strip()}
    P2 = calib["P2"]
    K = P2[:3, :3]
    # GT poses are cam0's; image_2 is cam0 shifted by the rectified baseline:
    # P2 = K [I | b], so X_cam2 = X_cam0 + b.
    b = np.linalg.solve(K, P2[:, 3])
    cam2_to_cam0 = np.eye(4)
    cam2_to_cam0[:3, 3] = -b
    poses = np.loadtxt(f"{root}/poses/{seq}.txt").reshape(-1, 3, 4)
    assert len(poses) == len(imgs), (len(poses), len(imgs))
    frames = []
    for i, p in enumerate(imgs):
        c2w0 = np.eye(4)
        c2w0[:3] = poses[i]
        frames.append(dict(fid=i, img=p, depth=None, K=K, c2w=c2w0 @ cam2_to_cam0))
    return frames


def kitti_depthval_frames(root, drive):
    imgs = sorted(glob.glob(f"{root}/{drive[:10]}/{drive}/image_02/data/*.png"))
    W, H = cv2.imread(imgs[0]).shape[1::-1]
    sel = sorted(glob.glob(f"{KITTI_SEL_INTR}/{drive}_image_*_image_02.txt"))
    K = np.loadtxt(sel[0]).reshape(3, 3)
    # The selection intrinsics are for the devkit's 1216x352 crop (centred
    # horizontally, bottom-anchored); verified against the odometry P2 of
    # 2011_09_30 and 2011_10_03 to the 4th decimal.
    K[0, 2] += (W - 1216) // 2
    K[1, 2] += H - 352
    gt_dir = f"{root}/val/{drive}/proj_depth/groundtruth/image_02"
    frames = []
    for i, p in enumerate(imgs):
        g = f"{gt_dir}/{osp.basename(p)}"
        frames.append(dict(fid=i, img=p, depth=g if osp.exists(g) else None,
                           depth_kind="kitti", K=K, c2w=None))
    return frames


def read_depth(path, kind):
    d = cv2.imread(path, cv2.IMREAD_UNCHANGED).astype(np.float32)
    if kind == "nrgbd":
        d = np.nan_to_num(d, nan=0.0) / 1000.0
        d[(d > 10) | (d < 1e-3)] = 0  # CUT3R's NRGBD loader
    else:
        d = d / 256.0
    return d


# ----------------------------------------------------------- preprocessing
def crop_geometry(W1, H1, size=512):
    """Replicates dust3r.utils.image.load_images (size != 224): long side to
    `size`, then a centred crop to multiples of 16."""
    S = max(W1, H1)
    Wn, Hn = (int(round(x * size / S)) for x in (W1, H1))
    cx, cy = Wn // 2, Hn // 2
    halfw, halfh = ((2 * cx) // 16) * 8, ((2 * cy) // 16) * 8
    if Wn == Hn:
        halfh = 3 * halfw / 4
    return (Wn, Hn), (int(cx - halfw), int(cy - halfh), int(cx + halfw), int(cy + halfh))


def transform_K(K, W1, H1, new, box):
    sx, sy = new[0] / W1, new[1] / H1
    K2 = K.astype(np.float64).copy()
    K2[0, 0] *= sx
    K2[1, 1] *= sy
    K2[0, 2] = (K[0, 2] + 0.5) * sx - 0.5 - box[0]
    K2[1, 2] = (K[1, 2] + 0.5) * sy - 0.5 - box[1]
    return K2


class StreamDataset(torch.utils.data.Dataset):
    def __init__(self, frames, order):
        self.frames = frames
        self.order = order

    def __len__(self):
        return len(self.order)

    def __getitem__(self, pos):
        f = self.frames[self.order[pos]]
        im = load_images([f["img"]], size=512, verbose=False)[0]
        H1, W1 = cv2.imread(f["img"]).shape[:2]
        new, box = crop_geometry(W1, H1)
        out = dict(pos=pos, fid=f["fid"], img=im["img"][0],
                   true_shape=torch.from_numpy(im["true_shape"][0]),
                   K=torch.from_numpy(transform_K(f["K"], W1, H1, new, box)))
        Hc, Wc = box[3] - box[1], box[2] - box[0]
        if f["depth"] is not None:
            d = read_depth(f["depth"], f["depth_kind"])
            assert d.shape == (H1, W1), (d.shape, H1, W1)
            d = cv2.resize(d, new, interpolation=cv2.INTER_NEAREST)
            out["depth"] = torch.from_numpy(d[box[1]:box[3], box[0]:box[2]].copy())
        else:
            out["depth"] = torch.zeros(Hc, Wc)
        out["has_depth"] = f["depth"] is not None
        c2w = f["c2w"]
        out["c2w_gt"] = torch.from_numpy(c2w if c2w is not None else np.full((4, 4), np.nan))
        return out


# ----------------------------------------------------------------- metrics
def sim3(src, dst, with_scale=True):
    """Umeyama: s, R, t minimising |s R src + t - dst|. float64 because KITTI
    frame-0 coordinates reach kilometres."""
    src, dst = src.double(), dst.double()
    ms, md = src.mean(0), dst.mean(0)
    xs, xd = src - ms, dst - md
    U, S, Vt = torch.linalg.svd(xd.T @ xs / len(src))
    D = torch.ones(3, dtype=src.dtype, device=src.device)
    if torch.det(U) * torch.det(Vt) < 0:
        D[2] = -1
    R = U @ torch.diag(D) @ Vt
    s = (S * D).sum() / (xs.square().sum() / len(src)) if with_scale else src.new_tensor(1.0)
    return s, R, md - s * R @ ms


def apply(s, R, t, X):
    return s * X.double() @ R.T + t


def frame_metrics(Xs, Xw, c2w, D, K, has_depth):
    m = {}
    xs, xw = Xs.reshape(-1, 3), Xw.reshape(-1, 3)
    zmed = Xs[..., 2].median().double()
    s, R, t = sim3(xs, xw)
    r = (apply(s, R, t, xs) - xw.double()).norm(dim=-1)
    m["sw_res_mean"], m["sw_res_med"] = r.mean() / zmed, r.median() / zmed
    T = torch.eye(4, dtype=torch.float64, device=Xs.device)
    T[:3, :3], T[:3, 3] = s * R, t
    m["T_sw"] = T
    r = (xs.double() @ c2w[:3, :3].double().T + c2w[:3, 3].double() - xw.double()).norm(dim=-1)
    m["pw_res_mean"], m["pw_res_med"] = r.mean() / zmed, r.median() / zmed
    m["xw_norm_med"] = xw.norm(dim=-1).median()
    m["xs_depth_med"] = zmed

    nan = torch.tensor(float("nan"), device=Xs.device)
    keys = ["self_shape_mean", "self_shape_med", "self_shape_scale", "world_shape_mean",
            "world_shape_med", "world_shape_scale", "self_absrel", "self_absrel_metric",
            "self_delta125", "n_valid"]
    V = (D > 0) & torch.isfinite(D) if has_depth else torch.zeros_like(D, dtype=torch.bool)
    if int(V.sum()) < 50:
        m.update({k: nan for k in keys})
        return m
    H, W = D.shape
    v, u = torch.meshgrid(torch.arange(H, device=D.device), torch.arange(W, device=D.device),
                          indexing="ij")
    Kd = K.double()
    Dd = D.double()
    Y = torch.stack([(u - Kd[0, 2]) / Kd[0, 0] * Dd, (v - Kd[1, 2]) / Kd[1, 1] * Dd, Dd], -1)[V]
    dmed = Dd[V].median()
    for name, X in (("self", Xs), ("world", Xw)):
        s, R, t = sim3(X[V], Y)
        r = (apply(s, R, t, X[V]) - Y).norm(dim=-1)
        m[f"{name}_shape_mean"], m[f"{name}_shape_med"] = r.mean() / dmed, r.median() / dmed
        m[f"{name}_shape_scale"] = s
    z, d = Xs[..., 2][V].double(), Dd[V]
    m["self_absrel_metric"] = ((z - d).abs() / d).mean()
    zm = z * (d.median() / z.median())
    m["self_absrel"] = ((zm - d).abs() / d).mean()
    m["self_delta125"] = (torch.maximum(zm / d, d / zm) < 1.25).double().mean()
    m["n_valid"] = V.sum().double()
    return m


# -------------------------------------------------------------------- main
def run_stream(model, loader, device, resets=()):
    """Yields (batch, res) per stream position, one frame per recurrent step.
    A position in `resets` is predicted with the current state, which the model
    then restores to its initial value, so the NEXT position starts fresh."""
    true = torch.tensor([True], device=device)
    false = torch.tensor([False], device=device)
    resets = set(resets)
    with torch.no_grad():
        for it in loader:
            i = int(it["pos"])
            img = it["img"].to(device, non_blocking=True)
            shape = it["true_shape"].to(device)
            view = dict(img=img, true_shape=shape, img_mask=true, ray_mask=false,
                        update=true, reset=true if i in resets else false)
            feat_ls, pos_i, _ = model._encode_image(img, shape)
            feat_i = feat_ls[-1]
            if i == 0:
                state_feat, state_pos = model._init_state(feat_i, pos_i)
                mem = model.pose_retriever.mem.expand(1, -1, -1)
                init_state_feat, init_mem = state_feat.clone(), mem.clone()
            res, (state_feat, mem) = model._forward_decoder_step(
                {i: view}, i, feat_i, pos_i, shape, init_state_feat, init_mem,
                state_feat, state_pos, mem)
            yield it, res


def git_rev():
    try:
        return subprocess.check_output(["git", "-C", REPO, "rev-parse", "HEAD"], text=True).strip()
    except Exception:
        return None


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--benchmark", required=True, choices=["nrgbd", "kitti", "kitti_depthval"])
    ap.add_argument("--scene", required=True)
    ap.add_argument("--root", required=True)
    ap.add_argument("--out", required=True)
    ap.add_argument("--order", default="palindrome", choices=["palindrome", "forward", "repeat", "chunked"])
    ap.add_argument("--chunk", type=int, default=50,
                    help="--order chunked: frames per chunk; the state is reset after each chunk")
    ap.add_argument("--overlap", type=int, default=1,
                    help="--order chunked: frames re-fed at the start of the next chunk")
    ap.add_argument("--repeat", type=int, default=4, help="passes for --order repeat")
    ap.add_argument("--reset-between-passes", type=int, default=0,
                    help="--order repeat: reset the state after the last frame of each pass")
    ap.add_argument("--start", type=int, default=0, help="first source frame of the segment")
    ap.add_argument("--end", type=int, default=0, help="one past the last source frame; 0 = all")
    ap.add_argument("--max-frames", type=int, default=0, help="truncate the sequence (smoke tests)")
    ap.add_argument("--dump-stride", type=int, default=8, help="pixel stride of the point dump; 0 disables")
    ap.add_argument("--workers", type=int, default=8)
    ap.add_argument("--tf32", type=int, default=0,
                    help="croco.py turns TF32 matmul on at import; off by default so "
                         "arithmetic rounding cannot pose as a distance effect")
    args = ap.parse_args()
    torch.backends.cuda.matmul.allow_tf32 = bool(args.tf32)
    torch.backends.cudnn.allow_tf32 = bool(args.tf32)

    frames = {"nrgbd": nrgbd_frames, "kitti": kitti_odo_frames,
              "kitti_depthval": kitti_depthval_frames}[args.benchmark](args.root, args.scene)
    frames = frames[args.start: args.end or None]
    if args.max_frames:
        frames = frames[: args.max_frames]
    N = len(frames)
    resets = []
    if args.order == "chunked":
        # Chunk j+1 starts on the last `overlap` frames of chunk j, so the first
        # of them has a pose in both chunk frames: that pair chains the two.
        order, s0 = [], 0
        while True:
            e = min(s0 + args.chunk, N)
            order += list(range(s0, e))
            if e >= N:
                break
            resets.append(len(order) - 1)
            s0 = e - args.overlap
    elif args.order == "repeat":
        order = list(range(N)) * args.repeat
        if args.reset_between_passes:
            resets = [p * N - 1 for p in range(1, args.repeat)]
    else:
        order = list(range(N)) + (list(range(N - 2, -1, -1)) if args.order == "palindrome" else [])

    device = "cuda"
    model = ARCroco3DStereo.from_pretrained(CKPT).to(device).eval()
    loader = torch.utils.data.DataLoader(StreamDataset(frames, order), batch_size=1,
                                         num_workers=args.workers, pin_memory=True)
    os.makedirs(args.out, exist_ok=True)

    rec = {}
    dump = {"Xs": [], "Xw": [], "conf_self": [], "conf": [], "depth": []}
    t0 = time.time()
    with torch.no_grad():
        for it, res in run_stream(model, loader, device, resets):
            i = int(it["pos"])
            Xs = res["pts3d_in_self_view"][0].float()
            Xw = res["pts3d_in_other_view"][0].float()
            c2w = pose_encoding_to_camera(res["camera_pose"].float())[0]
            D = it["depth"][0].to(device)
            m = frame_metrics(Xs, Xw, c2w, D, it["K"][0].to(device), bool(it["has_depth"]))
            m["c2w_pred"] = c2w
            m["c2w_gt"] = it["c2w_gt"][0]
            m["conf_self_mean"] = res["conf_self"][0].mean()
            m["conf_mean"] = res["conf"][0].mean()
            m["pos"], m["fid"] = torch.tensor(i), it["fid"][0]
            for k, v in m.items():
                rec.setdefault(k, []).append(v.detach().double().cpu().numpy())
            if args.dump_stride:
                st = args.dump_stride
                dump["Xs"].append(Xs[::st, ::st].cpu().numpy())
                dump["Xw"].append(Xw[::st, ::st].cpu().numpy())
                dump["conf_self"].append(res["conf_self"][0][::st, ::st].half().cpu().numpy())
                dump["conf"].append(res["conf"][0][::st, ::st].half().cpu().numpy())
                dump["depth"].append(D[::st, ::st].cpu().numpy())
            if i % 200 == 0 or i == len(order) - 1:
                el = time.time() - t0
                print(f"[{args.benchmark}/{args.scene}] {i + 1}/{len(order)} "
                      f"{(i + 1) / el:.1f} fps  |c|={m['c2w_pred'][:3, 3].norm():.2f} "
                      f"world/self shape={float(m['world_shape_med']):.4f}/{float(m['self_shape_med']):.4f}",
                      flush=True)

    elapsed = time.time() - t0
    np.savez(f"{args.out}/frames.npz", **{k: np.stack(v) for k, v in rec.items()})
    if args.dump_stride:
        np.savez(f"{args.out}/points_s{args.dump_stride}.npz", **{k: np.stack(v) for k, v in dump.items()})
    json.dump(dict(benchmark=args.benchmark, scene=args.scene, root=args.root, order=args.order,
                   n_frames=N, n_positions=len(order), max_frames=args.max_frames,
                   start=args.start, end=args.end, repeat=args.repeat if args.order == "repeat" else None,
                   chunk=args.chunk if args.order == "chunked" else None,
                   overlap=args.overlap if args.order == "chunked" else None,
                   resets=resets,
                   ckpt=CKPT, git=git_rev(), tf32=bool(args.tf32), gpu=torch.cuda.get_device_name(),
                   torch=torch.__version__, seconds=elapsed, fps=len(order) / elapsed,
                   hostname=os.uname().nodename, slurm_job=os.environ.get("SLURM_JOB_ID")),
              open(f"{args.out}/meta.json", "w"), indent=2)
    print(f"[done] {args.benchmark}/{args.scene}: {len(order)} positions in {elapsed:.0f}s", flush=True)


if __name__ == "__main__":
    main()
