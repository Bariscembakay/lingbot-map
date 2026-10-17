"""Side-by-side renders of the pose-headroom oracle on NRGBD.

Per scene: lingbot cloud with its own poses vs the same depth with GT poses,
each coloured by its distance to the GT surface (the `distance` field the eval
writes into pred_points_0.05.ply with points.vis), clipped at 10 cm. Up comes
from the GT cameras, since the eval cloud lives in the GT frame.
Usage: python gtpose_oracle_render.py <out_dir> [scene ...]
"""
import json, sys
from pathlib import Path
import numpy as np
import matplotlib; matplotlib.use("Agg")
import matplotlib.pyplot as plt
from plyfile import PlyData

C = Path("/group/compact-3dmem/campaigns/pose_oracle")
ARMS = [("predpose_rerun", "lingbot poses"), ("gtpose_oracle", "GT poses (same depth)")]
out = Path(sys.argv[1]); out.mkdir(parents=True, exist_ok=True)
scenes = sys.argv[2:] or sorted(p.name for p in (C / "gtpose_oracle/neural_rgbd").iterdir()
                                if p.is_dir() and not p.name.startswith("_") and p.name != "eval")
rng = np.random.default_rng(0)


def load(arm, scene, n=400_000):
    v = PlyData.read(C / arm / "neural_rgbd" / scene / "lingbot_map/eval/points/pred_points_0.05.ply")["vertex"]
    xyz = np.stack([v["x"], v["y"], v["z"]], 1)
    d = np.asarray(v["distance"])
    i = rng.choice(len(xyz), min(n, len(xyz)), replace=False)
    return xyz[i], d[i]


def gt_up(scene):
    rows = [l.split() for l in open(C / "gtpose_oracle/neural_rgbd" / scene / "gt/traj.txt")
            if l.strip() and not l.startswith("#")]
    R = np.array([[float(x) for x in r[1:13]] for r in rows]).reshape(-1, 3, 4)[:, :, :3]
    u = -R[:, :, 1].mean(0)  # OpenCV camera y points down
    return u / np.linalg.norm(u)


for scene in scenes:
    clouds = [load(a, scene) for a, _ in ARMS]
    u = gt_up(scene)
    e1 = np.cross(u, [1, 0, 0] if abs(u[0]) < 0.9 else [0, 1, 0]); e1 /= np.linalg.norm(e1)
    e2 = np.cross(u, e1)
    fig, ax = plt.subplots(2, 2, figsize=(14, 12))
    for col, ((xyz, d), (arm, title)) in enumerate(zip(clouds, ARMS)):
        m = json.load(open(C / arm / "neural_rgbd" / scene / "lingbot_map/eval/points.json"))
        t = json.load(open(C / arm / "neural_rgbd" / scene / "lingbot_map/eval/traj.json"))
        h, x1, x2 = xyz @ u, xyz @ e1, xyz @ e2
        keep = h < np.percentile(h, 80)  # drop the ceiling for the top view
        kw = dict(c=np.minimum(d, 0.10), cmap="turbo", vmin=0, vmax=0.10, s=0.08, linewidths=0)
        ax[0, col].scatter(x1[keep], x2[keep], **{**kw, "c": kw["c"][keep]})
        ax[0, col].set_title(f"{title}\nChamfer {m['chamfer']*100:.2f} cm | Acc {m['accuracy']*100:.2f} | "
                             f"Comp {m['completeness']*100:.2f} | ATE {t['ate']*100:.2f} cm", fontsize=11)
        sc = ax[1, col].scatter(x1 + 0.6 * x2, h + 0.4 * x2, **kw)
        for r in range(2):
            ax[r, col].set_aspect("equal"); ax[r, col].axis("off")
    fig.suptitle(f"NRGBD {scene}: colour = distance to GT surface (0-10 cm, clipped)", fontsize=13)
    fig.tight_layout()
    fig.colorbar(sc, ax=ax, shrink=0.4, label="distance to GT [m]")
    fig.savefig(out / f"{scene}.png", dpi=110)
    plt.close(fig)
    print(out / f"{scene}.png", flush=True)
