"""Pose-headroom oracle: lingbot's own depth placed with GT camera poses.

Builds two arms beside campaigns/reproduction/default, both symlinking lingbot's
saved depth/confidence/rgb/intrinsics so only the trajectory differs:
  predpose_rerun -- lingbot's own traj.txt (re-scored with today's eval code,
                    so the comparison never mixes code versions)
  gtpose_oracle  -- GT rotations, GT translations divided by s, where s is the
                    Sim(3) scale of lingbot's camera centres onto GT. Depth stays
                    in lingbot's units, so the cloud keeps lingbot's own
                    depth-vs-pose scale and the eval's Umeyama absorbs s; no GT
                    depth is used to scale anything.
  gtdepth_predpose -- the converse: GT depth/intrinsics/rgb with lingbot's own
                    rotations and translations times s (to metres), i.e. the
                    reconstruction error caused by lingbot's trajectory alone.
"""
import os, sys
from pathlib import Path
import numpy as np

SRC = Path("/group/compact-3dmem/campaigns/reproduction/default")
DST = Path("/group/compact-3dmem/campaigns/pose_oracle")
HEADER = ("# BSS Trajectory Format v2\n"
          "# frame_idx r00 r01 r02 tx r10 r11 r12 ty r20 r21 r22 tz\n"
          "# Represents the 3x4 camera-to-world (C2W) transformation matrix:\n")


def read_traj(p):
    rows = [l.split() for l in open(p) if l.strip() and not l.startswith("#")]
    idx = np.array([int(r[0]) for r in rows])
    m = np.array([[float(x) for x in r[1:13]] for r in rows]).reshape(-1, 3, 4)
    return idx, m


def sim3_scale(src, dst):
    xs, xd = src - src.mean(0), dst - dst.mean(0)
    U, D, Vt = np.linalg.svd(xd.T @ xs / len(src))
    S = np.ones(3); S[2] = np.sign(np.linalg.det(U @ Vt))
    return (D * S).sum() / (xs ** 2).sum(1).mean()


def link_method(src_m, dst_m):
    dst_m.mkdir(parents=True, exist_ok=True)
    # .complete.json: evaluate.py skips a method dir without it.
    for name in ("depth", "confidence", "rgb", "intrinsics.txt", ".complete.json"):
        if (src_m / name).exists() and not (dst_m / name).exists():
            os.symlink(src_m / name, dst_m / name)


for bench in sys.argv[1:]:
    for scene in sorted(p.name for p in (SRC / bench).iterdir()
                        if p.is_dir() and not p.name.startswith("_") and p.name != "eval"):
        s_src = SRC / bench / scene
        pidx, P = read_traj(s_src / "lingbot_map" / "traj.txt")
        gidx, G = read_traj(s_src / "gt" / "traj.txt")
        assert (pidx == gidx).all(), f"{bench}/{scene}: frame indices differ"
        s = sim3_scale(P[:, :, 3], G[:, :, 3])
        for arm in ("predpose_rerun", "gtpose_oracle", "gtdepth_predpose"):
            d = DST / arm / bench / scene
            d.mkdir(parents=True, exist_ok=True)
            if not (d / "gt").exists():
                os.symlink(s_src / "gt", d / "gt")
            m = d / "lingbot_map"
            if arm == "gtdepth_predpose":
                m.mkdir(parents=True, exist_ok=True)
                for name, src in (("depth", s_src / "gt/depth"), ("rgb", s_src / "gt/rgb"),
                                  ("intrinsics.txt", s_src / "gt/intrinsics.txt"),
                                  (".complete.json", s_src / "lingbot_map/.complete.json")):
                    if not (m / name).exists():
                        os.symlink(src, m / name)
            else:
                link_method(s_src / "lingbot_map", m)
            out = m / "traj.txt"
            if arm == "predpose_rerun":
                if not out.exists():
                    os.symlink(s_src / "lingbot_map" / "traj.txt", out)
                continue
            if out.exists():  # never rewrite under a running eval
                continue
            if arm == "gtpose_oracle":
                T = G.copy(); T[:, :, 3] /= s
            else:
                T = P.copy(); T[:, :, 3] *= s
            with open(out, "w") as f:
                f.write(HEADER)
                for i, m in zip(gidx, T):
                    f.write(f"{i} " + " ".join(f"{v:.9f}" for v in m.reshape(-1)) + "\n")
        print(f"{bench}/{scene}: {len(P)} frames, s={s:.4f}")
