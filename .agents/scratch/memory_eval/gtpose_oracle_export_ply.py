"""RGB prediction clouds of the oracle arms, in the GT frame, for the ASVGGT
viser viewer.

Each arm's cloud is rebuilt from its own depth/intrinsics/traj/rgb with the
benchmark's loader, then moved into the GT frame with the Sim(3) its traj eval
saved (eval/traj_transform.txt), i.e. camera-trajectory alignment only. The
points eval additionally runs Umeyama on points + ICP, which would hide part of
the pose error this comparison is meant to show.
Writes <arm>/<bench>/<scene>/pred_aligned.ply, the name the ASVGGT viewer's
catalogue picks up (campaign "lingbot_map", strategy "<bench> | <arm>").
Usage (from benchmark/, PYTHONPATH=.): python gtpose_oracle_export_ply.py <bench> [scene ...]
"""
import logging, sys
from pathlib import Path
import numpy as np
from plyfile import PlyData, PlyElement
from benchmark.core.config import ConfigManager
from benchmark.core.loader import BSSLoader
from benchmark.core.registry import ClassLoader
from benchmark.core.storage import BSSManager

C = Path("/group/compact-3dmem/campaigns/pose_oracle")
ARMS = ("predpose_rerun", "gtpose_oracle", "gtdepth_predpose")
N = 3_000_000
bench = sys.argv[1]
scenes = sys.argv[2:] or sorted(p.name for p in (C / ARMS[0] / bench).iterdir()
                                if p.is_dir() and not p.name.startswith("_") and p.name != "eval")
log = logging.getLogger("export")
cfg = ConfigManager(Path(f"configs/gtpose_oracle/predpose_rerun_{bench}.yaml")).get_dataset_config(bench)
dataset = ClassLoader.load_dataset(cfg["dataset_class"])(logger=log, **dict(cfg["params"]))
rng = np.random.default_rng(0)
for scene, arm in ((s, a) for s in scenes for a in ARMS):
    T_path = C / arm / bench / scene / "lingbot_map/eval/traj_transform.txt"
    f = C / arm / bench / scene / "pred_aligned.ply"
    if f.exists() or not T_path.exists():
        continue
    art = BSSManager(C / arm).get_artifact(bench, scene, "lingbot_map")
    xyzrgb, mask = BSSLoader(art, context=dataset).load_point_cloud_grid()
    p = xyzrgb[mask]
    p = p[rng.choice(len(p), min(N, len(p)), replace=False)]
    T = np.loadtxt(T_path)
    xyz = p[:, :3] @ T[:3, :3].T + T[:3, 3]
    rgb = p[:, 3:6]
    rgb = (rgb * 255 if rgb.max() <= 1.0 else rgb).clip(0, 255).astype(np.uint8)
    a = np.empty(len(p), dtype=[("x", "f4"), ("y", "f4"), ("z", "f4"),
                                ("red", "u1"), ("green", "u1"), ("blue", "u1")])
    a["x"], a["y"], a["z"] = xyz.T
    a["red"], a["green"], a["blue"] = rgb.T
    PlyData([PlyElement.describe(a, "vertex")]).write(f)
    print(f, flush=True)
