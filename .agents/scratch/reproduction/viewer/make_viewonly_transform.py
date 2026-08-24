"""Write a VIEW-ONLY pred->GT Sim(3) for datasets whose eval config disables traj.

WHY THIS EXISTS. The viewer cache stores GT-aligned clouds when
`eval/traj_transform.txt` is present -- SceneCache.is_valid() keys on it, and
precompute_viewer_cache.py applies it to both the points and the trajectory. That
file is a by-product of the TRAJECTORY metric, so any dataset whose config sets
`evaluation.traj.enable: false` never gets one, and its predictions stay in their
own arbitrary frame. Today that is neural_rgbd, 0 of 18 entries: a reconstruction
benchmark scored on points, so disabling traj is deliberate and correct.

The consequence is only a viewing one -- overlaying such a prediction on its GT
shows the frame mismatch rather than the model's error. Measured on
neural_rgbd/breakfast_room before this: pred extent [1.43 1.06 0.99] against GT
[4.01 1.45 3.55], about 2.8x out.

WHY A SEPARATE FILENAME. Enabling traj in the config would produce the transform
as a by-product, but it would also add an ATE the benchmark deliberately does not
report -- a research decision, not a plumbing one. And writing to
`traj_transform.txt` would both claim to be an eval artifact and be destroyed by
the next `evaluate.py` run, which rmtree's eval_dir. So this writes
`traj_transform_viewonly.txt`, which precompute_viewer_cache.py reads only as a
fallback when the real one is absent. Provenance stays obvious and nothing the
evaluator owns is touched.

Same estimator the trajectory evaluator uses: Umeyama with scale, on the
translations of temporally corresponding poses.

Usage (from benchmark/, in the lingbot_map env):
    python ../.agents/scratch/reproduction/viewer/make_viewonly_transform.py \
        <workspace> --dataset neural_rgbd [--method lingbot_map] [--dry-run]
"""
import argparse
import logging
import sys
from pathlib import Path

import numpy as np

sys.path.insert(0, '.')
import viewer
from viewer import scan_workspace, ClientClosures
from benchmark.core.storage import BSSManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('viewonly')

VIEWONLY_NAME = "traj_transform_viewonly.txt"


def umeyama(src, dst, with_scale=True):
    """Sim(3) mapping src onto dst. Same estimator as the trajectory evaluator."""
    n, dim = src.shape
    mu_s, mu_d = src.mean(0), dst.mean(0)
    sc, dc = src - mu_s, dst - mu_d
    Sigma = dc.T @ sc / n
    U, D, Vt = np.linalg.svd(Sigma)
    S = np.eye(dim)
    if np.linalg.det(U) * np.linalg.det(Vt) < 0:
        S[-1, -1] = -1
    R = U @ S @ Vt
    s = (D * S.diagonal()).sum() / ((sc ** 2).sum() / n) if with_scale else 1.0
    return s, R, mu_d - s * R @ mu_s


def one(workspace, dataset, scene, method, dry_run):
    tag = f"{dataset}/{scene}/{method}"
    mgr = BSSManager(Path(workspace))
    pred_art = mgr.get_artifact(dataset, scene, method)
    gt_art = mgr.get_artifact(dataset, scene, None)

    if pred_art.traj_transform_file.exists():
        return f"SKIP  {tag}: has a real traj_transform.txt"

    pred = ClientClosures._load_trajectory(None, viewer.BSSLoader(pred_art))
    gt = ClientClosures._load_trajectory(None, viewer.BSSLoader(gt_art))
    if not pred or not gt:
        return f"SKIP  {tag}: pred={len(pred)} gt={len(gt)} poses"

    # Only frames present in BOTH -- either side can drop poses (NaN rows are
    # excluded by _load_trajectory), and pairing by index would then misalign.
    shared = sorted(set(pred) & set(gt))
    if len(shared) < 3:
        return f"SKIP  {tag}: only {len(shared)} shared poses"
    P = np.stack([pred[i][:3, 3] for i in shared]).astype(np.float64)
    G = np.stack([gt[i][:3, 3] for i in shared]).astype(np.float64)

    s, R, t = umeyama(P, G, True)
    T = np.eye(4)
    T[:3, :3] = s * R
    T[:3, 3] = t
    rmse = float(np.sqrt((((P @ (s * R).T + t) - G) ** 2).sum(1).mean()))

    out = pred_art.eval_dir / VIEWONLY_NAME
    if dry_run:
        return f"DRY   {tag}: n={len(shared)} scale={s:.4f} rmse={rmse:.4f} -> {out}"
    out.parent.mkdir(parents=True, exist_ok=True)
    np.savetxt(out, T, fmt='%.10f',
               header=('4x4 Sim(3) alignment transformation matrix (VIEW ONLY)\n'
                       'Apply as: p_aligned = T @ p_original (homogeneous coords)\n'
                       'Fitted by make_viewonly_transform.py from pred/GT camera\n'
                       'translations because this dataset\'s eval config disables the\n'
                       'trajectory metric. NOT an evaluation artifact; contributes to\n'
                       'no reported number.\n'
                       f'shared_poses={len(shared)} scale={s:.6f} rmse={rmse:.6f}'))
    return f"DONE  {tag}: n={len(shared)} scale={s:.4f} rmse={rmse:.4f}"


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("workspace", type=Path)
    ap.add_argument("--dataset", required=True)
    ap.add_argument("--method", default="lingbot_map")
    ap.add_argument("--dry-run", action="store_true")
    args = ap.parse_args()

    _, structure = scan_workspace(args.workspace)
    scenes = structure.get(args.dataset, {})
    if not scenes:
        logger.error(f"no scenes for dataset {args.dataset} under {args.workspace}")
        return 1
    for scene, methods in sorted(scenes.items()):
        if args.method not in methods:
            logger.info(f"SKIP  {args.dataset}/{scene}: no {args.method}")
            continue
        logger.info(one(args.workspace, args.dataset, scene, args.method, args.dry_run))
    return 0


if __name__ == "__main__":
    sys.exit(main())
