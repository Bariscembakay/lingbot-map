"""Pre-warm viewer.py's SceneCache for every scene/method in a workspace, so
opening any scene in the live viser viewer is instant instead of computing
point clouds on first click.

Mirrors viewer.py's ClientClosures._rebuild_scene() for the *default* GUI
state (spatial_subsample=2, temporal_subsample irrelevant to caching,
remove_sky=False, confidence_threshold=0.3 when confidence data exists) --
that's the state a client sees on first load, so a cache warmed with these
params is a guaranteed hit. If you change the Spatial slider before Load,
the cache is invalidated and rebuilt because SceneCache.is_valid() checks
spatial_subsample, so keep the slider at 2 to get the fast path.

_load_trajectory / _load_intrinsics_as_dict / _generate_point_clouds are
ClientClosures instance methods that never touch `self` -- called unbound
below (passing None as self) to reuse them without spinning up a real viser
client.

Usage (from benchmark/, in the lingbot_map env):
    micromamba run -n lingbot_map python \
        ../.agents/scratch/reproduction/precompute_viewer_cache.py \
        <workspace_dir> [--dataset NAME] [--workers N]
"""
import argparse
import logging
import sys
from pathlib import Path
from concurrent.futures import ProcessPoolExecutor, as_completed

import numpy as np
from PIL import Image

sys.path.insert(0, '.')
import viewer
from viewer import scan_workspace, SceneCache, ClientClosures
from benchmark.core.storage import BSSManager

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(message)s')
logger = logging.getLogger('precompute')

SPATIAL_SUBSAMPLE = 2
CONFIDENCE_THRESHOLD = 0.3


def pred_viewonly_transform(artifact):
    """4x4 from eval/traj_transform_viewonly.txt, or None. See the fallback note
    in warm_one() for why this is a separate file rather than the real one."""
    f = artifact.eval_dir / "traj_transform_viewonly.txt"
    if not f.exists():
        return None
    try:
        T = np.loadtxt(f)
        return T if T.shape == (4, 4) else None
    except Exception as e:
        logger.warning(f"unreadable {f}: {e}")
        return None


def warm_one(workspace_str: str, dataset: str, scene: str, method: str) -> str:
    workspace = Path(workspace_str)
    tag = f"{dataset}/{scene}/{method}"
    try:
        bss_manager = BSSManager(workspace)
        method_name = None if method == 'gt' else method
        artifact = bss_manager.get_artifact(dataset, scene, method_name)
        loader = viewer.BSSLoader(artifact)
        N = loader.get_num_frames()

        traj_dict = ClientClosures._load_trajectory(None, loader)
        if not traj_dict:
            return f"SKIP  {tag}: no trajectory"

        umeyama_transform = loader.load_traj_transform()
        # Fallback for datasets whose eval config disables the trajectory metric
        # (neural_rgbd: a reconstruction benchmark scored on points, so no
        # traj_transform.txt is ever produced and its predictions would stay in
        # their own frame -- ~2.8x out on breakfast_room). Written by
        # make_viewonly_transform.py under a DIFFERENT name so it cannot be
        # mistaken for an eval artifact, and so `evaluate.py` -- which rmtree's
        # eval_dir -- cannot silently destroy it.
        if umeyama_transform is None:
            vo = pred_viewonly_transform(artifact)
            if vo is not None:
                umeyama_transform = vo
                logger.info(f"{dataset}/{scene}/{method}: using view-only transform")
        has_transform = umeyama_transform is not None

        metadata = artifact.read_metadata() or {}
        image_width = metadata.get('image_width', 0)
        image_height = metadata.get('image_height', 0)
        if not image_width or not image_height:
            rgb_list = loader.load_rgb_list()
            if rgb_list:
                image_height, image_width = rgb_list[0].shape[:2]

        cache = SceneCache(artifact.root, SPATIAL_SUBSAMPLE, workspace, remove_sky=False)

        # RGB thumbnails are a SEPARATE cache file with its own validity check, so
        # they are warmed before the scene-data early return -- otherwise the 287
        # entries whose scene data is already valid would never get one, which is
        # why only 5 of ~292 had thumbnails. Without them there is no way to see
        # WHAT the camera saw at frame t, here or in ASVGGT's viser_view_cloud.py,
        # which reads the same file. Same recipe as viewer.py's own block: LANCZOS
        # to 400 px tall, uint8, one per frame in order.
        thumbs = "thumbs cached"
        if not cache.is_valid_rgb(N, image_width, image_height):
            rgb_list = loader.load_rgb_list()
            if rgb_list:
                out = []
                for rgb in rgb_list:
                    im = Image.fromarray(rgb)
                    w = int(400 * (float(im.width) / im.height))
                    out.append(np.array(im.resize((w, 400), Image.Resampling.LANCZOS)))
                cache.save_rgb_thumbnails(out, image_width, image_height)
                thumbs = f"{len(out)} thumbs"
            else:
                thumbs = "no rgb"

        if cache.is_valid(N, image_width, image_height, has_transform):
            return f"CACHED {tag}: aligned={has_transform}, {thumbs}"

        has_confidence = artifact.confidence_dir.exists()
        conf_thresh = CONFIDENCE_THRESHOLD if has_confidence else 0.0

        point_clouds, use_global = ClientClosures._generate_point_clouds(
            None, loader, SPATIAL_SUBSAMPLE, umeyama_transform,
            confidence_threshold=conf_thresh, sky_masks=None,
        )

        is_aligned = False
        if umeyama_transform is not None:
            transformed = {}
            for frame_idx, c2w in traj_dict.items():
                c2w_t = umeyama_transform @ c2w
                U, _, Vh = np.linalg.svd(c2w_t[:3, :3])
                R_pure = U @ Vh
                if np.linalg.det(R_pure) < 0:
                    R_pure = -R_pure
                c2w_t[:3, :3] = R_pure
                transformed[frame_idx] = c2w_t
            traj_dict = transformed
            is_aligned = True

        if use_global:
            return f"GLOBAL {tag}: global point cloud, viewer doesn't cache these"

        cache.save(traj_dict, point_clouds, N, image_width, image_height, is_aligned)

        return f"DONE  {tag}: {len(point_clouds)} frames, aligned={is_aligned}, {thumbs}"
    except Exception as e:
        return f"ERROR {tag}: {e}"


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("workspace", type=Path)
    parser.add_argument("--dataset", default=None, help="Only warm this dataset (default: all)")
    parser.add_argument("--workers", type=int, default=16)
    parser.add_argument("--scene", default=None, help="Only warm this scene (smoke test)")
    args = parser.parse_args()

    _, structure = scan_workspace(args.workspace)
    jobs = []
    for dataset, scenes in structure.items():
        if args.dataset and dataset != args.dataset:
            continue
        for scene, methods in scenes.items():
            if args.scene and scene != args.scene:
                continue
            for method in methods:
                jobs.append((str(args.workspace), dataset, scene, method))

    logger.info(f"Warming {len(jobs)} (dataset/scene/method) entries with {args.workers} workers")

    done = 0
    with ProcessPoolExecutor(max_workers=args.workers) as pool:
        futures = [pool.submit(warm_one, *job) for job in jobs]
        for fut in as_completed(futures):
            done += 1
            logger.info(f"[{done}/{len(jobs)}] {fut.result()}")

    logger.info("All done")


if __name__ == "__main__":
    main()
