import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
import json
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per scene
MIN_NUM_IMAGES = 10

def process_single_scene(scene_key, COP3D_DIR, image_pool):
    """
    Process a single scene and return its metadata
    Returns: (scene_key, scene_data_dict) or (scene_key, None) if skipped

    Args:
        scene_key: Tuple of (obj, instance)
        COP3D_DIR: Root directory
        image_pool: List of frame indices for this scene
    """
    obj, instance = scene_key

    if len(image_pool) < MIN_NUM_IMAGES:
        return (scene_key, None)

    scene_dir = osp.join(COP3D_DIR, obj, instance)
    images_dir = osp.join(scene_dir, "images")

    if not osp.exists(images_dir):
        return (scene_key, None)

    # Collect metadata for all frames in the image pool
    c2ws = []
    intrinsics = []
    valid_frames = []

    for view_idx in image_pool:
        metadata_path = osp.join(images_dir, f"frame{view_idx:06n}.npz")

        if not osp.exists(metadata_path):
            logging.warning(f"Metadata not found: {metadata_path}")
            continue

        try:
            metadata = np.load(metadata_path)
            c2ws.append(metadata["camera_pose"].astype(np.float32))
            intrinsics.append(metadata["camera_intrinsics"].astype(np.float32))
            valid_frames.append(view_idx)
        except Exception as e:
            logging.warning(f"Failed to load metadata for {obj}/{instance}/frame{view_idx:06n}: {e}")
            continue

    if len(valid_frames) < MIN_NUM_IMAGES:
        return (scene_key, None)

    scene_data = {
        "frames": np.array(valid_frames, dtype=np.int32),
        "c2ws": np.array(c2ws),
        "intrinsics": np.array(intrinsics),
    }

    return (scene_key, scene_data)

def fetch_and_save_cop3d_metadata(COP3D_DIR, output_file, num_workers=None):
    """
    Process Cop3D dataset and save all splits' metadata in a single npz file

    Args:
        COP3D_DIR: Root directory of Cop3D dataset
        output_file: Output path for the npz file
        num_workers: Number of worker processes (default: cpu_count())
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    logging.info(f"Using {num_workers} worker processes")

    # Dictionary to store all splits
    all_splits_dict = {}

    # Process train, test, and val splits
    for split in ["train", "test", "val"]:
        # Load scene list from JSON file (like Co3d)
        scene_list_file = osp.join(COP3D_DIR, f"selected_seqs_{split}.json")
        if not osp.exists(scene_list_file):
            logging.warning(f"Scene list file not found: {scene_list_file}, skipping {split}")
            continue

        logging.info(f"Loading scene list from {scene_list_file}")
        with open(scene_list_file, "r") as f:
            scenes_json = json.load(f)
            # Filter out empty scenes
            scenes_json = {k: v for k, v in scenes_json.items() if len(v) > 0}
            # Flatten to (obj, instance) -> frame_list mapping
            scenes = {
                (k, k2): v2 for k, v in scenes_json.items() for k2, v2 in v.items()
            }

        logging.info(f"Found {len(scenes)} scenes in {split} split")

        # Prepare arguments for parallel processing
        scene_keys = list(scenes.keys())
        process_args = [(scene_key, COP3D_DIR, scenes[scene_key]) for scene_key in scene_keys]

        # Process scenes in parallel
        scene_dict = {}
        processed_scenes = 0
        skipped_scenes = 0

        with Pool(processes=num_workers) as pool:
            # Use starmap for multiple arguments
            results = list(tqdm(
                pool.starmap(process_single_scene, process_args),
                total=len(process_args),
                desc=f"Processing {split} scenes"
            ))

        # Collect results
        for scene_key, scene_data in results:
            if scene_data is not None:
                # Convert tuple key to string for npz storage
                obj, instance = scene_key
                scene_name = f"{obj}_{instance}"
                scene_dict[scene_name] = scene_data
                processed_scenes += 1
                logging.info(f"Processed {split}/{obj}/{instance}: {len(scene_data['frames'])} frames")
            else:
                skipped_scenes += 1

        # Store this split's data in the all_splits_dict
        all_splits_dict[split] = scene_dict
        logging.info(f"Cop3D {split} processing complete: {processed_scenes} scenes processed, {skipped_scenes} scenes skipped")

    # Save all splits in a single npz file
    np.savez_compressed(
        output_file,
        **all_splits_dict
    )
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Cop3D dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default="/home/storage-bucket/haian/zipmap_data/processed_cop3d", help="Input Cop3D directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: cpu_count())")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_scenes_dict.npz")

    fetch_and_save_cop3d_metadata(args.input_dir, args.output_file, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
