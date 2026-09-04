import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per scene
MIN_NUM_IMAGES = 10

def process_single_scene(scene, DYNAMIC_REPLICA_DIR, split):
    """
    Process a single scene and return its metadata
    Returns: (scene_name, scene_data_dict) or (scene_name, None) if skipped
    """
    scene_dir = osp.join(DYNAMIC_REPLICA_DIR, split, scene, "left")
    rgb_dir = osp.join(scene_dir, "rgb")
    cam_dir = osp.join(scene_dir, "cam")

    if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
        return (scene, None)

    # Get all RGB image basenames (without extension), sorted by numeric value
    try:
        rgb_files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith(".png")
        ], key=lambda x: float(osp.splitext(x)[0]))
    except Exception as e:
        logging.warning(f"Failed to list files in {rgb_dir}: {e}")
        return (scene, None)

    basenames = [osp.splitext(f)[0] for f in rgb_files]
    if len(basenames) < MIN_NUM_IMAGES:
        return (scene, None)

    images = []
    c2ws = []
    intrinsics = []

    for basename in basenames:
        cam_path = osp.join(cam_dir, basename + ".npz")
        if not osp.exists(cam_path):
            continue
        try:
            cam = np.load(cam_path)
            c2ws.append(cam["pose"].astype(np.float32))
            intrinsics.append(cam["intrinsics"].astype(np.float32))
            images.append(basename)
        except Exception as e:
            logging.warning(f"Failed to load camera data for {basename} in {scene}: {e}")
            continue

    if len(images) < MIN_NUM_IMAGES:
        return (scene, None)

    scene_data = {
        "images": np.array(images),
        "c2ws": np.array(c2ws),
        "intrinsics": np.array(intrinsics),
    }

    return (scene, scene_data)

def fetch_and_save_dynamic_replica_metadata(DYNAMIC_REPLICA_DIR, output_file, num_workers=None):
    """
    Process Dynamic Replica dataset and save metadata for all splits in a single file

    Args:
        DYNAMIC_REPLICA_DIR: Root directory of Dynamic Replica dataset
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

    # Process both train and test splits
    for split in ["train", "test"]:
        split_dir = osp.join(DYNAMIC_REPLICA_DIR, split)
        if not osp.exists(split_dir):
            logging.warning(f"Split directory not found: {split_dir}, skipping {split}")
            continue

        # Get all scenes in this split
        all_scenes = sorted([
            f for f in os.listdir(split_dir)
            if os.path.isdir(osp.join(split_dir, f))
        ])

        logging.info(f"Found {len(all_scenes)} scenes in {split} split")

        # Create partial function with parameters bound
        process_func = partial(process_single_scene, DYNAMIC_REPLICA_DIR=DYNAMIC_REPLICA_DIR, split=split)

        # Process scenes in parallel
        scene_dict = {}
        processed_scenes = 0
        skipped_scenes = 0

        with Pool(processes=num_workers) as pool:
            # Use imap for progress bar
            results = list(tqdm(
                pool.imap(process_func, all_scenes),
                total=len(all_scenes),
                desc=f"Processing {split} scenes"
            ))

        # Collect results
        for scene_name, scene_data in results:
            if scene_data is not None:
                scene_dict[scene_name] = scene_data
                processed_scenes += 1
                logging.info(f"Processed {split}/{scene_name}: {len(scene_data['images'])} images")
            else:
                skipped_scenes += 1

        # Store this split's data in the all_splits_dict
        all_splits_dict[split] = scene_dict
        logging.info(f"Dynamic Replica {split} processing complete: {processed_scenes} scenes processed, {skipped_scenes} scenes skipped")

    # Save all splits in a single npz file
    np.savez_compressed(
        output_file,
        **all_splits_dict
    )
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Dynamic Replica dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default='/home/storage-bucket/haian/zipmap_data/processed_dynamic_replica/', help="Input Dynamic Replica directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: cpu_count())")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_scenes_dict.npz")

    fetch_and_save_dynamic_replica_metadata(args.input_dir, args.output_file, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
