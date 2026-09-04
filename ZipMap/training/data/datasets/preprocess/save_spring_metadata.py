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

def process_single_scene(scene, SPRING_DIR):
    """
    Process a single scene and return its metadata
    Returns: (scene_name, scene_data_dict) or (scene_name, None) if skipped
    """
    scene_dir = osp.join(SPRING_DIR, scene)
    rgb_dir = osp.join(scene_dir, "rgb")
    cam_dir = osp.join(scene_dir, "cam")

    if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
        return (scene, None)

    # Get all RGB image basenames (without extension), sorted
    try:
        rgb_files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith(".png")
        ])
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

def fetch_and_save_spring_metadata(SPRING_DIR, output_file, num_workers=None):
    """
    Process Spring dataset and save all metadata in a single npz file

    Args:
        SPRING_DIR: Root directory of Spring dataset
        output_file: Output path for the npz file
        num_workers: Number of worker processes (default: cpu_count())
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get all scenes (top-level directories)
    all_scenes = sorted([
        f for f in os.listdir(SPRING_DIR)
        if os.path.isdir(osp.join(SPRING_DIR, f))
    ])

    logging.info(f"Found {len(all_scenes)} scenes to process")

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    logging.info(f"Using {num_workers} worker processes")

    # Create partial function with SPRING_DIR bound
    process_func = partial(process_single_scene, SPRING_DIR=SPRING_DIR)

    # Process scenes in parallel
    scene_dict = {}
    processed_scenes = 0
    skipped_scenes = 0

    with Pool(processes=num_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(process_func, all_scenes),
            total=len(all_scenes),
            desc="Processing scenes"
        ))

    # Collect results
    for scene_name, scene_data in results:
        if scene_data is not None:
            scene_dict[scene_name] = scene_data
            processed_scenes += 1
            logging.info(f"Processed {scene_name}: {len(scene_data['images'])} images")
        else:
            skipped_scenes += 1

    # Save as a single npz with dict-of-arrays
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    logging.info(f"Spring processing complete: {processed_scenes} scenes processed, {skipped_scenes} scenes skipped")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Spring dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default='/home/storage-bucket/haian/zipmap_data/processed_spring', help="Input Spring directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: cpu_count())")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_scenes_dict.npz")

    fetch_and_save_spring_metadata(args.input_dir, args.output_file, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
