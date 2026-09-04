import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
import json
import cv2
from multiprocessing import Pool, cpu_count
from functools import partial


# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10


def imread_cv2(path, flag=cv2.IMREAD_COLOR):
    """Read image using cv2."""
    return cv2.imread(path, flag)


def read_depthmap(depthpath):
    """Read depth map from WildRGBD format."""
    # WildRGBD stores depths in the depth scale of 1000.
    # When we load depth image and divide by 1000, we get depth in meters.
    depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
    depthmap = depthmap.astype(np.float32) / 1000.0
    return depthmap


def get_metadatapath(ROOT, obj, instance, view_idx):
    return osp.join(ROOT, obj, instance, "metadata", f"{view_idx:05d}.npz")


def get_impath(ROOT, obj, instance, view_idx):
    return osp.join(ROOT, obj, instance, "rgb", f"{view_idx:05d}.jpg")


def get_depthpath(ROOT, obj, instance, view_idx):
    return osp.join(ROOT, obj, instance, "depth", f"{view_idx:05d}.png")


def get_maskpath(ROOT, obj, instance, view_idx):
    return osp.join(ROOT, obj, instance, "masks", f"{view_idx:05d}.png")


def process_single_scene(args):
    """Process a single scene and return its data."""
    scene_key, image_pool, WildRGBD_DIR = args
    obj, instance = scene_key
    
    if len(image_pool) < MIN_NUM_IMAGES:
        return None
        
    # Check if scene directory exists
    scene_path = osp.join(WildRGBD_DIR, obj, instance)
    if not osp.exists(scene_path):
        return None
        
    # Load camera data for this scene
    sequence_trajectories = []
    sequence_intrinsics = []
    valid_images = []
    
    for view_idx in image_pool:
        try:
            # Check if all required files exist
            img_path = get_impath(WildRGBD_DIR, obj, instance, view_idx)
            depth_path = get_depthpath(WildRGBD_DIR, obj, instance, view_idx)
            metadata_path = get_metadatapath(WildRGBD_DIR, obj, instance, view_idx)
            
            if not all(osp.exists(p) for p in [img_path, depth_path, metadata_path]):
                print(f"Missing files for {obj}/{instance} view {view_idx}")
                continue
            
            # Load camera parameters
            input_metadata = np.load(metadata_path)
            camera_pose = input_metadata["camera_pose"].astype(np.float32)
            intrinsics = input_metadata["camera_intrinsics"].astype(np.float32)
            
            sequence_trajectories.append(camera_pose)
            sequence_intrinsics.append(intrinsics)
            valid_images.append(f"{view_idx:05d}")
            
        except Exception:
            print(f"Error processing {obj}/{instance} view {view_idx}")
            continue
    
    if len(valid_images) < MIN_NUM_IMAGES:
        return None
    
    # Store scene data using scene_key as string
    scene_name = f"{obj}_{instance.replace('/', '_')}"
    scene_data = {
        "obj": obj,
        "instance": instance,
        "images": np.array(valid_images),
        "c2ws": np.array(sequence_trajectories),
        "intrinsics": np.array(sequence_intrinsics),
    }
    
    return scene_name, scene_data


def fetch_and_save_wildrgbd_metadata(WildRGBD_DIR, output_file, split="train", num_workers=None):
    """
    Process WildRGBD dataset and save all metadata in a single npz file using multiprocessing
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load selected sequences from JSON file
    scenes_json_path = osp.join(WildRGBD_DIR, f"selected_seqs_{split}.json")
    if not osp.exists(scenes_json_path):
        raise FileNotFoundError(f"Selected sequences file not found: {scenes_json_path}")
    
    with open(scenes_json_path, "r") as f:
        scenes = json.load(f)
        scenes = {k: v for k, v in scenes.items() if len(v) > 0}
        scenes = {
            (k, k2): v2 for k, v in scenes.items() for k2, v2 in v.items()
        }
    
    # Remove problematic scene if it exists
    scenes.pop(("box", "scenes/scene_257"), None)
    
    # Prepare arguments for multiprocessing
    scene_args = [(scene_key, image_pool, WildRGBD_DIR) 
                  for scene_key, image_pool in scenes.items()]
    
    # Use multiprocessing to process scenes
    if num_workers is None:
        num_workers = min(cpu_count() * 2, 192)  # Cap at 8 to avoid too many processes
    
    logging.info(f"Processing {len(scene_args)} scenes using {num_workers} workers")
    
    scene_dict = {}
    processed_sequences = 0
    
    with Pool(num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_single_scene, scene_args),
            total=len(scene_args),
            desc=f"Processing WildRGBD {split} scenes"
        ))
    
    # Collect results
    for result in results:
        if result is not None:
            scene_name, scene_data = result
            scene_dict[scene_name] = scene_data
            processed_sequences += 1
            logging.info(f"Processed {scene_name}: {len(scene_data['images'])} images")
    
    # Save as a single npz with dict
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    
    logging.info(f"WildRGBD {split} processing complete: {processed_sequences} scenes")
    logging.info(f"All metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process WildRGBD dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/haian/Dataset/wildrgbd/",
                       help="Input WildRGBD directory")
    parser.add_argument("--output_file", type=str,                       help="Output npz file for all processed metadata")
    parser.add_argument("--split", type=str, default="train",
                       choices=["train", "test"],
                       help="Dataset split to process")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of worker processes (default: min(cpu_count, 8))")
    
    args = parser.parse_args()
    Test = False
    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, f"wildrgbd_metadata_{args.split}.npz")
        print(f"Output file not specified. Using default: {args.output_file}")
    
    if not Test:
        fetch_and_save_wildrgbd_metadata(args.input_dir, args.output_file, args.split, args.num_workers)
    else:
        # read the npz file
        data = np.load(args.output_file, allow_pickle=True)
        # print the keys
        print("Scenes:", list(data.keys()))
        print("Number of scenes:", len(data.keys()))
        
        # Print some example scene info
        for scene_name in list(data.keys())[:3]:
            scene_data = data[scene_name].item()
            print(f"\nScene '{scene_name}':") 
            print(f"  Object: {scene_data['obj']}")
            print(f"  Instance: {scene_data['instance']}")
            print(f"  Images: {len(scene_data['images'])}")
            print(f"  Sample images: {scene_data['images'][:3]}")


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()