import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count


def load_camera_data(img_basename, cam_dir):
    """Load camera pose and intrinsics for a single image"""
    cam_file = osp.join(cam_dir, img_basename + ".npz")
    if osp.exists(cam_file):
        cam_data = np.load(cam_file)
        return cam_data["pose"], cam_data["intrinsics"]
    else:
        print(f"!!!!!! Warning: Camera file {cam_file} not found")
        return None, None


def process_scene(scene_info):
    """Process a single scene to load all its data"""
    scene, split_dir = scene_info
    scene_dir = osp.join(split_dir, scene)
    npz_path = osp.join(scene_dir, "new_scene_metadata.npz")
    
    if not osp.exists(npz_path):
        print(f"Warning: {npz_path} does not exist, skipping.")
        return None
    
    try:
        with np.load(npz_path, allow_pickle=True) as data:
            images = data["images"]
            
        # Load camera poses and intrinsics for each image sequentially
        cam_dir = osp.join(scene_dir, "cam")
        
        poses = []
        intrinsics = []
        valid_indices = []
        
        for i, img_basename in enumerate(images):
            pose, intrinsic = load_camera_data(img_basename, cam_dir)
            if pose is not None and intrinsic is not None:
                poses.append(pose)
                intrinsics.append(intrinsic)
                valid_indices.append(i)
        
        # Filter images to only include those with valid camera data
        if len(valid_indices) != len(images):
            print(f"Warning: Scene {scene} has {len(images)} images but only {len(valid_indices)} valid camera files")
            images = [images[i] for i in valid_indices]
        
        if len(poses) == 0:
            print(f"Warning: No valid camera data found for scene {scene}, skipping.")
            return None
            
        return scene, {
            "images": np.array(images),
            "poses": np.array(poses),
            "intrinsics": np.array(intrinsics),
        }
        
    except Exception as e:
        print(f"Error loading {npz_path}: {e}")
        return None


def merge_scannet_npz_dict(scannet_dir, split_folder, output_file, num_workers=None):
    split_dir = osp.join(scannet_dir, split_folder)
    
    # Get all scene directories that start with "scene"
    scenes = [scene for scene in os.listdir(split_dir) if scene.startswith("scene")]
    
    if num_workers is None:
        num_workers = min(cpu_count()*2, len(scenes))
    
    print(f"Processing {len(scenes)} scenes using {num_workers} workers...")
    
    # Prepare scene info for multiprocessing
    scene_infos = [(scene, split_dir) for scene in scenes]
    
    scene_dict = {}
    scene_names = []
    
    # Process scenes in parallel
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(process_scene, scene_infos), 
            total=len(scene_infos), 
            desc=f"Merging {split_folder}"
        ))
    
    # Collect valid results
    for result in results:
        if result is not None:
            scene_name, scene_data = result
            scene_dict[scene_name] = scene_data
            scene_names.append(scene_name)
    
    print(f"Successfully processed {len(scene_names)} out of {len(scenes)} scenes")

    # Save as a single npz with dict-of-dicts
    np.savez_compressed(
        output_file,
        **scene_dict,
        scene_names=np.array(scene_names)
    )
    print(f"Saved merged file to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--scannet_dir", type=str, default='/home/storage-bucket/haian/zipmap_data/processed_scannet', help="Root ScanNet directory")
    parser.add_argument("--split", type=str, choices=["scans_train", "scans_test"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output npz file path")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: auto)")
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = osp.join(args.scannet_dir, args.split, "all_scenes_dict.npz")

    merge_scannet_npz_dict(args.scannet_dir, args.split, output_file, args.num_workers)

if __name__ == "__main__":
    main()