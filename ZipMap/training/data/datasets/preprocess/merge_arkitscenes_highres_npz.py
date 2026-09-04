import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm


def merge_arkitscenes_highres_npz_dict(arkitscenes_dir, split_folder, output_file):
    split_dir = osp.join(arkitscenes_dir, split_folder)
    
    # Get all scene directories
    all_scenes = sorted([
        d for d in os.listdir(split_dir)
        if osp.isdir(osp.join(split_dir, d))
    ])

    scene_dict = {}
    scene_names = []

    # all_scenes = all_scenes[:50]  # Uncomment for testing

    for scene in tqdm(all_scenes, desc=f"Merging {split_folder}"):
        scene_dir = osp.join(split_dir, scene)
        npz_path = osp.join(scene_dir, "scene_metadata.npz")
        if not osp.exists(npz_path):
            print(f"Warning: {npz_path} does not exist, skipping.")
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                # Sort images by name to maintain consistent ordering
                imgs_with_indices = sorted(
                    enumerate(data["images"]), key=lambda x: x[1]
                )
                imgs = [x[1] for x in imgs_with_indices]
                indices = [x[0] for x in imgs_with_indices]
                
                # Get corresponding intrinsics and trajectories in the same order
                intrins = data["intrinsics"][indices]
                traj = data["trajectories"][indices]
                
                scene_dict[scene] = {
                    "images": np.array(imgs),
                    "intrinsics": intrins,
                    "trajectories": traj,
                }
                scene_names.append(scene)
        except Exception as e:
            print(f"Error loading {npz_path}: {e}")

    # Save as a single npz with dict-of-dicts
    np.savez_compressed(
        output_file,
        **scene_dict,
        scene_names=np.array(scene_names)
    )
    print(f"Saved merged file to {output_file}")


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--arkitscenes_dir", type=str, default="/usr/local/home/storage-bucket/haian/zipmap_data/processed_arkitscenes_highres", help="Root ARKitScenes high-res directory")
    parser.add_argument("--split", type=str, choices=["Training", "Validation"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output npz file path")
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = osp.join(args.arkitscenes_dir, args.split, "all_scenes_dict.npz")

    merge_arkitscenes_highres_npz_dict(args.arkitscenes_dir, args.split, output_file)

if __name__ == "__main__":
    main()