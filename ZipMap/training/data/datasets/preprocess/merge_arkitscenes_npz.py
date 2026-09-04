import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm


def merge_arkitscenes_npz_dict(arkitscenes_dir, split_folder, output_file):
    split_dir = osp.join(arkitscenes_dir, split_folder)
    # Load all scene names from all_metadata.npz
    all_metadata_path = osp.join(split_dir, "all_metadata.npz")
    with np.load(all_metadata_path) as meta:
        all_scenes = meta["scenes"]

    scene_dict = {}
    scene_names = []

    # all_scenes = all_scenes[:50]

    for scene in tqdm(all_scenes, desc=f"Merging {split_folder}"):
        scene_dir = osp.join(split_dir, scene)
        npz_path = osp.join(scene_dir, "new_scene_metadata.npz")
        if not osp.exists(npz_path):
            print(f"Warning: {npz_path} does not exist, skipping.")
            continue
        try:
            with np.load(npz_path, allow_pickle=True) as data:
                imgs = data["images"]
                intrins = data["intrinsics"]
                traj = data["trajectories"]
                collections = data["image_collection"].item()
                scene_dict[scene] = {
                    "images": imgs,
                    "intrinsics": intrins,
                    "trajectories": traj,
                    "image_collection": collections,
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
    parser.add_argument("--arkitscenes_dir", type=str, default="/usr/local/home/storage-bucket/haian/zipmap_data/processed_arkitscenes", help="Root ARKitScenes directory")
    parser.add_argument("--split", type=str, choices=["Training", "Test"], required=True)
    parser.add_argument("--output", type=str, default=None, help="Output npz file path")
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = osp.join(args.arkitscenes_dir, args.split, "all_scenes_dict.npz")

    merge_arkitscenes_npz_dict(args.arkitscenes_dir, args.split, output_file)

if __name__ == "__main__":
    main() 