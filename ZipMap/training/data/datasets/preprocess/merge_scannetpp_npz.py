import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm


def merge_scannetpp_npz_dict(scannetpp_dir, output_file):
    # Load all scene names from all_metadata.npz
    all_metadata_path = osp.join(scannetpp_dir, "all_metadata.npz")
    with np.load(all_metadata_path) as meta:
        all_scenes = meta["scenes"]

    scene_dict = {}
    scene_names = []

    for scene in tqdm(all_scenes, desc="Merging ScanNet++"):
        scene_dir = osp.join(scannetpp_dir, scene)
        npz_path = osp.join(scene_dir, "new_scene_metadata.npz")
        imgs_on_disk = sorted(os.listdir(osp.join(scene_dir, "images"))) if osp.exists(osp.join(scene_dir, "images")) else []
        imgs_on_disk = set(map(lambda x: x[:-4], imgs_on_disk))

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
                    "imgs_on_disk": imgs_on_disk,
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
    parser.add_argument(
        "--scannetpp_dir",
        type=str,
        default="/usr/local/home/haian/Dataset/mast3r_data/processed_scannetpp",
        help="Root ScanNet++ directory",
    )
    parser.add_argument("--output", type=str, default=None, help="Output npz file path")
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = osp.join(args.scannetpp_dir, "all_scenes_dict.npz")

    merge_scannetpp_npz_dict(args.scannetpp_dir, output_file)


if __name__ == "__main__":
    main()
