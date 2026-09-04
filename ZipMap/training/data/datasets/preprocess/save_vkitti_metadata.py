import os
import os.path as osp
import numpy as np
import argparse

def fetch_and_save_vkitti_metadata(VKitti_DIR, output_file):
    """
    Process VKitti dataset and save sequence_list and extrinsics_dict in a single npz file.
    """
    sequence_list = []
    extrinsics_dict = {}
    intrinsics_dict = {}

    for scene in sorted(os.listdir(VKitti_DIR)):
        scene_path = osp.join(VKitti_DIR, scene)
        if not osp.isdir(scene_path):
            continue
        for variant in os.listdir(scene_path):
            variant_path = osp.join(scene_path, variant)
            if not osp.isdir(variant_path):
                continue
            rgb_dir = osp.join(variant_path, "frames", "rgb")
            if not osp.isdir(rgb_dir):
                continue
            for cam in os.listdir(rgb_dir):
                cam_rgb_dir = osp.join(rgb_dir, cam)
                if not osp.isdir(cam_rgb_dir):
                    continue
                rel_seq = osp.relpath(cam_rgb_dir, VKitti_DIR)
                sequence_list.append(rel_seq)
                extrinsic_path = osp.join(variant_path, "extrinsic.txt")
                intrinsic_path = osp.join(variant_path, "intrinsic.txt")
                cam_id = int(cam.split("_")[-1])
                # Extrinsics
                if osp.exists(extrinsic_path):
                    try:
                        extrinsics = np.loadtxt(extrinsic_path, delimiter=" ", skiprows=1)
                        cam_extrinsics = extrinsics[extrinsics[:, 1] == cam_id][:, 2:].reshape(-1, 16)
                        extrinsics_dict[rel_seq] = cam_extrinsics
                    except Exception as e:
                        print(f"Failed to load extrinsics for {rel_seq}: {e}")
                        extrinsics_dict[rel_seq] = None
                else:
                    extrinsics_dict[rel_seq] = None
                # Intrinsics
                if osp.exists(intrinsic_path):
                    try:
                        intrinsics = np.loadtxt(intrinsic_path, delimiter=" ", skiprows=1)
                        cam_intrinsics = intrinsics[intrinsics[:, 1] == cam_id][:, 2:].reshape(-1, 4)
                        intrinsics_dict[rel_seq] = cam_intrinsics
                    except Exception as e:
                        print(f"Failed to load intrinsics for {rel_seq}: {e}")
                        intrinsics_dict[rel_seq] = None
                else:
                    intrinsics_dict[rel_seq] = None

    np.savez_compressed(output_file, sequence_list=np.array(sequence_list), extrinsics_dict=extrinsics_dict, intrinsics_dict=intrinsics_dict)
    print(f"Saved {len(sequence_list)} sequences to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process VKitti dataset and save minimal metadata")
    parser.add_argument("--input_dir", type=str, required=True, help="Input VKitti directory")
    parser.add_argument("--output_file", type=str, default="./vkitti_metadata.npz", help="Output npz file for all processed metadata")
    args = parser.parse_args()
    fetch_and_save_vkitti_metadata(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
