import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
import h5py


# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10


def build_adjacency_list(S, thresh=0.2):
    """Build adjacency list from score matrix, following cut3r logic."""
    adjacency_list = [[] for _ in range(len(S))]
    S = S - thresh
    S[S < 0] = 0
    rows, cols = np.nonzero(S)
    for i, j in zip(rows, cols):
        adjacency_list[i].append((j, S[i][j]))
    return adjacency_list


def fetch_and_save_blendedmvs_metadata(BlendedMVS_DIR, output_file):
    """
    Process BlendedMVS dataset and save all metadata in a single npz file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Load h5 file with overlap information first
    h5_file_path = osp.join(BlendedMVS_DIR, "new_overlap.h5")
    if not osp.exists(h5_file_path):
        raise FileNotFoundError(f"Overlap file not found: {h5_file_path}")
    
    # Initialize data structures for single file output
    scene_dict = {}
    processed_sequences = 0
    
    with h5py.File(h5_file_path, "r") as f:
        for scene_dir in tqdm(f.keys(), desc="Processing BlendedMVS scenes"):
            group = f[scene_dir]
            basenames = group["basenames"][:]
            indices = group["indices"][:]
            values = group["values"][:]
            shape = group.attrs["shape"]
            
            # Check if scene directory actually exists
            scene_path = osp.join(BlendedMVS_DIR, scene_dir)
            if not osp.exists(scene_path):
                logging.warning(f"Scene directory not found: {scene_path}")
                continue
            
            # Reconstruct the sparse score matrix
            score_matrix = np.zeros(shape, dtype=np.float32)
            score_matrix[indices[0], indices[1]] = values
            
            # Load camera data for this scene
            sequence_trajectories = []
            sequence_intrinsics = []
            valid_images = []
            valid_basenames = []
            
            for idx, basename in enumerate(basenames):
                try:
                    # Convert bytes to string if necessary
                    if isinstance(basename, bytes):
                        basename = basename.decode('utf-8')
                    
                    # Check if image files exist
                    jpg_path = osp.join(scene_path, basename + ".jpg")
                    exr_path = osp.join(scene_path, basename + ".exr")
                    cam_path = osp.join(scene_path, basename + ".npz")
                    
                    if not all(osp.exists(p) for p in [jpg_path, exr_path, cam_path]):
                        print(f"Missing files for {basename} in {scene_dir}, skipping.")
                        continue
                    
                    # Load camera parameters
                    camera_params = np.load(cam_path)
                    intrinsics = np.float32(camera_params["intrinsics"])
                    
                    # Build camera pose (cam2world)
                    camera_pose = np.eye(4, dtype=np.float32)
                    camera_pose[:3, :3] = camera_params["R_cam2world"]
                    camera_pose[:3, 3] = camera_params["t_cam2world"]
                    
                    sequence_trajectories.append(camera_pose)
                    sequence_intrinsics.append(intrinsics)
                    valid_images.append(basename)
                    valid_basenames.append(idx)  # Store original index for score matrix
                    
                except Exception as e:
                    logging.warning(f"Failed to load camera data for {basename} in {scene_dir}: {e}")
                    continue
            
            if len(valid_images) < MIN_NUM_IMAGES:
                logging.debug(f"Skipping {scene_dir} - insufficient valid images ({len(valid_images)})")
                continue
            
            # Filter score matrix to only include valid images
            valid_indices = np.array(valid_basenames)
            
            # Build adjacency list following cut3r logic
            adjacency_list = build_adjacency_list(score_matrix, thresh=0.2)
            
            # Store scene data
            scene_dict[scene_dir] = {
                "images": np.array(valid_images),
                "c2ws": np.array(sequence_trajectories),
                "intrinsics": np.array(sequence_intrinsics),
                "score_matrix": adjacency_list,  # Store adjacency list instead of raw matrix
            }
            
            processed_sequences += 1
            logging.info(f"Processed {scene_dir}: {len(valid_images)} images")
    
    # Save as a single npz with dict
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    
    logging.info(f"BlendedMVS processing complete: {processed_sequences} scenes")
    logging.info(f"All metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process BlendedMVS dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/haian/Dataset/blendedmvs/",
                       help="Input BlendedMVS directory")
    parser.add_argument("--output_file", type=str,
                       default="./blendedmvs_metadata.npz",
                       help="Output npz file for all processed metadata")
    
    args = parser.parse_args()
    Test = False
    
    if not Test:
        fetch_and_save_blendedmvs_metadata(args.input_dir, args.output_file)
    else:
        # read the npz file
        data = np.load(args.output_file, allow_pickle=True)
        # print the keys
        print("Scenes:", list(data.keys()))
        print("Number of scenes:", len(data.keys()))
        
        # Print some example scene info
        for i, scene_name in enumerate(list(data.keys())[:3]):
            scene_data = data[scene_name].item()
            adjacency_list = scene_data['score_matrix']
            print(f"\nScene '{scene_name}':")
            print(f"  Images: {len(scene_data['images'])}")
            print(f"  Adjacency list length: {len(adjacency_list)}")
            if len(adjacency_list) > 0:
                avg_neighbors = sum(len(adj) for adj in adjacency_list) / len(adjacency_list)
                print(f"  Average neighbors per image: {avg_neighbors:.1f}")
            print(f"  Sample images: {scene_data['images'][:3]}")
        


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    main()