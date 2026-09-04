import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
import h5py
import multiprocessing as mp
from functools import partial


# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10


def load_invalid_dict(h5_file_path):
    """Load invalid file pairs from h5 file."""
    invalid_dict = {}
    with h5py.File(h5_file_path, "r") as h5f:
        for scene in h5f:
            data = h5f[scene]["invalid_pairs"][:]
            invalid_pairs = set(
                tuple(pair.decode("utf-8").split("_")) for pair in data
            )
            invalid_dict[scene] = invalid_pairs
    return invalid_dict


def process_scene(scene_info):
    """
    Process a single scene and return its metadata.
    
    Args:
        scene_info: Tuple of (scene, Waymo_DIR, invalid_dict)
    
    Returns:
        Dict of sequence metadata for this scene
    """
    try:
        scene, Waymo_DIR, invalid_dict = scene_info
        scene_dir = osp.join(Waymo_DIR, scene)
        
        if not os.path.exists(scene_dir):
            print(f"Scene directory does not exist: {scene_dir}")
            return {}
            
        invalid_pairs = invalid_dict.get(scene, set())
        scene_results = {}
        
        # Group files by sequence ID
        seq2frames = {}
        for f in os.listdir(scene_dir):
            if not f.endswith(".jpg"):
                continue
            basename = f[:-4]
            frame_id = basename.split("_")[0]
            seq_id = basename.split("_")[1]
            if seq_id == "5":  # Skip seq_id 5 as in original
                continue
            if (seq_id, frame_id) in invalid_pairs:
                continue  # Skip invalid files
            if seq_id not in seq2frames:
                seq2frames[seq_id] = []
            seq2frames[seq_id].append(frame_id)
        
        # Process each sequence in the scene
        for seq_id, frame_ids in seq2frames.items():
            frame_ids = sorted(frame_ids)
            num_imgs = len(frame_ids)
            
            if num_imgs < MIN_NUM_IMAGES:
                print(f"Skipping {scene}_{seq_id} - insufficient images ({num_imgs})")
                continue
            
            # Load camera data for this sequence
            sequence_trajectories = []
            sequence_intrinsics = []
            valid_images = []
            
            for frame_id in frame_ids:
                try:
                    impath = f"{frame_id}_{seq_id}"
                    
                    # Check if all required files exist
                    jpg_path = osp.join(scene_dir, impath + ".jpg")
                    exr_path = osp.join(scene_dir, impath + ".exr")
                    npz_path = osp.join(scene_dir, impath + ".npz")
                    
                    if not all(osp.exists(p) for p in [jpg_path, exr_path, npz_path]):
                        print(f"Missing files for {impath} in {scene}, skipping.")
                        continue
                    
                    # Load camera parameters
                    camera_params = np.load(npz_path)
                    intrinsics = np.float32(camera_params["intrinsics"])
                    camera_pose = np.float32(camera_params["cam2world"])  # cam2world
                    
                    sequence_trajectories.append(camera_pose)
                    sequence_intrinsics.append(intrinsics)
                    valid_images.append(impath)
                    
                except Exception as e:
                    print(f"Warning: Failed to load camera data for {impath} in {scene}: {e}")
                    continue
            
            if len(valid_images) < MIN_NUM_IMAGES:
                print(f"Skipping {scene}_{seq_id} - insufficient valid images ({len(valid_images)})")
                continue
            
            # Store sequence data
            sequence_key = f"{scene}_{seq_id}"
            scene_results[sequence_key] = {
                "images": np.array(valid_images),
                "c2ws": np.array(sequence_trajectories),
                "intrinsics": np.array(sequence_intrinsics),
                "scene": scene,
                "seq_id": seq_id,
            }
            
            print(f"Processed {sequence_key}: {len(valid_images)} images")
        
        return scene_results
    
    except Exception as e:
        print(f"Error: Failed to process scene {scene}: {e}")
        return {}


def fetch_and_save_waymo_metadata(Waymo_DIR, output_file, num_processes=None):
    """
    Process Waymo dataset and save all metadata in a single npz file using multiprocessing
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if num_processes is None:
        num_processes = mp.cpu_count() * 2
    
    # Load invalid files dict
    invalid_files_path = osp.join(Waymo_DIR, "invalid_files.h5")
    if not osp.exists(invalid_files_path):
        raise FileNotFoundError(f"Invalid files not found: {invalid_files_path}")
    
    invalid_dict = load_invalid_dict(invalid_files_path)
    
    # Get all scene directories
    scene_dirs = sorted([
        d for d in os.listdir(Waymo_DIR)
        if os.path.isdir(os.path.join(Waymo_DIR, d)) and d != "invalid_files.h5"
    ])
    
    # Prepare scene info for multiprocessing
    scene_infos = [(scene, Waymo_DIR, invalid_dict) for scene in scene_dirs]
    
    # Initialize data structures for single file output
    scene_dict = {}
    processed_sequences = 0
    
    # Process scenes in parallel
    print(f"Processing {len(scene_dirs)} scenes using {num_processes} processes")
    
    with mp.Pool(processes=num_processes) as pool:
        # Use tqdm to show progress
        results = []
        for scene_result in tqdm(
            pool.imap(process_scene, scene_infos), 
            total=len(scene_infos), 
            desc="Processing Waymo scenes"
        ):
            results.append(scene_result)
    
    # Combine results from all processes
    for scene_result in results:
        scene_dict.update(scene_result)
        processed_sequences += len(scene_result)
    
    # Save as a single npz with dict
    np.savez_compressed(
        output_file,
        **scene_dict
    )

    print(f"Waymo processing complete: {processed_sequences} sequences")
    print(f"All metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process Waymo dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/storage-bucket/haian/zipmap_data/processed_waymo/",
                       help="Input Waymo directory")
    parser.add_argument("--output_file", type=str,
                       help="Output npz file for all processed metadata")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use (default: use all CPU cores)")
    
    args = parser.parse_args()
    Test = False
    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "waymo_metadata.npz")
    if not Test:
        fetch_and_save_waymo_metadata(args.input_dir, args.output_file, args.num_processes)
    else:
        # read the npz file
        data = np.load(args.output_file, allow_pickle=True)
        # print the keys
        print("Sequences:", list(data.keys()))
        print("Number of sequences:", len(data.keys()))
        
        # Print some example sequence info
        for seq_name in list(data.keys())[:3]:
            seq_data = data[seq_name].item()
            print(f"\nSequence '{seq_name}':")
            print(f"  Images: {len(seq_data['images'])}")
            print(f"  Scene: {seq_data['scene']}")
            print(f"  Seq ID: {seq_data['seq_id']}")
            print(f"  Sample images: {seq_data['images'][:3]}")


if __name__ == "__main__":
    # Required for multiprocessing on Windows/Mac
    mp.set_start_method('spawn', force=True)
    logging.basicConfig(level=logging.INFO)
    main()