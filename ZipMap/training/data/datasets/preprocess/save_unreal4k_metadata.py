import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10

def process_sequence(seq_dir_info):
    """
    Process a single sequence directory and return its metadata
    Args:
        seq_dir_info: tuple of (UNREAL4K_DIR, seq_dir)
    Returns:
        tuple of (seq_key, sequence_data) or None if sequence should be skipped
    """
    UNREAL4K_DIR, seq_dir = seq_dir_info
    
    if not osp.exists(seq_dir):
        return None
        
    # Get all PNG files (RGB images)
    try:
        all_files = os.listdir(seq_dir)
    except PermissionError:
        return None
        
    png_files = sorted([
        f for f in all_files
        if f.endswith("_rgb.png")
    ])
    
    # Extract basenames (remove "_rgb.png" suffix)
    basenames = [f[:-8] for f in png_files]  # Remove "_rgb.png"
    
    if len(basenames) < MIN_NUM_IMAGES:
        return None
    
    images = []
    c2ws = []
    intrinsics = []
    
    for basename in basenames:
        try:
            # Check if all required files exist
            rgb_path = osp.join(seq_dir, basename + "_rgb.png")
            depth_path = osp.join(seq_dir, basename + "_depth.npy")
            cam_path = osp.join(seq_dir, basename + ".npz")
            
            if not all(osp.exists(p) for p in [rgb_path, depth_path, cam_path]):
                continue
            
            # Load camera parameters
            cam_data = np.load(cam_path)
            camera_pose = cam_data["cam2world"].astype(np.float32)
            intrinsics_data = cam_data["intrinsics"].astype(np.float32)
            
            c2ws.append(camera_pose)
            intrinsics.append(intrinsics_data)
            images.append(basename)
            
        except Exception:
            continue
    
    if len(images) < MIN_NUM_IMAGES:
        return None
    
    # Use relative path from UNREAL4K_DIR as sequence key
    seq_key = osp.relpath(seq_dir, UNREAL4K_DIR)
    sequence_data = {
        "images": np.array(images),
        "c2ws": np.array(c2ws),
        "intrinsics": np.array(intrinsics),
    }
    
    return seq_key, sequence_data

def fetch_and_save_unreal4k_metadata(UNREAL4K_DIR, output_file, num_workers=None):
    """
    Process UnReal4K dataset and save all metadata in a single npz file using multiprocessing
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 32)  # Cap at 32 to avoid too many open files
    
    # Get all scene directories (top-level directories)
    scene_dirs = sorted([
        d for d in os.listdir(UNREAL4K_DIR)
        if os.path.isdir(osp.join(UNREAL4K_DIR, d))
    ])

    # Process each scene with modes "0" and "1"
    seq_dirs = sorted([
        osp.join(UNREAL4K_DIR, scene, mode)
        for scene in scene_dirs
        for mode in ["0", "1"]
        if osp.exists(osp.join(UNREAL4K_DIR, scene, mode))
    ])
    
    # Prepare arguments for multiprocessing
    seq_dir_infos = [(UNREAL4K_DIR, seq_dir) for seq_dir in seq_dirs]
    
    logging.info(f"Processing {len(seq_dir_infos)} sequences using {num_workers} workers...")
    
    # Process sequences in parallel
    sequence_dict = {}
    processed_sequences = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_sequence, seq_dir_infos),
            total=len(seq_dir_infos),
            desc="Processing sequences"
        ))
    
    # Collect results
    for result in results:
        if result is not None:
            seq_key, sequence_data = result
            sequence_dict[seq_key] = sequence_data
            processed_sequences += 1
            logging.info(f"Processed {seq_key}: {len(sequence_data['images'])} images")
    
    # Save as a single npz with dict-of-arrays
    np.savez_compressed(
        output_file,
        **sequence_dict
    )
    logging.info(f"UnReal4K processing complete: {processed_sequences} sequences from {len(sequence_dict)} total")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process UnReal4K dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default="/home/storage-bucket/haian/zipmap_data/processed_unreal4k/", help="Input UnReal4K directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: min(cpu_count, 32))")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_sequences_dict.npz")
        
    fetch_and_save_unreal4k_metadata(args.input_dir, args.output_file, args.num_workers)

if __name__ == "__main__":
    main()