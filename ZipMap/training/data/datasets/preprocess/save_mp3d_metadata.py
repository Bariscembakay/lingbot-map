import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per scene
MIN_NUM_IMAGES = 10

def process_scene(scene_info):
    """
    Process a single scene directory and return its metadata
    Args:
        scene_info: tuple of (MP3D_DIR, scene_name)
    Returns:
        tuple of (scene_name, scene_data) or None if scene should be skipped
    """
    MP3D_DIR, scene = scene_info
    
    scene_dir = osp.join(MP3D_DIR, scene)
    rgb_dir = osp.join(scene_dir, "rgb")
    cam_dir = osp.join(scene_dir, "cam")
    overlap_file = osp.join(scene_dir, "overlap.npy")
    
    # Check if required directories and files exist
    if not all(osp.exists(p) for p in [rgb_dir, cam_dir, overlap_file]):
        return None
    
    try:
        # Get all PNG RGB files
        rgb_files = [f for f in os.listdir(rgb_dir) if f.endswith(".png")]
        basenames = sorted([f[:-4] for f in rgb_files])
        
        if len(basenames) < MIN_NUM_IMAGES:
            return None
        
        # Load overlap matrix
        overlap = np.load(overlap_file)
        
        images = []
        c2ws = []
        intrinsics = []
        
        for basename in basenames:
            try:
                # Check if all required files exist
                rgb_path = osp.join(rgb_dir, basename + ".png")
                cam_path = osp.join(cam_dir, basename + ".npz")
                
                if not all(osp.exists(p) for p in [rgb_path, cam_path]):
                    continue
                
                # Load camera parameters
                cam_data = np.load(cam_path)
                camera_pose = cam_data["pose"].astype(np.float32)
                intrinsics_data = cam_data["intrinsics"].astype(np.float32)
                
                c2ws.append(camera_pose)
                intrinsics.append(intrinsics_data)
                images.append(basename)
                
            except Exception:
                continue
        
        if len(images) < MIN_NUM_IMAGES:
            return None
        
        scene_data = {
            "images": np.array(images),
            "c2ws": np.array(c2ws),
            "intrinsics": np.array(intrinsics),
            "overlap": overlap,  # MP3D specific: include overlap matrix
        }
        
        return scene, scene_data
        
    except Exception:
        return None

def fetch_and_save_mp3d_metadata(MP3D_DIR, output_file, num_workers=None):
    """
    Process MP3D dataset and save all metadata in a single npz file using multiprocessing
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    if num_workers is None:
        num_workers = min(cpu_count(), 96)  # Cap at 32 to avoid too many open files
    
    # Get all scene directories
    all_scenes = sorted([
        d for d in os.listdir(MP3D_DIR)
        if os.path.isdir(osp.join(MP3D_DIR, d))
    ])
    
    # Prepare arguments for multiprocessing
    scene_infos = [(MP3D_DIR, scene) for scene in all_scenes]
    
    logging.info(f"Processing {len(scene_infos)} scenes using {num_workers} workers...")
    
    # Process scenes in parallel
    scene_dict = {}
    processed_scenes = 0
    
    with Pool(processes=num_workers) as pool:
        # Use imap for progress tracking
        results = list(tqdm(
            pool.imap(process_scene, scene_infos),
            total=len(scene_infos),
            desc="Processing scenes"
        ))
    
    # Collect results
    for result in results:
        if result is not None:
            scene_name, scene_data = result
            scene_dict[scene_name] = scene_data
            processed_scenes += 1
            logging.info(f"Processed {scene_name}: {len(scene_data['images'])} images")
    
    # Save as a single npz with dict-of-arrays
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    logging.info(f"MP3D processing complete: {processed_scenes} scenes from {len(scene_dict)} total")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process MP3D dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default="/home/storage-bucket/haian/zipmap_data/processed_mp3d/", help="Input MP3D directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: min(cpu_count, 32))")
    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_scenes_dict.npz")
        
    fetch_and_save_mp3d_metadata(args.input_dir, args.output_file, args.num_workers)

if __name__ == "__main__":
    main()