import os
import os.path as osp
import numpy as np
import argparse
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def process_single_scene(scene, dl3dv_dir):
    """Process a single scene and return its metadata."""
    scene_dir = osp.join(dl3dv_dir, scene, "dense")
    
    if not osp.exists(scene_dir):
        return None, f"Warning: {scene_dir} does not exist, skipping."
        
    rgb_dir = osp.join(scene_dir, "rgb")
    cam_dir = osp.join(scene_dir, "cam")
    
    if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
        return None, f"Warning: missing rgb or cam directory in {scene_dir}, skipping."

    try:
        rgb_paths = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith(".png")
        ])
        
        if len(rgb_paths) == 0:
            return None, f"Warning: no RGB images found in {rgb_dir}, skipping."

        # Load camera parameters for all images
        intrinsics = []
        trajectories = []
        valid_images = []
        
        for rgb_path in rgb_paths:
            basename = rgb_path[:-4]
            cam_file_path = osp.join(cam_dir, basename + ".npz")
            
            if not osp.exists(cam_file_path):
                continue
                
            try:
                cam_file = np.load(cam_file_path)
                intrinsic = cam_file["intrinsic"].astype(np.float32)
                pose = cam_file["pose"].astype(np.float32)
                
                intrinsics.append(intrinsic)
                trajectories.append(pose)
                valid_images.append(rgb_path)
            except Exception as e:
                continue
        
        if len(valid_images) == 0:
            return None, f"Warning: no valid images found in {scene}, skipping."
            
        # Convert to numpy arrays
        intrinsics = np.array(intrinsics)
        trajectories = np.array(trajectories)
        valid_images = np.array(valid_images)
        
        scene_data = {
            "images": valid_images,
            "intrinsics": intrinsics,
            "trajectories": trajectories,
        }
        
        return (scene, scene_data), None
        
    except Exception as e:
        return None, f"Error processing scene {scene}: {e}"


def save_dl3dv_metadata(dl3dv_dir, output_file, num_processes=None):
    if num_processes is None:
        num_processes = min(cpu_count(), 96)  

    all_scenes = sorted(
        [f for f in os.listdir(dl3dv_dir) if os.path.isdir(osp.join(dl3dv_dir, f))]
    )
    
    subscenes = []
    for scene in all_scenes:
        subscenes.extend([
            osp.join(scene, f)
            for f in os.listdir(osp.join(dl3dv_dir, scene))
            if os.path.isdir(osp.join(dl3dv_dir, scene, f))
            and len(os.listdir(osp.join(dl3dv_dir, scene, f))) > 0
        ])

    print(f"Processing {len(subscenes)} scenes using {num_processes} processes...")
    
    # Create a partial function with dl3dv_dir pre-filled
    process_func = partial(process_single_scene, dl3dv_dir=dl3dv_dir)
    
    scene_dict = {}
    scene_names = []
    
    # Process scenes in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(
            pool.imap(process_func, subscenes),
            total=len(subscenes),
            desc="Processing DL3DV scenes"
        ))
    
    # Collect results
    for result, error_msg in results:
        if error_msg:
            print(error_msg)
        elif result:
            scene, scene_data = result
            scene_dict[scene] = scene_data
            scene_names.append(scene)

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
        "--dl3dv_dir",
        type=str,
        required=True,
        help="Root DL3DV directory",
    )
    parser.add_argument("--output", type=str, default=None, help="Output npz file path")
    parser.add_argument(
        "--num_processes", 
        type=int, 
        default=None, 
        help="Number of processes to use (default: min(cpu_count(), 16))"
    )
    args = parser.parse_args()

    output_file = args.output
    if output_file is None:
        output_file = osp.join(args.dl3dv_dir, "all_scenes_dict.npz")

    save_dl3dv_metadata(args.dl3dv_dir, output_file, args.num_processes)


if __name__ == "__main__":
    main()