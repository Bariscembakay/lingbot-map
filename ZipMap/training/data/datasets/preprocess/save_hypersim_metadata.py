import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse


# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10


def fetch_and_save_hypersim_metadata(HyperSim_DIR, output_file):
    """
    Process HyperSim dataset and save all metadata in a single npz file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all scenes (top-level directories)
    all_scenes = sorted([
        f for f in os.listdir(HyperSim_DIR) 
        if os.path.isdir(osp.join(HyperSim_DIR, f))
    ])

    # Initialize data structures for single file output
    scene_dict = {}
    processed_sequences = 0
    
    for scene in tqdm(all_scenes, desc="Processing scenes"):
        scene_dir = osp.join(HyperSim_DIR, scene)
        
        # Get all subscenes (subdirectories within each scene)
        subscenes = []
        for f in os.listdir(scene_dir):
            subscene_dir = osp.join(scene_dir, f)
            if os.path.isdir(subscene_dir) and len(os.listdir(subscene_dir)) > 0:
                subscenes.append(f)
        
        if not subscenes:
            logging.debug(f"No valid subscenes found in {scene}")
            continue
            
        # Initialize nested dictionary for this scene
        scene_dict[scene] = {}
        
        for subscene in subscenes:
            subscene_dir = osp.join(scene_dir, subscene)
            
            # Get all RGB PNG files
            rgb_paths = sorted([
                f for f in os.listdir(subscene_dir) 
                if f.endswith("rgb.png")
            ])
            
            if len(rgb_paths) == 0:
                logging.debug(f"No RGB images found in {subscene_dir}")
                continue
            
            # Load camera data for this subscene
            sequence_trajectories = []
            sequence_intrinsics = []
            valid_images = []
            
            for rgb_path in rgb_paths:
                try:
                    # Load camera parameters
                    cam_path = rgb_path.replace("rgb.png", "cam.npz")
                    cam_file_path = osp.join(subscene_dir, cam_path)
                    
                    if not osp.exists(cam_file_path):
                        continue
                        
                    cam_file = np.load(cam_file_path)
                    intrinsics = cam_file["intrinsics"].astype(np.float32)
                    camera_pose = cam_file["pose"].astype(np.float32)  # cam2world
                    
                    sequence_trajectories.append(camera_pose)
                    sequence_intrinsics.append(intrinsics)
                    valid_images.append(rgb_path)
                    
                except Exception as e:
                    logging.warning(f"Failed to load camera data for {rgb_path}: {e}")
                    continue
            
            if len(valid_images) < MIN_NUM_IMAGES:
                logging.debug(f"Skipping {subscene_dir} - insufficient valid images ({len(valid_images)})")
                continue
            
            # Store subscene data
            scene_dict[scene][subscene] = {
                "images": np.array(valid_images),
                "c2ws": np.array(sequence_trajectories),
                "intrinsics": np.array(sequence_intrinsics),
            }
            
            processed_sequences += 1
            logging.info(f"Processed {scene}/{subscene}: {len(valid_images)} images")
    
    # Save as a single npz with dict-of-dicts
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    
    logging.info(f"HyperSim processing complete: {processed_sequences} subscenes from {len(scene_dict)} scenes")
    logging.info(f"All metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process HyperSim dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/usr/local/home/haian/Dataset/hypersim",
                       help="Input HyperSim directory")
    parser.add_argument("--output_file", type=str,
                       default="./all_scenes_dict.npz",
                       help="Output npz file for all processed metadata")
    
    args = parser.parse_args()
    Test = False
    
    if not Test:
        fetch_and_save_hypersim_metadata(args.input_dir, args.output_file)
    else:
        # read the npz file
        data = np.load(args.output_file, allow_pickle=True)
        # print the keys
        print("Top-level scenes:", list(data.keys()))
        print("Number of scenes:", len(data.keys()))
        
        # Print some example subscenes
        for i, scene_name in enumerate(list(data.keys())[:3]):
            scene_data = data[scene_name].item()
            print(f"\nScene '{scene_name}' has subscenes: {list(scene_data.keys())}")
            
            # Show info for first subscene
            if scene_data:
                first_subscene = list(scene_data.keys())[0]
                subscene_data = scene_data[first_subscene]
                print(f"  Subscene '{first_subscene}': {len(subscene_data['images'])} images")
        
        import ipdb; ipdb.set_trace()


if __name__ == "__main__":
    main()
