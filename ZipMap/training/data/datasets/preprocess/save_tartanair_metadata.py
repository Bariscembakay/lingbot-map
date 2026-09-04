import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10

SELECTED_SCENE_NAMES = [
    "abandonedfactory", 
    "amusement", 
    "endofworld", 
    "hospital", 
    "neighborhood", 
    "office", 
    "oldtown", 
    "seasonsforest", 
    "soulcity",
    "abandonedfactory_night", 
    "carwelding", 
    "gascola", 
    "japanesealley", 
    "ocean", 
    "office2", 
    "seasidetown", 
    "seasonsforest_winter", 
    "westerndesert"
]


def fetch_and_save_tartanair_metadata(TartanAir_DIR, output_file):
    """
    Process TartanAir dataset and save all metadata in a single npz file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # abandonedfactory, neighborhood, etc
    scene_dirs = sorted(SELECTED_SCENE_NAMES)

    # Initialize data structures for single file output
    scene_dict = {}
    scene_names = []
    processed_sequences = 0
    
    
    for scene in tqdm(scene_dirs, desc="Processing scenes"):
        for mode in ["Easy", "Hard"]:
            mode_dir = os.path.join(TartanAir_DIR, scene, mode)
            if not os.path.isdir(mode_dir):
                continue
                
            seq_dirs = sorted([
                os.path.join(mode_dir, d)
                for d in os.listdir(mode_dir)
                if os.path.isdir(os.path.join(mode_dir, d))
            ])
            
            for seq_dir in seq_dirs:
                # Create unique scene identifier
                seq_name = f"{osp.basename(seq_dir)}"
                
                # Get all PNG files and extract basenames
                png_files = [f for f in os.listdir(seq_dir) if f.endswith("_rgb.png")]
                basenames = sorted([f[:-8] for f in png_files])  # Remove "_rgb.png"
                
                if len(basenames) == 0:
                    logging.debug(f"No RGB images found in {seq_dir}")
                    continue
                
                # Load camera data for this sequence
                sequence_trajectories = []
                valid_images = []
                
                for basename in basenames:
                    try:
                        # Load camera parameters
                        cam_file = osp.join(seq_dir, basename + "_cam.npz")
                        if not osp.exists(cam_file):
                            continue
                        camera_params = np.load(cam_file)
                        camera_pose = camera_params["camera_pose"]  # cam2world
                        
                        sequence_trajectories.append(camera_pose)
                        valid_images.append(basename + "_rgb.png")  # Keep full filename
                    except Exception as e:
                        logging.warning(f"Failed to load camera data for {basename}: {e}")
                        continue
                
                # if len(valid_images) < MIN_NUM_IMAGES:
                #     print(f"Skipping {seq_dir} - insufficient valid images ({len(valid_images)})")
                #     continue
                
                
                # Store scene data in dictionary (same format as merge_arkitscenes_npz.py)
                # Initialize nested dictionaries if they don't exist
                if scene not in scene_dict:
                    scene_dict[scene] = {}
                if mode not in scene_dict[scene]:
                    scene_dict[scene][mode] = {}
                    
                scene_dict[scene][mode][seq_name] = {
                    "images": np.array(valid_images),
                    "c2ws": np.array(sequence_trajectories),
                }
                
                scene_names.append(seq_name)
                processed_sequences += 1
                logging.info(f"Processed {seq_name}: {len(valid_images)} images")
    
    # Save as a single npz with dict-of-dicts (same as merge_arkitscenes_npz.py)
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    
    logging.info(f"TartanAir processing complete: {processed_sequences} sequences from {len(scene_names)} scenes")
    logging.info(f"All metadata saved to {output_file}")


def main():
    parser = argparse.ArgumentParser(description="Process TartanAir dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/usr/local/home/haian/Dataset/mast3r_data/processed_tartanair",
                       help="Input TartanAir directory")
    parser.add_argument("--output_file", type=str,
                       default="./all_scenes_dict.npz",
                       help="Output npz file for all processed metadata")
    
    args = parser.parse_args()
    Test = False
    if not Test:
        fetch_and_save_tartanair_metadata(args.input_dir, args.output_file)
    else:
        # read the npz file
        data = np.load(args.output_file, allow_pickle=True)
        # print the keys
        print(data.keys())
        print(len(data.keys()))
        import ipdb; ipdb.set_trace()



if __name__ == "__main__":
    main()

    