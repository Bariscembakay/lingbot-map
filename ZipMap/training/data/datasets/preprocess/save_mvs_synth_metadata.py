'''
python save_mvs_synth_metadata.py --input_dir local_data/processed_mvs_synth \
                       --output_file local_data/processed_mvs_synth/all_scenes_dict.npz


'''


import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per scene
MIN_NUM_IMAGES = 10

def fetch_and_save_mvs_synth_metadata(MVS_SYNTH_DIR, output_file):
    """
    Process MVS-Synth dataset and save all metadata in a single npz file
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Get all scenes (top-level directories)
    all_scenes = sorted([
        f for f in os.listdir(MVS_SYNTH_DIR)
        if os.path.isdir(osp.join(MVS_SYNTH_DIR, f))
    ])

    # Initialize data structures for single file output
    scene_dict = {}
    processed_scenes = 0
    
    for scene in tqdm(all_scenes, desc="Processing scenes"):
        scene_dir = osp.join(MVS_SYNTH_DIR, scene)
        rgb_dir = osp.join(scene_dir, "rgb")
        cam_dir = osp.join(scene_dir, "cam")
        if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
            logging.debug(f"Skipping {scene}: missing rgb or cam directory")
            continue
        
        # Get all RGB image basenames (without extension)
        rgb_files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith(".jpg") or f.endswith(".png")
        ])
        basenames = [osp.splitext(f)[0] for f in rgb_files]
        if len(basenames) < MIN_NUM_IMAGES:
            logging.debug(f"Skipping {scene} - insufficient images ({len(basenames)})")
            continue
        
        images = []
        c2ws = []
        intrinsics = []
        for basename in basenames:
            cam_path = osp.join(cam_dir, basename + ".npz")
            if not osp.exists(cam_path):
                continue
            try:
                cam = np.load(cam_path)
                c2ws.append(cam["pose"].astype(np.float32))
                intrinsics.append(cam["intrinsics"].astype(np.float32))
                images.append(basename)
            except Exception as e:
                logging.warning(f"Failed to load camera data for {basename}: {e}")
                continue
        if len(images) < MIN_NUM_IMAGES:
            logging.debug(f"Skipping {scene} - insufficient valid images ({len(images)})")
            continue
        scene_dict[scene] = {
            "images": np.array(images),
            "c2ws": np.array(c2ws),
            "intrinsics": np.array(intrinsics),
        }
        processed_scenes += 1
        logging.info(f"Processed {scene}: {len(images)} images")
    # Save as a single npz with dict-of-arrays
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    logging.info(f"MVS-Synth processing complete: {processed_scenes} scenes from {len(scene_dict)} total")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process MVS-Synth dataset and save metadata")
    parser.add_argument("--input_dir", type=str, required=True, help="Input MVS-Synth directory")
    parser.add_argument("--output_file", type=str, default="./all_scenes_dict.npz", help="Output npz file for all processed metadata")
    args = parser.parse_args()
    fetch_and_save_mvs_synth_metadata(args.input_dir, args.output_file)

if __name__ == "__main__":
    main()
