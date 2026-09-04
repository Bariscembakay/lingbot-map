import os
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
import h5py
from PIL import Image
from pyquaternion import Quaternion
from scipy.spatial.transform import Rotation
logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10

def open_float16(image_path):
    """Load float16 depth image from MidAir dataset"""
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    return img




# # Define body-to-camera transformation based on MidAir documentation
# body_to_camera = np.array([
#     [0, 1, 0],
#     [0, 0, -1],
#     [1, 0, 0]
# ])
# body_to_camera_translation = np.array([0.0, 0.0, 0.5])
# body_to_camera_translation = np.array([0.0, 0.0, 0.0])  # Adjusted based on observed data
# def compute_c2w_for_left_cam(body_position, body_rotation_matrix):
#     """
#     Compute camera pose (c2w) from body frame pose.
    
#     Args:
#         body_position: 3D position of body frame in world coordinates
#         body_rotation_matrix: 3x3 rotation matrix from body frame to world frame
    
#     Returns:
#         4x4 camera-to-world transformation matrix
#     """
#     # Transform rotation to camera frame
#     camera_rotation_in_world = body_rotation_matrix @ np.array([
#     [0, 0, 1],
#     [1, 0, 0],
#     [0, 1, 0]
# ])
#     # Transform position to camera frame
#     camera_location_in_body = body_to_camera_translation
#     camera_location_in_world = body_rotation_matrix @ camera_location_in_body + body_position
    
#     # Construct c2w matrix
#     c2w = np.eye(4, dtype=np.float32)
#     c2w[:3, :3] = camera_rotation_in_world
#     c2w[:3, 3] = camera_location_in_world

#     return c2w
    



def compute_c2w_for_left_cam(body_position, body_rotation_matrix):
    """
    Compute camera pose (c2w) from body frame pose.
    
    Args:
        body_position: 3D position of body frame in world coordinates
        body_rotation_matrix: 3x3 rotation matrix from body frame to world frame
    
    Returns:
        4x4 camera-to-world transformation matrix
    """
    # Transform rotation to camera frame
    camera_rotation_in_world = body_rotation_matrix @ np.array([
    [0, 0, 1],
    [1, 0, 0],
    [0, 1, 0]
    ])

    # Construct c2w matrix
    c2w = np.eye(4, dtype=np.float32)
    c2w[:3, :3] = camera_rotation_in_world
    c2w[:3, 3] = body_position

    return c2w
    


def process_sequence(args):
    """
    Process a single sequence and return its data
    """
    sequence_path, MidAir_DIR = args
    sequence_data = {}
    
    # Parse sequence path to get scene and condition info
    # e.g., /path/to/Kite_training/cloudy -> scene=Kite_training, condition=cloudy
    parts = sequence_path.replace(MidAir_DIR, '').strip('/').split('/')
    if len(parts) != 2:
        logging.warning(f"Unexpected path structure: {sequence_path}")
        return {}, [], 0
    
    scene, condition = parts
    
    # Load HDF5 file
    hdf5_path = os.path.join(sequence_path, "sensor_records.hdf5")
    if not os.path.exists(hdf5_path):
        logging.warning(f"No HDF5 file found at {hdf5_path}")
        return {}, [], 0
    
    with h5py.File(hdf5_path, "r") as database:
        # Get all trajectory datasets - there can be multiple trajectories per sequence
        trajectory_keys = list(database.keys())
        logging.debug(f"Found {len(trajectory_keys)} trajectories in {hdf5_path}")
        
        total_processed = 0
        all_sequence_names = []
        
        for traj_key in trajectory_keys:
            traj_data = database[traj_key]
            
            # Extract camera data paths and poses
            color_left_paths = [path.decode('utf-8') if isinstance(path, bytes) else path 
                                for path in traj_data["camera_data"]["color_left"]]
            depth_paths = [path.decode('utf-8') if isinstance(path, bytes) else path 
                            for path in traj_data["camera_data"]["depth"]]
            
            # Extract poses - position and orientation
            positions = traj_data["groundtruth"]["position"][:]  # Shape: (N, 3)
           
            orientations = traj_data["groundtruth"]["attitude"][:]  # Shape: (N, 4) quaternions
            
            valid_images = []
            sequence_trajectories = []
            
            for i, (rgb_path, depth_path) in enumerate(zip(color_left_paths, depth_paths)):
                # Check if image files exist - paths are relative to trajectory directories
                # e.g., color_left/trajectory_3000/00000000.png
                rgb_full_path = os.path.join(os.path.dirname(hdf5_path), rgb_path)
                depth_full_path = os.path.join(os.path.dirname(hdf5_path), depth_path)
                
                if not os.path.exists(rgb_full_path) or not os.path.exists(depth_full_path):
                    continue
                
                # Get corresponding pose (camera data is at 4x lower frequency)
                pose_idx = i * 4
                if pose_idx >= len(positions):
                    continue
                
                position = positions[pose_idx]
                orientation = orientations[pose_idx]  # quaternion [qw, qx, qy, qz]
                
                # Convert quaternion to rotation matrix using scipy
                # MidAir format: [qw, qx, qy, qz] -> scipy expects [qx, qy, qz, qw]
                quat_scipy = [orientation[1], orientation[2], orientation[3], orientation[0]]
                R_body_to_world = Rotation.from_quat(quat_scipy).as_matrix()
                
                # Compute camera pose from body pose using proper transformation
                camera_pose = compute_c2w_for_left_cam(position, R_body_to_world)
                
                sequence_trajectories.append(camera_pose)
                # Store the full relative path for later use
                valid_images.append(rgb_path)
                
    
            
            if len(valid_images) < MIN_NUM_IMAGES:
                logging.debug(f"Skipping {sequence_path}/{traj_key} - insufficient valid images ({len(valid_images)})")
                continue
            
            # Create unique sequence name
            seq_name = f"{scene}_{condition}_{traj_key}"
            
            # Store sequence data
            if scene not in sequence_data:
                sequence_data[scene] = {}
            if condition not in sequence_data[scene]:
                sequence_data[scene][condition] = {}
                
            sequence_data[scene][condition][seq_name] = {
                "images": np.array(valid_images),
                "c2ws": np.array(sequence_trajectories),
                "trajectory_key": traj_key,
            }
            
            logging.info(f"Processed {scene}/{condition}/{traj_key}: {len(valid_images)} images")
            all_sequence_names.append(seq_name)
            total_processed += 1
        
        return sequence_data, all_sequence_names, total_processed
        


def fetch_and_save_midair_metadata(MidAir_DIR, output_file, num_processes=None):
    """
    Process MidAir dataset and save all metadata in a single npz file using multiprocessing
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)
    
    # Find all sequence directories (e.g., Kite_training/cloudy, PLE_test/spring)
    sequence_dirs = []
    for scene in os.listdir(MidAir_DIR):
        scene_path = os.path.join(MidAir_DIR, scene)
        if not os.path.isdir(scene_path):
            continue
        
        for condition in os.listdir(scene_path):
            condition_path = os.path.join(scene_path, condition)
            if not os.path.isdir(condition_path):
                continue
            
            # Check if this directory has sensor_records.hdf5
            hdf5_path = os.path.join(condition_path, "sensor_records.hdf5")
            if os.path.exists(hdf5_path):
                sequence_dirs.append(condition_path)
    
    logging.info(f"Found {len(sequence_dirs)} sequences to process")
    
    # Use multiprocessing to process sequences in parallel
    if num_processes is None:
        num_processes = min(cpu_count() * 2, len(sequence_dirs))
    
    logging.info(f"Processing {len(sequence_dirs)} sequences using {num_processes} processes")
    
    # Prepare arguments for multiprocessing
    args_list = [(seq_dir, MidAir_DIR) for seq_dir in sequence_dirs]
    
    # Initialize data structures for single file output
    scene_dict = {}
    all_scene_names = []
    total_processed_sequences = 0
    
    # Use multiprocessing Pool to process sequences in parallel
    with Pool(processes=num_processes) as pool:
        results = list(tqdm(pool.imap(process_sequence, args_list), total=len(sequence_dirs), desc="Processing sequences"))
    
    # Merge results from all processes
    for sequence_data, scene_names, processed_sequences in results:
        # Merge sequence_data into scene_dict
        for scene_key, scene_value in sequence_data.items():
            if scene_key not in scene_dict:
                scene_dict[scene_key] = {}
            for condition_key, condition_value in scene_value.items():
                if condition_key not in scene_dict[scene_key]:
                    scene_dict[scene_key][condition_key] = {}
                scene_dict[scene_key][condition_key].update(condition_value)
        
        all_scene_names.extend(scene_names)
        total_processed_sequences += processed_sequences
    
    # Save as a single npz with dict-of-dicts
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    
    logging.info(f"MidAir processing complete: {total_processed_sequences} sequences from {len(all_scene_names)} total sequences")
    logging.info(f"All metadata saved to {output_file}")

def test_single_scene(MidAir_DIR):
    """Test processing of a single scene without multiprocessing for debugging"""
    logging.info("Running single scene test mode")
    
    # Find the first available sequence directory
    sequence_dirs = []
    for scene in os.listdir(MidAir_DIR):
        scene_path = os.path.join(MidAir_DIR, scene)
        if not os.path.isdir(scene_path):
            continue
        
        for condition in os.listdir(scene_path):
            condition_path = os.path.join(scene_path, condition)
            if not os.path.isdir(condition_path):
                continue
            
            # Check if this directory has sensor_records.hdf5
            hdf5_path = os.path.join(condition_path, "sensor_records.hdf5")
            if os.path.exists(hdf5_path):
                sequence_dirs.append(condition_path)
                break
        if sequence_dirs:
            break
    
    if not sequence_dirs:
        logging.error("No valid sequence directories found for testing")
        return
    
    test_sequence = sequence_dirs[0]
    logging.info(f"Testing with sequence: {test_sequence}")
    
    # Process single sequence
    sequence_data, scene_names, processed_sequences = process_sequence((test_sequence, MidAir_DIR))
    
    if processed_sequences > 0:
        logging.info(f"Successfully processed {processed_sequences} sequences:")
        for scene_name in scene_names:
            logging.info(f"  - {scene_name}")
        
        # Print detailed structure
        for scene_key, scene_value in sequence_data.items():
            logging.info(f"Scene: {scene_key}")
            for condition_key, condition_value in scene_value.items():
                logging.info(f"  Condition: {condition_key}")
                for seq_name, seq_data in condition_value.items():
                    logging.info(f"    Sequence: {seq_name}")
                    logging.info(f"      Images: {len(seq_data['images'])}")
                    logging.info(f"      Poses: {len(seq_data['c2ws'])}")
                    logging.info(f"      Trajectory key: {seq_data['trajectory_key']}")
                    
                    # Show first few image paths
                    if len(seq_data['images']) > 0:
                        logging.info(f"      Sample image paths:")
                        for i, img_path in enumerate(seq_data['images'][:3]):
                            logging.info(f"        {i}: {img_path}")
        
        # Save the test data to npz file
        output_file = os.path.join(MidAir_DIR, "midair_all_scenes_dict.npz")
        np.savez_compressed(output_file, **sequence_data)
        logging.info(f"Test data saved to {output_file}")
    else:
        logging.warning("No sequences were successfully processed")

def main():
    parser = argparse.ArgumentParser(description="Process MidAir dataset and save metadata")
    parser.add_argument("--input_dir", type=str, 
                       default="/home/haian/codebase/zipmap_code/raw_data/mid_air",
                       help="Input MidAir directory")
    parser.add_argument("--output_file", type=str,
   
                       help="Output npz file for all processed metadata")
    parser.add_argument("--num_processes", type=int, default=None,
                       help="Number of processes to use (default: 2 * number of CPU cores)")
    parser.add_argument("--test_single", action="store_true",
                       help="Test mode: process only one scene without multiprocessing")
    
    args = parser.parse_args()
    Test = False
    if args.output_file is None:
        args.output_file = os.path.join(args.input_dir, "midair_all_scenes_dict.npz")
    if not Test:
        if args.test_single:
            test_single_scene(args.input_dir)
        else:
            fetch_and_save_midair_metadata(args.input_dir, args.output_file, args.num_processes)
    else:
        # Read the npz file for testing
        data = np.load(args.output_file, allow_pickle=True)
        print("Available scenes:", list(data.keys()))
        print("Number of scenes:", len(data.keys()))
        
        # Print structure
        for scene_name in list(data.keys())[:3]:  # Show first 3 scenes
            scene_data = data[scene_name].item()
            print(f"\nScene: {scene_name}")
            for condition in scene_data:
                condition_data = scene_data[condition]
                print(f"  Condition: {condition}")
                for seq_name in condition_data:
                    seq_data = condition_data[seq_name]
                    print(f"    Sequence: {seq_name}, Images: {len(seq_data['images'])}")
        
        import ipdb; ipdb.set_trace()

if __name__ == "__main__":
    main()