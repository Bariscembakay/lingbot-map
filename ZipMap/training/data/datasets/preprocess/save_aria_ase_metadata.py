import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from pathlib import Path
from PIL import Image
from scipy.spatial.transform import Rotation as R
import multiprocessing as mp
from functools import partial

try:
    from projectaria_tools.projects import ase
    from projectaria_tools.core import calibration
    ARIA_TOOLS_AVAILABLE = True
except ImportError:
    ARIA_TOOLS_AVAILABLE = False
    logging.warning("projectaria_tools not available. Cannot process Aria ASE dataset.")

logging.basicConfig(level=logging.INFO, format='%(asctime)s - %(levelname)s - %(message)s')

# Reduce noise from various image processing libraries
logging.getLogger('PIL').setLevel(logging.WARNING)
logging.getLogger('PIL.PngImagePlugin').setLevel(logging.WARNING)
logging.getLogger('PIL.Image').setLevel(logging.WARNING)

# Minimum number of images required per sequence
MIN_NUM_IMAGES = 10

def _transform_from_Rt(R, t):
    M = np.identity(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M

def _read_trajectory_line(line):
    line = line.rstrip().split(",")
    pose = {}
    pose["timestamp"] = int(line[1])
    translation = np.array([float(p) for p in line[3:6]])
    quat_xyzw = np.array([float(o) for o in line[6:10]])
    rot_matrix = R.from_quat(quat_xyzw).as_matrix()
    rot_matrix = np.array(rot_matrix)
    pose["position"] = translation
    pose["rotation"] = rot_matrix
    pose["transform"] = _transform_from_Rt(rot_matrix, translation)
    return pose

def read_trajectory_file(filepath):
    assert os.path.exists(filepath), f"Could not find trajectory file: {filepath}"
    with open(filepath, "r") as f:
        _ = f.readline()  # header
        positions = []
        rotations = []
        transforms = []
        timestamps = []
        for line in f.readlines():
            pose = _read_trajectory_line(line)
            positions.append(pose["position"])
            rotations.append(pose["rotation"])
            transforms.append(pose["transform"])
            timestamps.append(pose["timestamp"])
        positions = np.stack(positions).astype(np.float32)
        rotations = np.stack(rotations).astype(np.float32)
        transforms = np.stack(transforms).astype(np.float32)
        timestamps = np.array(timestamps)
    return {
        "ts": positions,
        "Rs": rotations,
        "Ts_world_from_device": transforms,
        "timestamps": timestamps,
    }

def process_single_scene(scene_id, ARIA_ASE_DIR):
    """Process a single scene and return its metadata"""
    if not ARIA_TOOLS_AVAILABLE:
        return None, f"projectaria_tools not available"

    scene_dir = Path(ARIA_ASE_DIR) / str(scene_id)

    if not scene_dir.exists():
        return None, f"Scene {scene_id} directory does not exist"

    rgb_dir = scene_dir / "rgb"
    depth_dir = scene_dir / "depth"
    trajectory_path = scene_dir / "trajectory.csv"

    if not (rgb_dir.exists() and depth_dir.exists() and trajectory_path.exists()):
        return None, f"Scene {scene_id} missing directories or trajectory"

    try:
        # Load Aria camera calibration (need to load in each process)
        device = ase.get_ase_rgb_calibration()

        # Load trajectory
        trajectory = read_trajectory_file(trajectory_path)
        num_frames = len(trajectory["Ts_world_from_device"])

        # Find actual available frames
        available_frames = []
        sequence_trajectories = []
        sequence_intrinsics = []

        for frame_idx in range(num_frames):
            frame_id = str(frame_idx).zfill(7)
            rgb_path = rgb_dir / f"vignette{frame_id}.jpg"
            depth_path = depth_dir / f"depth{frame_id}.png"

            if rgb_path.exists() and depth_path.exists():
                try:
                    # # Verify we can load the images
                    # _ = Image.open(rgb_path)
                    # _ = Image.open(depth_path)

                    available_frames.append(frame_idx)  # Just store the frame number

                    # Get camera intrinsics (after rectification and rotation)
                    focal_length = device.get_focal_lengths()[0]
                    pinhole = calibration.get_linear_camera_calibration(
                        512, 512, focal_length, "camera-rgb",
                        device.get_transform_device_camera()
                    )
                    pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole)
                    principal = pinhole_cw90.get_principal_point()
                    focal_lengths = pinhole_cw90.get_focal_lengths()

                    # Store intrinsic matrix
                    intrinsic_matrix = np.array([
                        [focal_lengths[0], 0, principal[0]],
                        [0, focal_lengths[1], principal[1]],
                        [0, 0, 1.0]
                    ], dtype=np.float32)
                    sequence_intrinsics.append(intrinsic_matrix)

                    # Get camera pose (world-to-camera)
                    T_world_from_device = trajectory["Ts_world_from_device"][frame_idx].astype(np.float32)
                    c2w_rotation = pinhole_cw90.get_transform_device_camera().to_matrix().astype(np.float32)
                    c2w_final = T_world_from_device @ c2w_rotation

                    # Convert to w2c and store
                    # Using closed_form_inverse_se3 equivalent (simple matrix inverse for SE3)
                    w2c = np.eye(4, dtype=np.float32)
                    w2c[:3, :3] = c2w_final[:3, :3].T
                    w2c[:3, 3] = -c2w_final[:3, :3].T @ c2w_final[:3, 3]
                    sequence_trajectories.append(w2c)

                except Exception as e:
                    logging.warning(f"Failed to process frame {frame_idx} in scene {scene_id}: {e}")
                    # Skip individual frame failures
                    continue

        if len(available_frames) < MIN_NUM_IMAGES:
            return None, f"Scene {scene_id} insufficient valid images ({len(available_frames)})"

        # Return scene data
        scene_data = {
            "frame_indices": np.array(available_frames),  # frame numbers (can reconstruct filenames)
            "c2ws": np.array(sequence_trajectories),  # world-to-camera poses
            "intrinsics": np.array(sequence_intrinsics),  # intrinsic matrices
        }

        return (str(scene_id), scene_data), f"Processed scene {scene_id}: {len(available_frames)} images"

    except Exception as e:
        return None, f"Failed to process scene {scene_id}: {e}"


def fetch_and_save_aria_ase_metadata(ARIA_ASE_DIR, output_file, max_scene_id=500, num_workers=None):
    """
    Process Aria Synthetic Environment dataset and save all metadata in a single npz file
    """
    if not ARIA_TOOLS_AVAILABLE:
        raise ImportError("projectaria_tools is required for processing Aria ASE dataset.")

    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Set default number of workers (conservative to avoid I/O bottlenecks)
    if num_workers is None:
        num_workers = min(mp.cpu_count()*2, 192)  # Conservative default

    logging.info(f"Using {num_workers} worker processes")

    # Create list of scene IDs to process
    scene_ids = list(range(max_scene_id))

    # Initialize data structures for single file output
    scene_dict = {}
    processed_scenes = 0
    failed_scenes = 0

    # Create partial function with fixed ARIA_ASE_DIR
    process_func = partial(process_single_scene, ARIA_ASE_DIR=ARIA_ASE_DIR)

    # Process scenes in parallel
    with mp.Pool(processes=num_workers) as pool:
        # Use tqdm to show progress
        results = list(tqdm(
            pool.imap(process_func, scene_ids),
            total=len(scene_ids),
            desc="Processing scenes"
        ))

    # Collect results
    for result, message in results:
        if result is not None:
            scene_id, scene_data = result
            scene_dict[scene_id] = scene_data
            processed_scenes += 1
            logging.info(message)
        else:
            failed_scenes += 1
            logging.debug(message)

    # Save as a single npz file
    np.savez_compressed(
        output_file,
        **scene_dict
    )

    logging.info(f"Aria ASE processing complete: {processed_scenes} scenes processed, {failed_scenes} scenes failed")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Aria Synthetic Environment dataset and save metadata")
    parser.add_argument("--input_dir", type=str,
                       default="/home/storage-bucket/haian/zipmap_data/processed_aria_synthetic_environments",
                       help="Input Aria ASE directory")
    parser.add_argument("--output_file", type=str,
                       help="Output npz file for all processed metadata")
    parser.add_argument("--max_scene_id", type=int,
                       default=40000,
                       help="Maximum scene ID to process")
    parser.add_argument("--num_workers", type=int,
                       default=None,
                       help="Number of worker processes (default: min(cpu_count, 16))")
    parser.add_argument("--test", action="store_true",
                       help="Test mode: read and inspect existing npz file")

    args = parser.parse_args()
    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "aria_ase_metadata_20k.npz")

    if not args.test:
        fetch_and_save_aria_ase_metadata(args.input_dir, args.output_file, args.max_scene_id, args.num_workers)
    else:
        # Read and inspect the npz file
        if not os.path.exists(args.output_file):
            print(f"File {args.output_file} does not exist!")
            return

        data = np.load(args.output_file, allow_pickle=True)
        print(f"Found {len(data.keys())} scenes")

        # Inspect first few scenes
        for i, scene_key in enumerate(list(data.keys())[:3]):
            scene_data = data[scene_key].item()
            frame_indices = scene_data['frame_indices']
            print(f"\nScene {scene_key}:")
            print(f"  Number of frames: {len(frame_indices)}")
            print(f"  Frame indices range: {frame_indices[0]} - {frame_indices[-1]}")
            print(f"  First few frame indices: {frame_indices[:3]}")
            print(f"  Camera poses shape: {scene_data['c2ws'].shape}")
            print(f"  Intrinsics shape: {scene_data['intrinsics'].shape}")

            # Show example reconstructed filenames
            print(f"  Example filenames:")
            for idx in frame_indices[:3]:
                frame_id = str(idx).zfill(7)
                print(f"    RGB: vignette{frame_id}.jpg, Depth: depth{frame_id}.png")

if __name__ == "__main__":
    main()