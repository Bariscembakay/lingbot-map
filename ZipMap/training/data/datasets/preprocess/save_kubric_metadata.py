import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial
from PIL import Image


# Minimum number of frames required per sequence
MIN_NUM_FRAMES = 10


def quaternion_to_rotation_matrix(quaternion):
    """
    Convert quaternion to rotation matrix.
    Based on the official Kubric implementation.

    The official Kubric code reorders quaternions as:
    rotation_matrix_3d.from_quaternion(tf.concat([quat[1:], quat[0:1]], axis=0))
    This reorders [q0, q1, q2, q3] -> [q1, q2, q3, q0] before conversion.

    Args:
        quaternion: (4,) array in Kubric's storage format

    Returns:
        rotation_matrix: (3, 3) rotation matrix
    """
    # Reorder quaternion following official Kubric implementation
    # [q0, q1, q2, q3] -> [q1, q2, q3, q0]
    if len(quaternion.shape) == 1:
        reordered = np.array([quaternion[1], quaternion[2], quaternion[3], quaternion[0]])
    else:
        reordered = np.concatenate([quaternion[..., 1:], quaternion[..., 0:1]], axis=-1)

    # Extract components (now in w, x, y, z format for standard quaternion formula)
    if len(reordered.shape) == 1:
        x, y, z, w  = reordered
    else:
        x = reordered[..., 0]
        y = reordered[..., 1]
        z = reordered[..., 2]
        w = reordered[..., 3]

    # Normalize quaternion
    norm = np.sqrt(w*w + x*x + y*y + z*z)
    w, x, y, z = w/norm, x/norm, y/norm, z/norm

    # Compute rotation matrix using standard formula (w, x, y, z format)
    R = np.array([
        [1 - 2*(y*y + z*z),     2*(x*y - w*z),     2*(x*z + w*y)],
        [    2*(x*y + w*z), 1 - 2*(x*x + z*z),     2*(y*z - w*x)],
        [    2*(x*z - w*y),     2*(y*z + w*x), 1 - 2*(x*x + y*y)]
    ], dtype=np.float32)

    return R


def build_camera_extrinsics(positions, quaternions):
    """
    Build camera extrinsics (cam2world) from positions and quaternions.

    Args:
        positions: (T, 3) camera positions in world space
        quaternions: (T, 4) camera orientations as quaternions

    Returns:
        extrinsics: (T, 4, 4) cam2world transformation matrices
    """
    T = positions.shape[0]
    extrinsics = np.zeros((T, 4, 4), dtype=np.float32)

    for i in range(T):
        R = quaternion_to_rotation_matrix(quaternions[i])
        t = positions[i]

        # Build 4x4 cam2world matrix
        extrinsics[i, :3, :3] = R
        extrinsics[i, :3, 3] = t
        extrinsics[i, 3, 3] = 1.0

    return extrinsics


def build_camera_intrinsics(focal_length, sensor_width, image_width, image_height):
    """
    Build camera intrinsics matrix from focal length and sensor width.
    Converts from Kubric's camera parameters to standard OpenCV format.

    Based on the official Kubric implementation:
    - f_x = focal_length / sensor_width (normalized)
    - f_y = focal_length / sensor_width * height / width (aspect ratio corrected)
    - Then we multiply by image dimensions to get pixel coordinates

    Args:
        focal_length: scalar or (T,) array - camera focal length
        sensor_width: scalar or (T,) array - camera sensor width
        image_width: int, image width in pixels
        image_height: int, image height in pixels

    Returns:
        intrinsics: (3, 3) or (T, 3, 3) intrinsics matrix in OpenCV format
    """
    is_array = isinstance(focal_length, np.ndarray)

    if is_array:
        T = len(focal_length)
        intrinsics = np.zeros((T, 3, 3), dtype=np.float32)

        for i in range(T):
            fl = focal_length[i] if isinstance(focal_length, np.ndarray) else focal_length
            sw = sensor_width[i] if isinstance(sensor_width, np.ndarray) else sensor_width

            # Kubric's normalized intrinsics
            f_x_norm = fl / sw
            f_y_norm = fl / sw * image_height / image_width

            # Convert to pixel coordinates
            fx = f_x_norm * image_width
            fy = f_y_norm * image_height
            cx = image_width / 2.0
            cy = image_height / 2.0

            intrinsics[i] = np.array([
                [fx, 0, cx],
                [0, fy, cy],
                [0, 0, 1]
            ], dtype=np.float32)
    else:
        # Kubric's normalized intrinsics
        f_x_norm = focal_length / sensor_width
        f_y_norm = focal_length / sensor_width * image_height / image_width

        # Convert to pixel coordinates
        fx = f_x_norm * image_width
        fy = f_y_norm * image_height
        cx = image_width / 2.0
        cy = image_height / 2.0

        intrinsics = np.array([
            [fx, 0, cx],
            [0, fy, cy],
            [0, 0, 1]
        ], dtype=np.float32)

    return intrinsics

def process_single_sequence(seq_name, KUBRIC_DIR):
    """
    Process a single Kubric sequence and extract camera parameters.

    Expected Kubric data structure:
    - KUBRIC_DIR/
      - sequence_name/
        - sequence_name.npy  (contains camera parameters)
        - frames/  (RGB images)
        - depths/  (depth maps as .npy files)

    The .npy file should contain:
    - 'camera': dict with:
      - 'focal_length': scalar or (T,) - camera focal length
      - 'sensor_width': scalar or (T,) - camera sensor width
      - 'positions': (T, 3) - camera positions in world space
      - 'quaternions': (T, 4) - camera orientations as [x,y,z,w]

    Returns: (seq_name, sequence_data_dict) or (seq_name, None) if skipped

    The sequence_data_dict contains:
    - 'images': (T,) array of image basenames
    - 'intrinsics': (T, 3, 3) camera intrinsics matrices
    - 'extrinsics': (T, 4, 4) camera extrinsics matrices (cam2world)
    """
    seq_dir = osp.join(KUBRIC_DIR, seq_name)
    npy_path = osp.join(seq_dir, seq_name + ".npy")
    rgb_dir = osp.join(seq_dir, "frames")

    if not osp.exists(npy_path) or not osp.exists(rgb_dir):
        return (seq_name, None)

    # Get all RGB image filenames, sorted
    try:
        img_files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith((".png", ".jpg", ".jpeg"))
        ])
    except Exception as e:
        logging.warning(f"Failed to list files in {rgb_dir}: {e}")
        return (seq_name, None)

    if len(img_files) < MIN_NUM_FRAMES:
        return (seq_name, None)

    # Load camera parameters from annotation file
    try:
        annot_dict = np.load(npy_path, allow_pickle=True).item()

        intrinsics = None
        extrinsics = None

        if "camera" in annot_dict:
            camera_dict = annot_dict["camera"]

            # Extract camera parameters
            focal_length = camera_dict.get("focal_length", None)
            sensor_width = camera_dict.get("sensor_width", None)
            positions = camera_dict.get("positions", None)  # (T, 3)
            quaternions = camera_dict.get("quaternions", None)  # (T, 4)

            # Get image dimensions from first image
            image_width, image_height = None, None
            if len(img_files) > 0:
                try:
                    sample_img_path = osp.join(rgb_dir, img_files[0])
                    sample_img = Image.open(sample_img_path)
                    image_width, image_height = sample_img.size
                except Exception as e:
                    logging.warning(f"Failed to get image dimensions for {seq_name}: {e}")

            # Build intrinsics matrix if we have the required parameters
            if focal_length is not None and sensor_width is not None and image_width is not None:
                intrinsics = build_camera_intrinsics(
                    focal_length, sensor_width, image_width, image_height
                )

            # Build extrinsics matrix if we have positions and quaternions
            if positions is not None and quaternions is not None:
                extrinsics = build_camera_extrinsics(positions, quaternions)
        else:
            logging.warning(f"No camera data found for {seq_name}")
            return (seq_name, None)

    except Exception as e:
        logging.warning(f"Failed to load annotation data for {seq_name}: {e}")
        return (seq_name, None)

    # Check that we have camera parameters
    if intrinsics is None or extrinsics is None:
        logging.warning(f"Incomplete camera data for {seq_name}")
        return (seq_name, None)

    # Ensure intrinsics is (T, 3, 3)
    if intrinsics.ndim == 2:  # (3, 3) - same for all frames
        intrinsics = np.repeat(intrinsics[None], len(img_files), axis=0)
    elif intrinsics.shape[0] != len(img_files):
        # Truncate if dimension mismatch
        min_len = min(len(img_files), intrinsics.shape[0])
        intrinsics = intrinsics[:min_len]
        img_files = img_files[:min_len]

    # Ensure extrinsics is (T, 4, 4)
    if extrinsics.ndim == 2:  # (4, 4) - same for all frames
        extrinsics = np.repeat(extrinsics[None], len(img_files), axis=0)
    elif extrinsics.shape[0] != len(img_files):
        min_len = min(len(img_files), extrinsics.shape[0])
        extrinsics = extrinsics[:min_len]
        img_files = img_files[:min_len]

    sequence_data = {
        "images": np.array([osp.splitext(f)[0] for f in img_files]),
        "intrinsics": intrinsics.astype(np.float32),  # (T, 3, 3)
        "extrinsics": extrinsics.astype(np.float32),  # (T, 4, 4) cam2world
    }

    return (seq_name, sequence_data)

def fetch_and_save_kubric_metadata(KUBRIC_DIR, output_file, num_workers=None):
    """
    Process Kubric dataset and save metadata

    Args:
        KUBRIC_DIR: Root directory of Kubric dataset
        output_file: Output path for the npz file
        num_workers: Number of worker processes (default: cpu_count())
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    logging.info(f"Using {num_workers} worker processes")

    # Get all sequences in the dataset
    try:
        all_sequences = sorted([
            f for f in os.listdir(KUBRIC_DIR)
            if os.path.isdir(osp.join(KUBRIC_DIR, f))
        ])
    except Exception as e:
        logging.error(f"Failed to list sequences in {KUBRIC_DIR}: {e}")
        return

    logging.info(f"Found {len(all_sequences)} sequences in {KUBRIC_DIR}")

    # Create partial function with parameters bound
    process_func = partial(process_single_sequence, KUBRIC_DIR=KUBRIC_DIR)

    # Process sequences in parallel
    sequence_dict = {}
    processed_sequences = 0
    skipped_sequences = 0

    with Pool(processes=num_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(process_func, all_sequences),
            total=len(all_sequences),
            desc="Processing sequences"
        ))

    # Collect results
    for seq_name, seq_data in results:
        if seq_data is not None:
            sequence_dict[seq_name] = seq_data
            processed_sequences += 1
            num_frames = len(seq_data['images'])
            logging.info(f"Processed {seq_name}: {num_frames} frames with camera parameters")
        else:
            skipped_sequences += 1

    # Save to npz file
    np.savez_compressed(
        output_file,
        **sequence_dict
    )
    logging.info(f"Kubric processing complete: {processed_sequences} sequences processed, "
                f"{skipped_sequences} sequences skipped")
    logging.info(f"Metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process Kubric dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default="/home/storage-bucket/haian/zipmap_data/processed_cotracker3_kubric",
                       help="Input Kubric directory")
    parser.add_argument("--output_file", type=str, default=None,
                       help="Output npz file for processed metadata")
    parser.add_argument("--num_workers", type=int, default=None,
                       help="Number of worker processes (default: cpu_count())")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "kubric_metadata.npz")

    fetch_and_save_kubric_metadata(args.input_dir, args.output_file,
                                   num_workers=args.num_workers)

if __name__ == "__main__":
    main()
