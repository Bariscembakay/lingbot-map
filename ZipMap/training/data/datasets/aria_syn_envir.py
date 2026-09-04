# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional


import os
import os.path as osp
import logging
import random
import numpy as np
import cv2
from data.dataset_util import *
from data.base_dataset import BaseDataset

from PIL import Image
from pathlib import Path
from scipy.spatial.transform import Rotation as R
try:
    from projectaria_tools.projects import ase
    from projectaria_tools.core import calibration
    from projectaria_tools.core.image import InterpolationMethod
    ARIA_TOOLS_AVAILABLE = True
except ImportError:
    ARIA_TOOLS_AVAILABLE = False
    logging.warning("projectaria_tools not available. AriaASEDataset will not work.")

def _transform_from_Rt(R, t):
    M = np.identity(4)
    M[:3, :3] = R
    M[:3, 3] = t
    return M


# Reads a Ground truth trajectory line
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


def distance_to_depth(K, dist, uv=None):
    """Convert ray distance to Z-axis depth"""
    original_shape = dist.shape
    if uv is None and len(dist.shape) >= 2:
        # create mesh grid according to d
        uv = np.stack(np.meshgrid(np.arange(dist.shape[1]), np.arange(dist.shape[0])), -1)
        uv = uv.reshape(-1, 2)
        dist_flat = dist.reshape(-1)
    else:
        dist_flat = dist

    if isinstance(dist_flat, np.ndarray):
        # z * np.sqrt(x_temp**2+y_temp**2+z_temp**2) = dist
        uvh = np.concatenate([uv, np.ones((len(uv), 1))], -1)
        uvh = uvh.T # 3, N
        temp_point = np.linalg.inv(K) @ uvh # 3, N
        temp_point = temp_point.T # N, 3
        z = dist_flat / np.linalg.norm(temp_point, axis=1)
        z = z.reshape(original_shape)
    return z


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


class AriaASEDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ARIA_SE_DIR: str = "/home/haian/Dataset/ase_raw/",
        min_num_images: int = 24,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 10,
        max_scene_id: int = 40000,
        video_prob: float = 0.6
    ):
        if not ARIA_TOOLS_AVAILABLE:
            raise ImportError("projectaria_tools is required for AriaASEDataset. Please install it.")

        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.ARIA_SE_DIR = ARIA_SE_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.max_scene_id = max_scene_id
        self.video_prob = video_prob
        self.is_metric = True

        logging.info(f"ARIA_SE_DIR is {self.ARIA_SE_DIR}")

        if split == "train":
            self.len_train = len_train
            self.scene_range = (0, int(max_scene_id))
        elif split == "test":
            self.len_train = len_test
            # Use last 20% of scenes for testing
            self.scene_range = (int(0.8 * max_scene_id), max_scene_id)
        else:
            raise ValueError(f"Invalid split: {split}")

        # Load Aria camera calibration
        self.device = ase.get_ase_rgb_calibration()

        self._load_data()

    def _load_data(self):
        """Load scene metadata from precomputed file or process from raw data"""
        # Check for precomputed metadata
        # metadata_file = osp.join(self.ARIA_SE_DIR, "aria_ase_metadata.npz")
        metadata_file = osp.join(self.ARIA_SE_DIR, "aria_ase_metadata_40k.npz")

        if osp.exists(metadata_file):
            self._load_from_metadata(metadata_file)
        else:
            logging.warning(f"Precomputed metadata not found at {metadata_file}.")
            logging.warning("Please run save_aria_ase_metadata.py first for faster loading.")
            self._load_from_raw_data()

    def _load_from_metadata(self, metadata_file):
        """Load data from precomputed metadata file"""
        logging.info(f"Loading Aria ASE metadata from precomputed file {metadata_file}")

        # Load precomputed metadata
        all_metadata = np.load(metadata_file, allow_pickle=True)

        # Get available scenes based on split
        available_scene_ids = []
        for scene_id in range(self.scene_range[0], self.scene_range[1]):
            if str(scene_id) in all_metadata:
                available_scene_ids.append(scene_id)

        if self.debug:
            available_scene_ids = available_scene_ids[:10]

        # Initialize data structures
        self.scenes = []
        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.scene_img_list = []
        self.start_img_ids = []

        offset = 0
        actual_scene_idx = 0  # Track the actual scene index (after filtering)

        for _, scene_id in enumerate(available_scene_ids):
            scene_data = all_metadata[str(scene_id)].item()

            frame_indices = scene_data["frame_indices"]
            sequence_trajectories = scene_data["c2ws"]
            sequence_intrinsics = scene_data["intrinsics"]

            if len(frame_indices) < self.min_num_images:
                continue

            # Store scene data
            self.scenes.append(scene_id)
            img_ids = list(range(offset, offset + len(frame_indices)))

            # Create start positions for sequences (like TartanAir)
            cut_off = self.min_num_images
            start_img_ids_ = img_ids[: len(frame_indices) - cut_off + 1]

            self.scene_img_list.append(img_ids)
            self.sceneids.extend([actual_scene_idx] * len(frame_indices))
            self.start_img_ids.extend(start_img_ids_)

            actual_scene_idx += 1  # Increment only when we actually add a scene

            # Store frame data
            frame_data = []
            intrinsic_data = []

            for i, frame_idx in enumerate(frame_indices):
                # Create frame_id string and store scene_id/frame_id format
                frame_id = str(frame_idx).zfill(7)
                frame_data.append(f"{scene_id}/{frame_id}")

                # Convert intrinsic matrix to [width, height, fx, fy, cx, cy] format
                K = sequence_intrinsics[i]
                intrinsic_data.append([512, 512, float(K[0, 0]), float(K[1, 1]),
                                     float(K[0, 2]), float(K[1, 2])])

            self.images.extend(frame_data)
            self.intrinsics.extend(intrinsic_data)
            self.trajectories.extend(sequence_trajectories)

            offset += len(frame_indices)


        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"Aria_SE {self.split} data loaded: {len(self.start_img_ids)} start positions from {len(self.scenes)} scenes")

    def _load_from_raw_data(self):
        """Load data by processing raw files (fallback method)"""
        # Find available scenes
        available_scenes = []
        for scene_id in range(self.scene_range[0], self.scene_range[1]):
            scene_dir = Path(self.ARIA_SE_DIR) / str(scene_id)
            if scene_dir.exists():
                rgb_dir = scene_dir / "rgb"
                depth_dir = scene_dir / "depth"
                trajectory_path = scene_dir / "trajectory.csv"

                if rgb_dir.exists() and depth_dir.exists() and trajectory_path.exists():
                    # Count available frames
                    rgb_files = list(rgb_dir.glob("vignette*.jpg"))
                    depth_files = list(depth_dir.glob("depth*.png"))

                    if len(rgb_files) >= self.min_num_images and len(depth_files) >= self.min_num_images:
                        available_scenes.append(scene_id)

        if self.debug:
            available_scenes = available_scenes[:10]
        logging.info(f"Found {len(available_scenes)} valid scenes for {self.split}")

        # Initialize data structures
        self.scenes = []
        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.scene_img_list = []
        self.start_img_ids = []

        offset = 0
        actual_scene_idx = 0  # Track the actual scene index (after filtering)

        for _, scene_id in enumerate(available_scenes):
            scene_dir = Path(self.ARIA_SE_DIR) / str(scene_id)

            # Load trajectory
            trajectory_path = scene_dir / "trajectory.csv"
            trajectory = read_trajectory_file(trajectory_path)
            num_frames = len(trajectory["Ts_world_from_device"])

            # Find actual available frames
            rgb_dir = scene_dir / "rgb"
            available_frames = []
            for frame_idx in range(num_frames):
                frame_id = str(frame_idx).zfill(7)
                rgb_path = rgb_dir / f"vignette{frame_id}.jpg"
                depth_path = scene_dir / "depth" / f"depth{frame_id}.png"

                if rgb_path.exists() and depth_path.exists():
                    available_frames.append(frame_idx)

            if len(available_frames) < self.min_num_images:
                continue

            # Store scene data
            self.scenes.append(scene_id)
            img_ids = list(range(offset, offset + len(available_frames)))

            # Create start positions for sequences
            cut_off = self.min_num_images
            start_img_ids_ = img_ids[: len(available_frames) - cut_off + 1]

            self.scene_img_list.append(img_ids)
            self.sceneids.extend([actual_scene_idx] * len(available_frames))
            self.start_img_ids.extend(start_img_ids_)

            actual_scene_idx += 1  # Increment only when we actually add a scene

            # Store frame data
            frame_data = []
            intrinsic_data = []
            trajectory_data = []

            for frame_idx in available_frames:
                frame_id = str(frame_idx).zfill(7)
                frame_data.append(f"{scene_id}/{frame_id}")

                # Get camera intrinsics (after rectification and rotation)
                focal_length = self.device.get_focal_lengths()[0]
                pinhole = calibration.get_linear_camera_calibration(
                    512, 512, focal_length, "camera-rgb",
                    self.device.get_transform_device_camera()
                )
                pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole)
                principal = pinhole_cw90.get_principal_point()
                focal_lengths = pinhole_cw90.get_focal_lengths()

                # Store as [width, height, fx, fy, cx, cy] format
                intrinsic_data.append([512, 512, float(focal_lengths[0]), float(focal_lengths[1]),
                                     float(principal[0]), float(principal[1])])

                # Get camera pose (world-to-camera)
                T_world_from_device = trajectory["Ts_world_from_device"][frame_idx].astype(np.float32)
                c2w_rotation = pinhole_cw90.get_transform_device_camera().to_matrix().astype(np.float32)
                c2w_final = T_world_from_device @ c2w_rotation
                # Convert to w2c
                w2c = closed_form_inverse_se3(c2w_final[None])[0].astype(np.float32)
                trajectory_data.append(w2c)

            self.images.extend(frame_data)
            self.intrinsics.extend(intrinsic_data)
            self.trajectories.extend(trajectory_data)

            offset += len(available_frames)


        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"AriaASE {self.split} data loaded: {len(self.start_img_ids)} start positions from {len(self.scenes)} scenes")

    def _show_stats(self):
        logging.info(f"AriaASE {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of start positions: {len(self.start_img_ids)}")

        # Statistics on scene level
        scene_image_counts = [len(scene_imgs) for scene_imgs in self.scene_img_list]
        if scene_image_counts:
            min_count = min(scene_image_counts)
            max_count = max(scene_image_counts)
            avg_count = sum(scene_image_counts) / len(scene_image_counts)
            logging.info(f"  Image count per scene: min={min_count}, max={max_count}, avg={avg_count:.2f}")

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        # Use TartanAir-style sampling logic
        start_id = self.start_img_ids[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.5,
            block_shuffle=24,
        )
        ids = np.array(all_image_ids)[pos]

        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            scene_id_for_image = self.sceneids[image_idx]
            scene = self.scenes[scene_id_for_image]

            # Parse frame info
            frame_path = self.images[image_idx]  # Format: "scene_id/frame_id"
            frame_id = frame_path.split('/')[-1]

            scene_dir = Path(self.ARIA_SE_DIR) / str(scene)

            # Get camera pose (this is already w2c from preprocessing)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)
            extri_opencv = camera_pose[:3]

            # Get the rotated intrinsics that match the rotated image
            intri_data = self.intrinsics[image_idx]
            fx, fy, cx, cy = intri_data[2], intri_data[3], intri_data[4], intri_data[5]
            intri = np.array([[fx, 0, cx], [0, fy, cy], [0, 0, 1]], dtype=np.float32)

            # Load raw RGB and depth
            rgb_path = scene_dir / "rgb" / f"vignette{frame_id}.jpg"
            depth_path = scene_dir / "depth" / f"depth{frame_id}.png"

            # Load raw images
            raw_rgb = cv2.imread(str(rgb_path), cv2.IMREAD_UNCHANGED)
            raw_depth = Image.open(depth_path)  # uint16
            raw_depth = np.array(raw_depth)

            # Apply camera calibration (rectification and rotation) - exactly like preprocessing
            focal_length = self.device.get_focal_lengths()[0]
            pinhole = calibration.get_linear_camera_calibration(
                512, 512, focal_length, "camera-rgb",
                self.device.get_transform_device_camera()
            )

            # Rectify images
            rectified_rgb = calibration.distort_by_calibration(
                raw_rgb, pinhole, self.device, InterpolationMethod.BILINEAR
            )

            # Convert depth to float32 before rectification (like preprocessing)
            raw_depth_float = raw_depth.astype(np.float32)
            rectified_depth = calibration.distort_by_calibration(
                raw_depth_float, pinhole, self.device
            )

            # Rotate 270 degrees (k=3) - like preprocessing
            image = np.rot90(rectified_rgb, k=3)
            depth_map = np.rot90(rectified_depth, k=3)

            # Process depth: convert from ray depth (mm) to Z-axis depth (meters)
            inf_value = np.iinfo(np.uint16).max
            depth_map[depth_map == inf_value] = 0
            depth_map = depth_map.astype(np.float32) / 1000.0  # Convert mm to meters

            # Convert ray depth to Z-axis depth using the rotated intrinsics
            # Get rotated intrinsics after calibration
            pinhole_cw90 = calibration.rotate_camera_calib_cw90deg(pinhole)
            principal = pinhole_cw90.get_principal_point()
            focal_lengths = pinhole_cw90.get_focal_lengths()
            K_rotated = np.array([
                [focal_lengths[0], 0, principal[0]],
                [0, focal_lengths[1], principal[1]],
                [0, 0, 1.0]
            ])

            # Convert ray depth to Z-axis depth
            depth_map = distance_to_depth(K_rotated, depth_map)

            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=-1, max_depth=30.0
            )


            original_size = np.array(image.shape[:2])

            (
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                world_coords_points,
                cam_coords_points,
                point_mask,
                _,
            ) = self.process_one_image(
                image,
                depth_map,
                extri_opencv,
                intri,
                original_size,
                target_image_shape,
                filepath=str(rgb_path),
            )

            if not np.array_equal(image.shape[:2], target_image_shape):
                logging.error(f"Wrong shape for {scene}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map.astype(np.float32))
            extrinsics.append(extri_opencv.astype(np.float32))
            intrinsics.append(intri_opencv.astype(np.float32))
            cam_points.append(cam_coords_points.astype(np.float32) if cam_coords_points is not None else None)
            world_points.append(world_coords_points.astype(np.float32) if world_coords_points is not None else None)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "aria_ase"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "aria_syn_envir",
            "ids": np.array(ids),
            "frame_num": len(extrinsics),
            "images": images,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch


