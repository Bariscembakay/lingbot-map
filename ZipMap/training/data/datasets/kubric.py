# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import random
import logging
from data.dataset_util import *
from data.base_dataset import BaseDataset




class KubricDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        KUBRIC_DIR: str = "/path/to/kubric/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 8,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.KUBRIC_DIR = KUBRIC_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True  # Kubric uses depth and camera parameters
        self.video_prob = video_prob
        logging.info(f"KUBRIC_DIR is {self.KUBRIC_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split == "test" or split == "valid":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self._load_data()

    def _load_data(self):
        # Check for precomputed metadata
        metadata_file = osp.join(self.KUBRIC_DIR, f"kubric_metadata.npz")

        assert osp.exists(metadata_file), f"Metadata file not found: {metadata_file}"

        logging.info(f"Kubric loading {self.split} metadata from {metadata_file}")
        sequences_npz = np.load(metadata_file, allow_pickle=True)

        # Initialize data structures
        self.sequences = []  # All sequence names
        self.images_list = []  # List of image basenames per sequence
        self.intrinsics_list = []  # List of intrinsics arrays per sequence (T, 3, 3)
        self.extrinsics_list = []  # List of extrinsics arrays per sequence (T, 4, 4) cam2world
        self.sequence_indices = []  # List of valid sequences with sufficient frames/points

        j = 0

        for seq_name in sorted(sequences_npz.files):
            seq_data = sequences_npz[seq_name].item()

            # Extract data from precomputed metadata
            valid_images = seq_data["images"]  # Basenames
            sequence_intrinsics = seq_data["intrinsics"]  # (T, 3, 3)
            sequence_extrinsics = seq_data["extrinsics"]  # (T, 4, 4) cam2world

            # Filter by minimum frames
            if len(valid_images) < self.min_num_images:
                logging.debug(f"Skipping {seq_name} - insufficient frames ({len(valid_images)})")
                continue

            # Add this sequence to our dataset
            self.sequences.append(seq_name)
            self.images_list.append(valid_images)
            self.intrinsics_list.append(sequence_intrinsics)
            self.extrinsics_list.append(sequence_extrinsics)
            self.sequence_indices.append(j)

            j += 1

            if self.debug and j >= 1:  # Limit to 1 sequence in debug mode
                break

        self.sequence_list_len = len(self.sequence_indices)
        logging.info(f"Kubric {self.split} data loaded: {self.sequence_list_len} sequences with depth + camera")

    def _show_stats(self):
        """Show statistics about the dataset based on loaded metadata."""
        frame_counts = {}

        for i, seq_name in enumerate(self.sequences):
            num_frames = len(self.images_list[i])
            frame_counts[seq_name] = num_frames

        logging.info(f"Frame stats - Mean: {np.mean(list(frame_counts.values())):.2f}, "
                    f"Max: {np.max(list(frame_counts.values()))}, "
                    f"Min: {np.min(list(frame_counts.values()))}")
        logging.info(f"Total sequences: {len(self.sequences)}")

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

        # Get the actual sequence index
        actual_seq_idx = self.sequence_indices[seq_index]

        # Get sequence data
        sequence_name = self.sequences[actual_seq_idx]
        image_basenames = self.images_list[actual_seq_idx]
        sequence_intrinsics = self.intrinsics_list[actual_seq_idx]  # (T, 3, 3) or None
        sequence_extrinsics = self.extrinsics_list[actual_seq_idx]  # (T, 4, 4) or None

        num_frames = len(image_basenames)

        # Sample frames from the sequence
        cut_off = img_per_seq
        if num_frames < cut_off:
            start_frame_options = [0]
        else:
            # Random start position
            start_frame_options = list(range(num_frames - cut_off + 1))
        start_frame = np.random.choice(start_frame_options)

        # Sample with interval (like video frames)
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_frame,
            list(range(num_frames)),
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.5,
            block_shuffle=24
        )

        # Get the selected frame indices
        sampled_frame_indices = pos[:img_per_seq]

        # Reverse order with 0.5 probability
        if random.random() < 0.5:
            sampled_frame_indices = sampled_frame_indices[::-1]

        sampled_basenames = [image_basenames[i] for i in sampled_frame_indices]

        # Load and process images
        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        original_sizes = []

        # Build path to frames directory
        frames_dir = osp.join(self.KUBRIC_DIR, sequence_name, "frames")
        depth_dir = osp.join(self.KUBRIC_DIR, sequence_name, "depths")

        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []

        for i, (basename, frame_idx) in enumerate(zip(sampled_basenames, sampled_frame_indices)):
            # Load RGB image
            image_filepath = osp.join(frames_dir, basename + ".png")
            if not osp.exists(image_filepath):
                # Try other extensions
                for ext in [".jpg", ".jpeg"]:
                    alt_path = osp.join(frames_dir, basename + ext)
                    if osp.exists(alt_path):
                        image_filepath = alt_path
                        break

            image = imread_cv2(image_filepath)
            original_size = np.array(image.shape[:2])
            original_sizes.append(original_size)

   
            depth_filepath = osp.join(depth_dir, basename + ".npy")

            depth_map = np.load(depth_filepath).astype(np.float32)

            # Process depth map
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=99, max_depth=30.0)

            # Get camera parameters for this frame
            intri = sequence_intrinsics[frame_idx].copy()
            camera_pose = sequence_extrinsics[frame_idx].copy()  # cam2world

            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]

            # Process image with depth and camera parameters
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
                filepath=image_filepath,
            )

            if not np.array_equal(image.shape[:2], target_image_shape):
                logging.error(f"Wrong shape for {sequence_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)

        set_name = "kubric"
        sequence_name = sequence_name.replace("/", "_")

        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else sequence_name),
            "dataset_name": "kubric",
            "ids": np.array(sampled_frame_indices),
            "frame_num": len(images),
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


