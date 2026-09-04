# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional


import os.path as osp
import numpy as np
import os
import random
import logging

from data.dataset_util import *
from data.base_dataset import BaseDataset




class Cop3DDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        COP3D_DIR: str = "/path/to/cop3d/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 5,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.COP3D_DIR = COP3D_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = False  # Cop3D is NOT metric (like Co3d)
        logging.info(f"COP3D_DIR is {self.COP3D_DIR}")

        if split == "train":
            self.len_train = len_train
        elif split in ["test", "val"]:
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self._load_data()

    def _load_data(self):
        # Check for precomputed metadata
        metadata_file = osp.join(self.COP3D_DIR, "all_scenes_dict.npz")

        assert os.path.exists(metadata_file), f"Metadata file not found: {metadata_file}. Please run the preprocessing script to generate it."
        all_scenes_npz = np.load(metadata_file, allow_pickle=True)

        # The file structure is: {split: {scene: {frames, c2ws, intrinsics}}}
        if self.split not in all_scenes_npz:
            raise ValueError(f"Split '{self.split}' not found in {metadata_file}")

        all_scenes_dict = all_scenes_npz[self.split].item()

        # Initialize data structures
        self.scenes = []  # All scene names (obj_instance); len = number of scenes
        self.scene_keys = []  # All scene tuples (obj, instance); len = number of scenes
        self.images = []  # All frame indices; len = total frames
        self.trajectories = []  # Camera poses; len = total frames
        self.intrinsics_list = []  # Camera intrinsics; len = total frames
        self.sequence_list = []  # list of list of image indices; len = number of scenes

        offset = 0
        j = 0

        # Fast path: use precomputed metadata
        logging.info(f"Cop3D loading {self.split} metadata from precomputed {metadata_file}")

        for scene_name in sorted(all_scenes_dict.keys()):
            scene_data = all_scenes_dict[scene_name]
            # Handle both dict and numpy object formats
            if hasattr(scene_data, 'item'):
                scene_data = scene_data.item()

            # Extract data from precomputed metadata
            valid_frames = scene_data["frames"]  # Frame indices
            sequence_trajectories = scene_data["c2ws"]  # Camera poses (cam2world)
            sequence_intrinsics = scene_data["intrinsics"]  # Camera intrinsics

            if len(valid_frames) < self.min_num_images:
                logging.debug(f"Skipping {scene_name} - insufficient images ({len(valid_frames)})")
                continue

            # Parse scene_name back to (obj, instance) tuple
            obj, instance = scene_name.split("_", 1)

            # Add this scene to our dataset
            img_ids = list(np.arange(len(valid_frames)) + offset)

            self.scenes.append(scene_name)
            self.scene_keys.append((obj, instance))
            self.sequence_list.append(img_ids)
            self.images.extend(valid_frames)
            self.trajectories.extend(sequence_trajectories)
            self.intrinsics_list.extend(sequence_intrinsics)

            offset += len(valid_frames)
            j += 1

            if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                break

        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"Cop3D {self.split} data loaded: {len(self.sequence_list)} scene trajectories")

    def _show_stats(self):
        """Show statistics about the dataset based on loaded metadata."""
        # Calculate number of images per sequence from metadata
        image_num = dict()
        for i, scene_name in enumerate(self.scenes):
            num_images = len(self.sequence_list[i])
            image_num[scene_name] = num_images

        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(image_num)), list(image_num.values()))
        plt.xlabel("Sequence Index")
        plt.ylabel("Number of Images")
        plt.title(f"Cop3D {self.split.capitalize()} Dataset: Number of Images per Sequence")
        plt.tight_layout()
        plt.savefig(f"cop3d_{self.split}_image_count_histogram.png")

        mean = np.mean(list(image_num.values()))
        max_val = np.max(list(image_num.values()))
        min_val = np.min(list(image_num.values()))
        logging.info(f"Mean number of images per sequence: {mean:.2f}")
        logging.info(f"Max number of images per sequence: {max_val}")
        logging.info(f"Min number of images per sequence: {min_val}")
        total_image_num = np.sum(list(image_num.values()))
        logging.info(f"Total number of images in dataset: {total_image_num}")

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

        # Get image indices for this sequence
        image_idxs = self.sequence_list[seq_index]

        # Sample images from the sequence
        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]
        start_id = np.random.choice(start_image_idxs)

        # Cop3D is video-like; use fixed-interval sampling (like original)
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            image_idxs,
            max_interval=self.max_interval,
            video_prob=1.0,
            fix_interval_prob=0.9,
            block_shuffle=24
        )
        ids = np.array(image_idxs)[pos]

        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        obj, instance = self.scene_keys[seq_index]
        scene_dir = osp.join(self.COP3D_DIR, obj, instance)

        for image_idx in ids:
            frame_idx = self.images[image_idx]

            # Construct full paths
            images_dir = osp.join(scene_dir, "images")

            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]

            # Load RGB image (Cop3D uses JPG)
            image_filepath = osp.join(images_dir, f"frame{frame_idx:06n}.jpg")

            image = imread_cv2(image_filepath)

            # Cop3D has no depth - create dummy depth (all ones like original)
            depth_map = np.ones((image.shape[0], image.shape[1]), dtype=np.float32)

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
                filepath=image_filepath,
            )

            if not np.array_equal(image.shape[:2], target_image_shape):
                logging.error(f"Wrong shape for {obj}/{instance}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "cop3d"
        scene_path = self.scenes[seq_index]
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "cop3d",
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


