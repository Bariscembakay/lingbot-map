# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import random
import logging

from data.dataset_util import *
from data.base_dataset import BaseDataset



# Scenes to use (as defined in the original dataset)
SCENES_TO_USE = [
    "cnb_dlab_0215_3rd",
    "cnb_dlab_0215_ego1",
    "cnb_dlab_0225_3rd",
    "cnb_dlab_0225_ego1",
    "dancing",
    "dancingroom0_3rd",
    "footlab_3rd",
    "footlab_ego1",
    "footlab_ego2",
    "girl",
    "girl_egocentric",
    "human_egocentric",
    "human_in_scene",
    "human_in_scene1",
    "kg",
    "kg_ego1",
    "kg_ego2",
    "kitchen_gfloor",
    "kitchen_gfloor_ego1",
    "kitchen_gfloor_ego2",
    "scene_carb_h_tables",
    "scene_carb_h_tables_ego1",
    "scene_carb_h_tables_ego2",
    "scene_j716_3rd",
    "scene_j716_ego1",
    "scene_j716_ego2",
    "scene_recording_20210910_S05_S06_0_3rd",
    "scene_recording_20210910_S05_S06_0_ego2",
    "scene1_0129",
    "scene1_0129_ego",
    "seminar_h52_3rd",
    "seminar_h52_ego1",
    "seminar_h52_ego2",
]


class PointOdysseyDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        POINTODYSSEY_DIR: str = "/path/to/pointodyssey/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 4,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.POINTODYSSEY_DIR = POINTODYSSEY_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True  # PointOdyssey is metric
        logging.info(f"POINTODYSSEY_DIR is {self.POINTODYSSEY_DIR}")
        self.video_prob = video_prob

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
        metadata_file = osp.join(self.POINTODYSSEY_DIR, "all_scenes_dict.npz")
        all_scenes_dict_exists = True
        all_scenes_npz = np.load(metadata_file, allow_pickle=True)
        # The file structure is: {split: {scene: {images, c2ws, intrinsics}}}
        if self.split in all_scenes_npz:
            all_scenes_dict = all_scenes_npz[self.split].item()
        else:
            logging.warning(f"Split '{self.split}' not found in {metadata_file}, falling back to slow loading")
            all_scenes_dict_exists = False


        # Initialize data structures
        self.scenes = []  # All scene paths; len = number of scenes
        self.images = []  # All Image names; len = total images
        self.trajectories = []  # Camera poses; len = total images
        self.intrinsics_list = []  # Camera intrinsics; len = total images
        self.sequence_list = []  # list of list of image indices; len = number of scenes

        offset = 0
        j = 0

        logging.info(f"PointOdyssey loading {self.split} metadata from precomputed {metadata_file}")

        for scene_name in sorted(all_scenes_dict.keys()):
            # Only use scenes in the approved list
            if scene_name not in SCENES_TO_USE:
                continue

            scene_data = all_scenes_dict[scene_name]
            # Handle both dict and numpy object formats
            if hasattr(scene_data, 'item'):
                scene_data = scene_data.item()

            # Extract data from precomputed metadata
            valid_images = scene_data["images"]  # These are basenames
            sequence_trajectories = scene_data["c2ws"]  # Camera poses (cam2world)
            sequence_intrinsics = scene_data["intrinsics"]  # Camera intrinsics

            if len(valid_images) < self.min_num_images:
                logging.debug(f"Skipping {scene_name} - insufficient images ({len(valid_images)})")
                continue

            # Add this scene to our dataset
            img_ids = list(np.arange(len(valid_images)) + offset)

            self.scenes.append(scene_name)
            self.sequence_list.append(img_ids)
            self.images.extend(valid_images)
            self.trajectories.extend(sequence_trajectories)
            self.intrinsics_list.extend(sequence_intrinsics)

            offset += len(valid_images)
            j += 1

            if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                break



        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"PointOdyssey {self.split} data loaded: {len(self.sequence_list)} scene trajectories")

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
        plt.title(f"PointOdyssey {self.split.capitalize()} Dataset: Number of Images per Sequence")
        plt.tight_layout()
        plt.savefig(f"pointodyssey_{self.split}_image_count_histogram.png")

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

        # PointOdyssey is video-like; use fixed-interval sampling
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            image_idxs,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.8,
            block_shuffle=24
        )
        ids = np.array(image_idxs)[pos]

        # reverse order with 0.5 prob
        if random.random() < 0.5:
            ids = ids[::-1]

        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        scene_path = self.scenes[seq_index]

        for image_idx in ids:
            basename = self.images[image_idx]
            # Construct full paths
            scene_dir = osp.join(self.POINTODYSSEY_DIR, self.split, scene_path)
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")

            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]

            # Load RGB image and depth map
            image_filepath = osp.join(rgb_dir, basename + ".jpg")
            depth_filepath = osp.join(depth_dir, basename + ".npy")

            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)

            # Process depth map (from original: invalid and > 1000m)
            depth_map[~np.isfinite(depth_map)] = 0  # invalid
            depth_map[depth_map > 1000] = 0  # invalid (PointOdyssey uses 1000m threshold)
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

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
                logging.error(f"Wrong shape for {scene_path}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "pointodyssey"
        scene_path = scene_path.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "pointodyssey",
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
