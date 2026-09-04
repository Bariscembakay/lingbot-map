# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import os
import random
import logging
import re
from data.dataset_util import *
from data.base_dataset import BaseDataset


class MatrixCityDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        MATRIXCITY_DIR: str = "/path/to/matrixcity/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 3,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.MATRIXCITY_DIR = MATRIXCITY_DIR
        self.max_interval = max_interval
        self.video_prob = video_prob
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True  # MatrixCity has metric depth
        logging.info(f"MATRIXCITY_DIR is {self.MATRIXCITY_DIR}")

        if split == "train":
            self.len_train = len_train
            self.split_list = ["train",  'test']
        elif split == "test":
            self.len_train = len_test
            self.split_list = ['test']
        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split
        self._load_data()

    def _load_data(self):
        # Check for precomputed metadata
        transforms_file = osp.join(self.MATRIXCITY_DIR, "transforms_all.npz")
        assert os.path.exists(transforms_file), f"Precomputed transforms_all.npz not found at {transforms_file}. Please run preprocess_matrixcity.py first."

        # Load precomputed metadata
        all_transforms = np.load(transforms_file, allow_pickle=True)
        transforms_dict = all_transforms['arr_0'].item()  # Convert numpy array back to dict

        # Initialize data structures
        self.scenes = []  # All scene paths; len = number of scenes
        self.images = []  # All Image names; len = total images
        self.trajectories = []  # Camera poses; len = total images
        self.intrinsics_list = []  # Camera intrinsics; len = total images
        self.sceneids = []  # Maps image index to scene ID
        self.scene_img_list = []  # List of image indices for each scene
        self.start_img_ids = []  # Valid start positions for sequences

        offset = 0
        j = 0

        logging.info("MatrixCity loading metadata from precomputed transforms_all.npz")

        for scene_key, scene_data in transforms_dict.items():
            cur_cat, type_, split_name, scene_name = scene_key

            if split_name not in self.split_list:
                continue

            # Extract image data - sort numerically by filename
            def extract_number(filename):
                """Extract number from filename for proper sorting"""
                match = re.search(r'\d+', filename)
                if match:
                    return int(match.group())
                return 0

            image_names = sorted(scene_data.keys(), key=extract_number)  # Image basenames like '0000.png'
            
            if len(image_names) < self.min_num_images:
                logging.debug(f"Skipping {scene_key} - insufficient images ({len(image_names)})")
                continue

            # Extract poses and intrinsics
            sequence_trajectories = []
            sequence_intrinsics = []
            valid_images = []

            for img_name in image_names:
                img_data = scene_data[img_name]
                intrinsic = img_data["intrinsic"]
                c2w = img_data["c2w"]  # cam2world matrix

                sequence_trajectories.append(c2w)
                sequence_intrinsics.append(intrinsic)
                valid_images.append(img_name)

            # Add this scene to our dataset
            img_ids = list(np.arange(len(valid_images)) + offset)

            num_imgs = len(valid_images)
            cut_off = self.min_num_images
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]

            scene_identifier = f"{cur_cat}_{type_}_{split_name}_{scene_name}"
            self.scenes.append((cur_cat, type_, split_name, scene_name))
            self.scene_img_list.append(img_ids)
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(valid_images)
            self.trajectories.extend(sequence_trajectories)
            self.intrinsics_list.extend(sequence_intrinsics)
            self.start_img_ids.extend(start_img_ids_)

            offset += len(valid_images)
            j += 1

            if self.debug and j >= 2:  # Limit to 2 scenes in debug mode
                break

        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"MatrixCity {self.split} data loaded: {len(self.start_img_ids)} start positions from {len(self.scenes)} scenes")

    def _show_stats(self):
        logging.info(f"MatrixCity {self.split} dataset statistics:")
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

        # Use Cut3r-style sampling logic exactly like OmniObject3D
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
            block_shuffle=24
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

        cur_cat, type_, split_name, scene_name = self.scenes[scene_id]

        for image_idx in ids:
            basename = self.images[image_idx]
            # Construct full paths based on preprocessing structure
            scene_dir = osp.join(self.MATRIXCITY_DIR, cur_cat, type_, split_name, scene_name)

            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            camera_pose[:3, :3] = camera_pose[:3, :3] * 100
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]

            # Load RGB image and depth map
            image_filepath = osp.join(scene_dir, basename)
            depth_filepath = osp.join(scene_dir, basename.replace('.png', '.exr'))
            # Load RGB image
            image = imread_cv2(image_filepath)
            # print(f"Loading image from {image_filepath}")
            # Load depth map (EXR format from preprocessing)
            depth_map = imread_cv2(depth_filepath) * 100  # meter

            # Process depth map - MatrixCity specific processing
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

            # Apply reasonable depth range for city scenes
            if type_ == 'aerial':
                max_depth = 1000.0   # Aerial views can see very far
            else:
                max_depth = 200.0   # Street views have closer range   

            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=99, max_depth=max_depth
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
                filepath=image_filepath,
            )

            if not np.array_equal(image.shape[:2], target_image_shape):
                logging.error(f"Wrong shape for {scene_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "matrixcity"
        scene_path = f"{cur_cat}_{type_}_{scene_name}_{start_id}"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "matrixcity",
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

