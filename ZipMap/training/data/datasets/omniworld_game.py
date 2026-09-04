# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import random
import numpy as np
from typing import Dict, Any
import cv2

from data.dataset_util import *  # utilities and geometry helpers
from data.base_dataset import BaseDataset



class OmniWorldGameDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        OMNIWORLD_GAME_DIR: str = "/path/to/processed_OmniWorld-Game/",
        min_num_images: int = 24,
        len_train: int = 20000,
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
        self.OMNIWORLD_GAME_DIR = OMNIWORLD_GAME_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.video_prob = video_prob
        # OmniWorld-Game is synthetic/game data
        self.is_metric = False
        logging.info(f"OMNIWORLD_GAME_DIR is {self.OMNIWORLD_GAME_DIR}")

        if self.split == "train":
            self.split_list = ["train", "test"]
        else:
            self.split_list = [self.split]

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._load_data()

    def _load_cameras_dict(self, cameras_path: str) -> Dict[str, Any]:
        """
        Load the cameras dictionary saved by the preprocessor.
        Expected format: npz file with 'data' key containing dict with split -> scene_id -> split_idx -> camera_data
        """
        if not osp.exists(cameras_path):
            raise FileNotFoundError(f"Cameras file not found: {cameras_path}")

        npz_file = np.load(cameras_path, allow_pickle=True)
        cams = npz_file['data'].item()

        if not isinstance(cams, dict):
            raise TypeError(f"Unexpected cameras file format at {cameras_path}: {type(cams)}")
        return cams

    def _load_data(self):
        cameras_path = osp.join(self.OMNIWORLD_GAME_DIR, "omniworld_game_metadata.npz")
        logging.info(f"Loading OmniWorld-Game cameras from: {cameras_path}")
        cameras_dict = self._load_cameras_dict(cameras_path)

        # Check that all splits in split_list exist
        for split in self.split_list:
            if split not in cameras_dict:
                raise KeyError(f"Split '{split}' not found in cameras file at {cameras_path}")

        # Initialize containers
        self.scenes = []            # scene identifiers (scene_id + split_idx)
        self.scene_splits = []      # track which dataset split each scene came from
        self.scene_ids = []         # original scene_id (for directory lookup)
        self.split_indices = []     # split index within the scene
        self.images = []            # list of basenames across all sequences
        self.trajectories = []      # list of 4x4 world2cam poses
        self.intrinsics_list = []   # list of 3x3 intrinsics
        self.sequence_list = []     # per-sequence list of indices into self.images

        offset = 0
        j = 0

        # Iterate through all dataset splits (train, test)
        for dataset_split in self.split_list:
            split_cams = cameras_dict[dataset_split]

            # Iterate through scenes within the split
            for scene_id in sorted(split_cams.keys()):
                scene_data = split_cams[scene_id]

                # Each scene has multiple "split_X" entries
                for split_key in sorted(scene_data.keys()):
                    if not split_key.startswith("split_"):
                        continue

                    entry = scene_data[split_key]
                    basenames = entry.get("basenames", [])
                    Ks = entry.get("K", None)
                    poses = entry.get("pose", None)

                    if Ks is None or poses is None:
                        logging.warning(f"Missing K/pose for {scene_id}/{split_key}, skipping")
                        continue

                    # Ensure aligned lengths
                    n = len(basenames)
                    if n == 0 or Ks.shape[0] != n or poses.shape[0] != n:
                        logging.warning(
                            f"{scene_id}/{split_key} has inconsistent lengths: "
                            f"basenames={n}, K={getattr(Ks, 'shape', None)}, "
                            f"pose={getattr(poses, 'shape', None)}; skipping"
                        )
                        continue

                    if n < self.min_num_images:
                        logging.debug(f"Skipping {scene_id}/{split_key} - insufficient images ({n})")
                        continue

                    img_ids = list(np.arange(n) + offset)

                    # Extract split index from split_key (e.g., "split_0" -> 0)
                    split_idx = int(split_key.split("_")[1])

                    # Store metadata
                    scene_name = f"{scene_id}_{split_key}"
                    self.scenes.append(scene_name)
                    self.scene_splits.append(dataset_split)
                    self.scene_ids.append(scene_id)
                    self.split_indices.append(split_idx)
                    self.sequence_list.append(img_ids)
                    self.images.extend(list(basenames))
                    self.trajectories.extend([p for p in poses])
                    self.intrinsics_list.extend([k for k in Ks])

                    offset += n
                    j += 1

                    if self.debug and j >= 1:
                        break

                if self.debug and j >= 1:
                    break

            if self.debug and j >= 1:
                break

        self.sequence_list_len = len(self.scenes)

        # Compute sampling weights based on number of images in each sequence
        self.sequence_weights = np.array([len(seq) for seq in self.sequence_list], dtype=np.float32)
        self.sequence_weights /= self.sequence_weights.sum()  # Normalize to probabilities

        splits_str = "+".join(self.split_list) if len(self.split_list) > 1 else self.split
        logging.info(
            f"OmniWorld-Game {splits_str} data loaded: {len(self.sequence_list)} sequences "
            f"and {len(self.images)} images from {len(self.scenes)} scene-splits."
        )

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
        plt.title("OmniWorld-Game Dataset: Number of Images per Sequence")
        plt.tight_layout()
        plt.savefig("omniworld_game_image_count_histogram.png")

        mean = np.mean(list(image_num.values()))
        max_val = np.max(list(image_num.values()))
        min_val = np.min(list(image_num.values()))
        logging.info(f"Mean number of images per sequence: {mean:.2f}")
        logging.info(f"Max number of images per sequence: {max_val}")
        logging.info(f"Min number of images per sequence: {min_val}")
        total_image_num = np.sum(list(image_num.values()))
        logging.info(f"Total number of images in dataset: {total_image_num}")

    def _load_depth(self, depth_path: str) -> tuple:
        """
        Load depth map from 16-bit PNG.

        Returns:
            depthmap: (H, W) float32 depth values
            valid: (H, W) bool mask of valid pixels
        """

        # Read 16-bit PNG with OpenCV (IMREAD_UNCHANGED preserves 16-bit depth)
        depth_uint16 = cv2.imread(depth_path, cv2.IMREAD_UNCHANGED)
        if depth_uint16 is None:
            raise IOError(f"Failed to read depth image: {depth_path}")

        depthmap = depth_uint16.astype(np.float32) / 65535.0

        # Filter invalid regions following OmniWorld's guide
        near_mask = depthmap < 0.0015  # too close
        far_mask = depthmap > (65500.0 / 65535.0)  # sky / too far
        far_mask = depthmap > np.percentile(depthmap[~far_mask], 96)

        # Convert to metric depth
        near, far = 1.0, 1000.0
        depthmap = depthmap / (far - depthmap * (far - near)) / 0.004

        valid = ~(near_mask | far_mask)
        depthmap[~valid] = 0.0

        # Additional conservative thresholding
        depthmap = threshold_depth_map(
            depthmap, min_percentile=-1, max_percentile=99
        )

        return depthmap, valid

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        if self.inside_random and self.training:
            # Sample sequence index weighted by number of images in each sequence
            seq_index = np.random.choice(self.sequence_list_len, p=self.sequence_weights)

        image_idxs = self.sequence_list[seq_index]

        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]

        start_id = np.random.choice(start_image_idxs)

        # OmniWorld-Game is video-like; use fixed-interval sampling
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

        # Get scene metadata
        scene_id = self.scene_ids[seq_index]
        scene_root = osp.join(self.OMNIWORLD_GAME_DIR, scene_id)
        rgb_dir = osp.join(scene_root, "color")
        depth_dir = osp.join(scene_root, "depth")

        for image_idx in ids:
            basename = self.images[image_idx]

            # Load intrinsics and camera pose (pose is cam2world)
            intri = np.array(self.intrinsics_list[image_idx])
            world2cam = np.array(self.trajectories[image_idx])
            extri_opencv = world2cam[:3]

            # Filepaths
            image_filepath = osp.join(rgb_dir, f"{basename}.png")
            depth_filepath = osp.join(depth_dir, f"{basename}.png")

            # Load RGB
            image = imread_cv2(image_filepath)

            # Load depth
            depth_map, valid_mask = self._load_depth(depth_filepath)

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
                logging.error(
                    f"Wrong shape for {scene_id}: expected {target_image_shape}, got {image.shape[:2]}"
                )
                continue
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        set_name = "omniworld_game"
        seq_name_tag = seq_name if seq_name is not None else self.scenes[seq_index].replace("/", "_")

        batch = {
            "seq_name": set_name + "_" + seq_name_tag,
            "dataset_name": "omniworld_game",
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

