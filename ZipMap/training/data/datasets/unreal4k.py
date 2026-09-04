# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import os
import random
import logging

from data.dataset_util import *
from data.base_dataset import BaseDataset


# R_conv matrix from original UnReal4K loader
R_conv = np.array([[0, 1, 0, 0], [1, 0, 0, 0], [0, 0, 1, 0], [0, 0, 0, 1]]).astype(np.float32)

class UnReal4KDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        UNREAL4K_DIR: str = "/path/to/unreal4k/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 2,
        video_prob: float = 0.8,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.UNREAL4K_DIR = UNREAL4K_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True  # UnReal4K is metric
        self.video_prob = video_prob
        logging.info(f"UNREAL4K_DIR is {self.UNREAL4K_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def _load_data(self):
        # Check for precomputed metadata
        assert os.path.exists(osp.join(self.UNREAL4K_DIR, "all_sequences_dict.npz")), "Precomputed all_sequences_dict.npz not found. Please run save_unreal4k_metadata.py first."
        all_sequences_dict = np.load(osp.join(self.UNREAL4K_DIR, "all_sequences_dict.npz"), allow_pickle=True)
        # Initialize data structures
        self.scenes = []  # All sequence paths; len = number of sequences
        self.images = []  # All Image names; len = total images
        self.trajectories = []  # Camera poses; len = total images
        self.intrinsics_list = []  # Camera intrinsics; len = total images
        self.sequence_list = []  # list of list of image indices; len = number of sequences
        
        offset = 0
        j = 0
      
        # Fast path: use precomputed metadata
        logging.info("UnReal4K loading metadata from precomputed all_sequences_dict.npz")
        
        for seq_key in all_sequences_dict.keys():
            seq_data = all_sequences_dict[seq_key].item()  # Convert numpy array back to dict
            
            # Extract data from precomputed metadata
            valid_images = seq_data["images"]  # These are basenames
            sequence_trajectories = seq_data["c2ws"]  # Camera poses (cam2world)
            sequence_intrinsics = seq_data["intrinsics"]  # Camera intrinsics
            
            if len(valid_images) < self.min_num_images:
                logging.debug(f"Skipping {seq_key} - insufficient images ({len(valid_images)})")
                continue
            
            # Add this sequence to our dataset
            img_ids = list(np.arange(len(valid_images)) + offset)
            
            self.scenes.append(seq_key)
            self.sequence_list.append(img_ids)
            self.images.extend(valid_images)
            self.trajectories.extend(sequence_trajectories)
            self.intrinsics_list.extend(sequence_intrinsics)
            
            offset += len(valid_images)
            j += 1
            
            if self.debug and j >= 1:  # Limit to 1 sequence in debug mode
                break


        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"UnReal4K {self.split} data loaded: {len(self.sequence_list)} sequence trajectories")

    def _show_stats(self):
        logging.info(f"UnReal4K {self.split} dataset statistics:")
        logging.info(f"  Number of sequences: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of sequence starts: {len(self.sequence_list)}")
        # Statistics on sequence level
        sequence_image_counts = [len(seq_imgs) for seq_imgs in self.sequence_list]
        if sequence_image_counts:
            min_count = min(sequence_image_counts)
            max_count = max(sequence_image_counts)
            avg_count = sum(sequence_image_counts) / len(sequence_image_counts)
            logging.info(f"  Image count per sequence: min={min_count}, max={max_count}, avg={avg_count:.2f}")

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
        
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            image_idxs,
            max_interval=self.max_interval,
            video_prob=0.8,
            fix_interval_prob=0.5,
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
        scene_path = self.scenes[seq_index]
        
        for image_idx in ids:
            basename = self.images[image_idx]
            # Construct full paths
            seq_dir = osp.join(self.UNREAL4K_DIR, scene_path)
            
            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            
            # Apply R_conv transformation as in original
            camera_pose = R_conv @ camera_pose
            
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image and depth map
            image_filepath = osp.join(seq_dir, basename + "_rgb.png")
            depth_filepath = osp.join(seq_dir, basename + "_depth.npy")
            
            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)
            
            # Process depth map similar to original UnReal4K processing
            sky_mask = depth_map >= 1000
            depth_map[sky_mask] = 0  # sky
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=98, max_depth=200.0
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
        
        set_name = "unreal4k"
        scene_path = scene_path.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "unreal4k",
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


