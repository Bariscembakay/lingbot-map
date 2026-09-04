# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import os
import random
import logging
import json
import re

from data.dataset_util import *
from data.base_dataset import BaseDataset


def extract_number(filename):
    """Extract number from filename for proper sorting"""
    match = re.search(r'\d+', filename)
    if match:
        return int(match.group())
    return 0

class OmniObject3DDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        OMNIOBJECT3D_DIR: str = "/path/to/omniobject3d/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 100,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.OMNIOBJECT3D_DIR = OMNIOBJECT3D_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = False  # OmniObject3D is not metric initially, but has scale corrections
        logging.info(f"OMNIOBJECT3D_DIR is {self.OMNIOBJECT3D_DIR}")
        
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
        assert os.path.exists(osp.join(self.OMNIOBJECT3D_DIR, "all_scenes_dict.npz")), "Precomputed all_scenes_dict.npz not found. Please run save_omniobject3d_metadata.py first."
        all_scenes_dict = np.load(osp.join(self.OMNIOBJECT3D_DIR, "all_scenes_dict.npz"), allow_pickle=True)
        
        # Load scales from scale.json if available
        scale_file = osp.join(self.OMNIOBJECT3D_DIR, "scale.json")
        if osp.exists(scale_file):
            with open(scale_file, "r") as f:
                self.scales = json.load(f)
        else:
            logging.warning("scale.json not found, using default scale of 1.0")
            self.scales = {}

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
        
        # Fast path: use precomputed metadata
        logging.info("OmniObject3D loading metadata from precomputed all_scenes_dict.npz")
        
        for scene_name in all_scenes_dict.keys():
            scene_data = all_scenes_dict[scene_name].item()  # Convert numpy array back to dict
            
            # Extract data from precomputed metadata
            valid_images = scene_data["images"]  # These are basenames
            sequence_trajectories = scene_data["c2ws"]  # Camera poses (cam2world)
            sequence_intrinsics = scene_data["intrinsics"]  # Camera intrinsics
            
            if len(valid_images) < self.min_num_images:
                logging.debug(f"Skipping {scene_name} - insufficient images ({len(valid_images)})")
                continue
            
            # Add this scene to our dataset
            img_ids = list(np.arange(len(valid_images)) + offset)
            
            num_imgs = len(valid_images)
            cut_off = self.min_num_images
            start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
            
            self.scenes.append(scene_name)
            self.scene_img_list.append(img_ids)
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(valid_images)
            self.trajectories.extend(sequence_trajectories)
            self.intrinsics_list.extend(sequence_intrinsics)
            self.start_img_ids.extend(start_img_ids_)
            
            offset += len(valid_images)
            j += 1
            
            if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                break

        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"OmniObject3D {self.split} data loaded: {len(self.start_img_ids)} start positions from {len(self.scenes)} scenes")

    def _show_stats(self):
        logging.info(f"OmniObject3D {self.split} dataset statistics:")
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
        
        # Use Cut3r-style sampling logic exactly like original
        start_id = self.start_img_ids[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=self.max_interval,
            video_prob=0.0,  # OmniObject3D is not video-like
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
        scene_name = self.scenes[scene_id]
        
        for image_idx in ids:
            basename = self.images[image_idx]
            # Construct full paths
            scene_dir = osp.join(self.OMNIOBJECT3D_DIR, scene_name)
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            
            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world (already corrected with scale)
            
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image and depth map
            image_filepath = osp.join(rgb_dir, basename + ".png")
            depth_filepath = osp.join(depth_dir, basename + ".npy")
            
            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)
            
            # Apply scale correction to depth map (same as in original)
            if scene_name in self.scales:
                scale = self.scales[scene_name]
                depth_map = depth_map / scale / 1000.0
            
            # Process depth map - OmniObject3D specific processing
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            
            # Apply reasonable depth range for object-centric scenes
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=99, max_depth=10.0  # Objects are close
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
        
        set_name = "omniobject3d"
        scene_path = scene_name.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "omniobject3d",
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


