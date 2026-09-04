# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import os
import random
import logging

from data.dataset_util import *
from data.base_dataset import BaseDataset


class MP3DDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        MP3D_DIR: str = "/path/to/mp3d/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 100,  # Not used for MP3D since it uses overlap-based sampling
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.MP3D_DIR = MP3D_DIR
        self.max_interval = max_interval  # Not used but kept for consistency
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True  # MP3D is metric
        logging.info(f"MP3D_DIR is {self.MP3D_DIR}")
        
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
        assert os.path.exists(osp.join(self.MP3D_DIR, "all_scenes_dict.npz")), "Precomputed all_scenes_dict.npz not found. Please run save_mp3d_metadata.py first."
        all_scenes_dict = np.load(osp.join(self.MP3D_DIR, "all_scenes_dict.npz"), allow_pickle=True)

        # Initialize data structures - MP3D uses different structure than other datasets
        self.scenes = []  # Scene names
        self.images = []  # List of tuples (scene, img_idx, basename)
        self.trajectories = {}  # Dict mapping scene -> list of camera poses
        self.intrinsics_list = {}  # Dict mapping scene -> list of intrinsics
        self.overlaps = {}  # Dict mapping scene -> overlap matrix
        self.scene_img_list = {}  # Dict mapping scene -> global image indices
        
        offset = 0
        j = 0
        
        # Fast path: use precomputed metadata
        logging.info("MP3D loading metadata from precomputed all_scenes_dict.npz")
        
        for scene_name in all_scenes_dict.keys():
            scene_data = all_scenes_dict[scene_name].item()  # Convert numpy array back to dict
            
            # Extract data from precomputed metadata
            valid_images = scene_data["images"]  # These are basenames
            sequence_trajectories = scene_data["c2ws"]  # Camera poses (cam2world)
            sequence_intrinsics = scene_data["intrinsics"]  # Camera intrinsics
            overlap_matrix = scene_data["overlap"]  # MP3D specific overlap matrix
            
            if len(valid_images) < self.min_num_images:
                logging.debug(f"Skipping {scene_name} - insufficient images ({len(valid_images)})")
                continue
            
            # Store MP3D-specific data structures
            self.scenes.append(scene_name)
            self.trajectories[scene_name] = sequence_trajectories
            self.intrinsics_list[scene_name] = sequence_intrinsics  
            self.overlaps[scene_name] = overlap_matrix
            
            num_imgs = len(valid_images)
            scene_img_indices = np.arange(num_imgs) + offset
            self.scene_img_list[scene_name] = scene_img_indices
            
            # Store images as (scene, local_img_idx, basename) tuples like original
            for i, basename in enumerate(valid_images):
                self.images.append((scene_name, i, basename))
            
            offset += num_imgs
            j += 1
            
            if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                break

        self.sequence_list_len = len(self.images)
        logging.info(f"MP3D {self.split} data loaded: {len(self.images)} images from {len(self.scenes)} scenes")

    def _show_stats(self):
        logging.info(f"MP3D {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        # Statistics on scene level
        scene_image_counts = [len(self.scene_img_list[scene]) for scene in self.scenes]
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
        
        # Use MP3D-specific overlap-based sampling logic
        num_views_possible = 0
        num_unique = img_per_seq 
        
        rng = np.random.RandomState()  # Create RNG for sampling
        
        # Find a valid starting image with enough overlapping views
        max_attempts = 100
        attempts = 0
        while num_views_possible < num_unique - 1 and attempts < max_attempts:
            if seq_index is None or attempts > 0:
                seq_index = random.randint(0, self.sequence_list_len - 1)
            
            scene, img_idx, _ = self.images[seq_index]
            overlap = self.overlaps[scene]
            
            # Find overlapping images for this reference image
            sel_img_idx = np.where(overlap[:, 0] == img_idx)[0]
            overlap_sel = overlap[sel_img_idx]
            overlap_sel = overlap_sel[
                (overlap_sel[:, 2] > 0.01) & (overlap_sel[:, 2] < 1)
            ]
            num_views_possible = len(overlap_sel)

            attempts += 1
        ref_id = self.scene_img_list[scene][img_idx]
        overlapping_ids = self.scene_img_list[scene][overlap_sel[:, 1].astype(np.int64)]
        if num_views_possible < num_unique - 1:
            logging.warning(f"Could not find enough overlapping views for scene {scene}, using what's available")
            # Fallback to any available images in the scene
            replace = True
        else:
            if np.random.rand() < 0.5:
                replace = False if not self.allow_duplicate_img else True
            else:
                replace = False
        weights = overlap_sel[:, 2] / np.sum(overlap_sel[:, 2])
        
        sampled_ids = rng.choice(
            overlapping_ids,
            img_per_seq - 1,
            replace=replace,
            p=weights,
        )
        global_image_idxs = np.concatenate([[ref_id], sampled_ids])

        # Convert global indices back to local indices within scene
        scene_offset = self.scene_img_list[scene][0]
        selected_local_indices = global_image_idxs - scene_offset
    
        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        
        for local_idx in selected_local_indices:
            local_idx = int(local_idx)  # Ensure it's an integer
            basename = self.images[self.scene_img_list[scene][0] + local_idx][2]  # Get basename from images list
            
            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[scene][local_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[scene][local_idx], dtype=np.float32)  # cam2world
            
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image and depth map
            scene_dir = osp.join(self.MP3D_DIR, scene)
            image_filepath = osp.join(scene_dir, "rgb", basename + ".png")
            depth_filepath = osp.join(scene_dir, "depth", basename + ".npy")
            
            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)
            
            # Process depth map - MP3D specific processing
            depth_map[~np.isfinite(depth_map)] = 0  # invalid
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            
            # Apply reasonable depth range for indoor scenes
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=99, max_depth=20.0  # Indoor scenes
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
                logging.error(f"Wrong shape for {scene}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
                
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        
        set_name = "mp3d"
        scene_path = scene.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "mp3d",
            "ids": np.array(selected_local_indices),
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


