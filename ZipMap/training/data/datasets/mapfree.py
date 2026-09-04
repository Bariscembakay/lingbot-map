# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional


import os
import os.path as osp
import logging
import random
import numpy as np
import cv2
from data.dataset_util import *
from data.base_dataset import BaseDataset




class MapFreeDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MapFree_DIR: str = "/home/haian/Dataset/MapFree/",
        min_num_images: int = 24,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 30,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.MapFree_DIR = MapFree_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        logging.info(f"MapFree_DIR is {self.MapFree_DIR}")
        self.video_prob = video_prob
        # MapFree doesn't have train/test splits in the same way, use the full dataset
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def imgid2path(self, img_id, scene):
        """Convert image ID to file path"""
        first_seq_id, first_frame_id = img_id
        return osp.join(
            self.MapFree_DIR,
            scene,
            f"dense{first_seq_id}",
            "rgb",
            f"frame_{first_frame_id:05d}.jpg",
        )

    def _load_data(self):
        cache_file = f"{self.MapFree_DIR}/cached_metadata_48_col_only.npz"
        # cache_file = f"{self.MapFree_DIR}/cached_metadata_4.npz"
        assert osp.exists(cache_file), f"cached_metadata_48_col_only.npz not found at {cache_file}. Please ensure the cache file exists."

        logging.info(f"Loading cached MapFree metadata from {cache_file}")
        with np.load(cache_file) as data:
            self.scenes = list(data["scenes"])
            self.sceneids = data["sceneids"]
            self.scope = data["scope"]
            self.video_flags = data["video_flags"]
            self.groups = data["groups"]
            self.id_ranges = data["id_ranges"]
            self.images = data["images"]

        if self.debug:
            # Limit to first 10 groups for debugging
            debug_limit = min(100, len(self.scope))
            self.scope = self.scope[:debug_limit]
            self.sceneids = self.sceneids[:debug_limit]
            self.video_flags = self.video_flags[:debug_limit]
            if debug_limit < len(self.scope):
                self.groups = self.groups[:self.scope[debug_limit-1] + (self.scope[debug_limit] - self.scope[debug_limit-1]) if debug_limit < len(self.scope) else len(self.groups)]
            self.id_ranges = self.id_ranges[:debug_limit]

        # Update dataset length
        self.sequence_list_len = len(self.sceneids)
        logging.info(f"MapFree {self.split} data loaded: {len(self.scope)} image groups from {len(self.scenes)} scenes.")

    def _show_stats(self):
        logging.info(f"MapFree {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of image groups: {len(self.scope)}")
        logging.info(f"  Number of total images: {len(self.images)}")
        # Statistics on video vs collection
        video_count = np.sum(self.video_flags)
        collection_count = len(self.video_flags) - video_count
        logging.info(f"  Video sequences: {video_count}")
        logging.info(f"  Image collections: {collection_count}")

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

        scene = self.scenes[self.sceneids[seq_index]]
        
        # Use the original Cut3R logic for selecting images
        if random.random() < 0.6:
            # Video sequence mode
            chosed_ids = np.arange(self.id_ranges[seq_index][0], self.id_ranges[seq_index][1])
            cut_off = img_per_seq if not self.allow_duplicate_img else max(img_per_seq // 3, 3)
            if cut_off > len(chosed_ids):
                cut_off = len(chosed_ids)
            start_ids = chosed_ids[:len(chosed_ids) - cut_off + 1]
            start_id = random.choice(start_ids)
            pos, ordered_video = self.get_seq_from_start_id(
                img_per_seq,
                start_id,
                chosed_ids.tolist(),
                max_interval=self.max_interval,
                video_prob=self.video_prob,
                fix_interval_prob=0.5,
                block_shuffle=24,
            )
            chosed_ids = np.array(chosed_ids)[pos]
            image_idxs = self.images[chosed_ids]
        else:
            # Image collection mode
            ordered_video = False
            seq_start_index = self.scope[seq_index]
            seq_end_index = self.scope[seq_index + 1] if seq_index < len(self.scope) - 1 else None
            image_idxs = (
                self.groups[seq_start_index:seq_end_index]
                if seq_end_index is not None
                else self.groups[seq_start_index:]
            )
            image_idxs, overlap_scores = image_idxs[:, :2], image_idxs[:, 2]
            replace = (
                True
                if self.allow_duplicate_img
                or len(overlap_scores[overlap_scores > 0]) < img_per_seq
                else False
            )
            rng = np.random.default_rng()
            image_idxs = rng.choice(
                image_idxs,
                img_per_seq,
                replace=replace,
                p=overlap_scores / np.sum(overlap_scores),
            )
            image_idxs = image_idxs.astype(np.int64)

        
        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in image_idxs:
            img_path = self.imgid2path(image_idx, scene)
            depth_path = img_path.replace("rgb", "depth").replace(".jpg", ".npy")
            cam_path = img_path.replace("rgb", "cam").replace(".jpg", ".npz")
            sky_mask_path = img_path.replace("rgb", "sky_mask")
            
            # Load RGB image
            image = imread_cv2(img_path)
            
            # Load depth map
            depth_map = np.load(depth_path)
            
            # Load camera parameters
            camera_params = np.load(cam_path)
            camera_pose = camera_params["pose"].astype(np.float32)  # cam2world
            intri = camera_params["intrinsic"].astype(np.float32)
            
            # Load sky mask and apply to depth
            sky_mask = cv2.imread(sky_mask_path, cv2.IMREAD_UNCHANGED) >= 127
            depth_map[sky_mask] = -1.0
            depth_map[depth_map > 400.0] = 0.0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=90)
            # Convert cam2world to w2c for consistency with ScanNet format
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
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
                filepath=img_path,
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

        set_name = "mapfree"
        seq_name = scene
        # Convert chosed_ids to simple integer indices for tensor conversion
        if 'chosed_ids' in locals():
            ids_array = np.array(chosed_ids, dtype=np.int64)
        else:
            # For image collection mode, use simple sequential indices
            ids_array = np.arange(len(image_idxs), dtype=np.int64)

        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "mapfree",
            "ids": ids_array,
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


