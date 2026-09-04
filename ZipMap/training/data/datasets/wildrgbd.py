# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import numpy as np
import random
import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset


class WildRGBDDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        WildRGBD_DIR: str = "/home/haian/Dataset/wildrgbd/",
        min_num_images: int = 24,
        len_train: int = 20000,
        len_test: int = 1000,
        mask_bg: str = "rand",
        video_prob: float = 0.5,
    ):
        """
        Initialize the WildRGBDDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            WildRGBD_DIR (str): Directory path to WildRGBD data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            mask_bg (str): Background masking strategy ("rand", "True", "False").
        """
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.video_prob = video_prob
        self.WildRGBD_DIR = WildRGBD_DIR
        self.min_num_images = min_num_images
        self.split = split
        self.is_metric = True  # WildRGBD has metric depth
        self.mask_bg = mask_bg
        
        assert mask_bg in ("True", "False", "rand"), f"Invalid mask_bg: {mask_bg}"
        
        logging.info(f"WildRGBD_DIR is {self.WildRGBD_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self._load_data()

    def _load_data(self):
        """Load WildRGBD data from precomputed metadata."""
        metadata_file = osp.join(self.WildRGBD_DIR, f"wildrgbd_metadata_{self.split}.npz")
        if not osp.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Please run preprocessing first.")
        
        logging.info(f"WildRGBD loading data from precomputed wildrgbd_metadata_{self.split}.npz")
        self._load_from_metadata(metadata_file)
        
        self.invalid_scenes = []
        
        logging.info(f"WildRGBD {self.split} data loaded: {len(self.images)} images from {self.num_scenes} scenes")

    def _load_from_metadata(self, metadata_file):
        """Load from precomputed metadata file."""
        data = np.load(metadata_file, allow_pickle=True)
        
        # Initialize data structures
        self.data_dict = {}
        self.all_ref_imgs = []
        self.scenes = []
        self.sceneids = []
        self.images = []
        self.trajectories = []
        self.intrinsics_list = []
        self.id_ranges = []
        
        offset = 0
        j = 0
        for scene_name in data.files:
            if self.debug and j >= 1:
                break
                
            scene_data = data[scene_name].item()
            obj = scene_data["obj"]
            instance = scene_data["instance"]
            images = scene_data["images"]
            c2ws = scene_data["c2ws"]
            intrinsics = scene_data["intrinsics"]
            
            if len(images) < self.min_num_images:
                continue
            
            # Convert c2w to w2c
            w2c_poses = closed_form_inverse_se3(np.array(c2ws))[:, :3]
            
            # Store in dataset format - use same key format as preprocessing script
            scene_key = scene_name
            self.data_dict[scene_key] = {
                "obj": obj,
                "instance": instance,
                "images": images,
                "c2ws": c2ws,
                "intrinsics": intrinsics,
            }
            
            # Create cut_off based on num_views like in original
            cut_off = self.min_num_images
            
            # Add all possible ref views (similar to Cut3R pattern, but limit based on cut_off)
            valid_start_indices = max(0, len(images) - cut_off + 1)
            self.all_ref_imgs.extend(
                [(scene_key, b) for b in range(valid_start_indices)]
            )
            
            self.scenes.append(scene_key)
            self.sceneids.extend([j] * len(images))
            self.images.extend(images)
            self.trajectories.extend(w2c_poses)
            self.intrinsics_list.extend(intrinsics)
            self.id_ranges.append((offset, offset + len(images)))
            
            offset += len(images)
            j += 1
        
        self.num_imgs = len(self.all_ref_imgs)
        self.num_scenes = len(self.data_dict.keys())

        # count obj number
        self.objects = set()
        for scene_key in self.data_dict.keys():
            obj = self.data_dict[scene_key]["obj"]
            self.objects.add(obj)
        self.num_objects = len(self.objects)
        # print(f"Number of objects: {self.num_objects}")

    def _show_stats(self):
        logging.info(f"WildRGBD {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of ref images: {len(self.all_ref_imgs)}")
        # Statistics on scene level
        scene_image_counts = {}
        for scene_id in self.sceneids:
            scene_image_counts[scene_id] = scene_image_counts.get(scene_id, 0) + 1
        image_counts = list(scene_image_counts.values())
        if image_counts:
            min_count = min(image_counts)
            max_count = max(image_counts)
            avg_count = sum(image_counts) / len(image_counts)
            logging.info(f"  Image count per scene: min={min_count}, max={max_count}, avg={avg_count:.2f}")


    def _get_metadatapath(self, obj, instance, view_idx):
        return osp.join(self.WildRGBD_DIR, obj, instance, "metadata", f"{view_idx:05d}.npz")

    def _get_impath(self, obj, instance, view_idx):
        return osp.join(self.WildRGBD_DIR, obj, instance, "rgb", f"{view_idx:05d}.jpg")

    def _get_depthpath(self, obj, instance, view_idx):
        return osp.join(self.WildRGBD_DIR, obj, instance, "depth", f"{view_idx:05d}.png")

    def _get_maskpath(self, obj, instance, view_idx):
        return osp.join(self.WildRGBD_DIR, obj, instance, "masks", f"{view_idx:05d}.png")

    def _read_depthmap(self, depthpath):
        """Read depth map from WildRGBD format."""
        # WildRGBD stores depths in the depth scale of 1000.
        # When we load depth image and divide by 1000, we get depth in meters.
        depthmap = imread_cv2(depthpath, cv2.IMREAD_UNCHANGED)
        depthmap = depthmap.astype(np.float32) / 1000.0
        return depthmap

    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        """
        Retrieve data following WildRGBD's approach.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, len(self.all_ref_imgs) - 1)
        
        # Get ref view info
        scene_key, ref_img_idx = self.all_ref_imgs[seq_index]
        rng = np.random.default_rng()
        
        invalid_seq = True
        while invalid_seq:
            # Skip invalid scenes
            while scene_key in self.invalid_scenes:
                seq_index = rng.integers(low=0, high=len(self.all_ref_imgs))
                scene_key, ref_img_idx = self.all_ref_imgs[seq_index]

            scene_data = self.data_dict[scene_key]
            obj = scene_data["obj"]
            instance = scene_data["instance"]
            images = scene_data["images"]
            c2ws = scene_data["c2ws"]
            intrinsics_data = scene_data["intrinsics"]
            
            if len(images) < img_per_seq:
                self.invalid_scenes.append(scene_key)
                continue

            # Generate image sequence
            imgs_idxs, ordered_video = self.get_seq_from_start_id(
                img_per_seq, ref_img_idx, list(range(len(images))), video_prob=self.video_prob
            )
            
            if imgs_idxs is not None:
                invalid_seq = False
                
        # Decide if we mask the background
        mask_bg = (self.mask_bg == "True") or (
            self.mask_bg == "rand" and rng.choice(2, p=[0.9, 0.1])
        )
        
        target_image_shape = self.get_target_shape(aspect_ratio)
        images_list = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics_list = []
        original_sizes = []
        
        for idx in imgs_idxs:
            # Convert image name back to view index
            view_idx = int(images[idx])
            intri = np.array(intrinsics_data[idx], dtype=np.float32)
            c2w = c2ws[idx]
            
            # Convert c2w to w2c
            w2c = closed_form_inverse_se3(c2w[None])[0]
            
            extri_opencv = w2c[:3]
            
            # Load RGB image and depth map
            image_filepath = self._get_impath(obj, instance, view_idx)
            depth_filepath = self._get_depthpath(obj, instance, view_idx)
            
            image = imread_cv2(image_filepath)
            depth_map = self._read_depthmap(depth_filepath)
            
            if mask_bg:
                # Load and apply object mask
                maskpath = self._get_maskpath(obj, instance, view_idx)
                if osp.exists(maskpath):
                    maskmap = imread_cv2(maskpath, cv2.IMREAD_UNCHANGED).astype(np.float32)
                    maskmap = (maskmap / 255.0) > 0.1
                    depth_map *= maskmap
            
            # Process depth map with percentile thresholding
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=90, max_depth=2.0
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
                logging.error(f"Wrong shape for {scene_key}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
            
            images_list.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics_list.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        
        set_name = "wildrgbd"
        scene_name_str = f"{obj}_{instance.replace('/', '_')}"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_name_str),
            "dataset_name": "wildrgbd",
            "ids": np.array(imgs_idxs),
            "frame_num": len(extrinsics),
            "images": images_list,
            "depths": depths,
            "extrinsics": extrinsics,
            "intrinsics": intrinsics_list,
            "cam_points": cam_points,
            "world_points": world_points,
            "point_masks": point_masks,
            "original_sizes": original_sizes,
        }
        return batch

