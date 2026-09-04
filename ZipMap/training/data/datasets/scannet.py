# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import random
import numpy as np
import cv2
from data.dataset_util import *
from data.base_dataset import BaseDataset




class ScanNetDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ScanNet_DIR: str = "/home/haian/Dataset/processed_scannet/",
        min_num_images: int = 24,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 30,
        video_prob: float = 0.6,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.ScanNet_DIR = ScanNet_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        self.video_prob = video_prob
        logging.info(f"ScanNet_DIR is {self.ScanNet_DIR}")
        if split == "train":
            split_folder = "scans_train"
            self.len_train = len_train
        elif split == "test":
            split_folder = "scans_test"
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.split_folder = split_folder
        self._load_data()


    def _load_data(self):
        all_scenes_dict_path = osp.join(self.ScanNet_DIR, self.split_folder, "all_scenes_dict.npz")
        assert osp.exists(all_scenes_dict_path), f"all_scenes_dict.npz not found at {all_scenes_dict_path}. Please run merge_scannet_npz.py first."
        all_scenes_dict = np.load(all_scenes_dict_path, allow_pickle=True)
        all_scenes = all_scenes_dict['scene_names']
            
        if self.debug:
            all_scenes = all_scenes[:100]

        # Preload all scene metadata
        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.scenes = []
        self.start_img_ids = []
        self.scene_img_list = []
        offset = 0
        j = 0

        for i, scene in enumerate(all_scenes):
            scene_data = all_scenes_dict[scene].item()
            imgs = scene_data["images"]
            poses = scene_data["poses"]
            intrins = scene_data["intrinsics"]

            num_imgs = len(imgs)
            if num_imgs < self.min_num_images:
                logging.debug(f"Skipping scene {scene} because it has less than {self.min_num_images} images (num_imgs = {num_imgs})")
                continue

            # Create image IDs for this scene
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = self.min_num_images
            if num_imgs >= cut_off:
                start_img_ids_ = img_ids[:num_imgs - cut_off + 1]
            else:
                start_img_ids_ = []

            if not start_img_ids_:
                logging.debug(f"Skipping scene {scene} because it has no valid sequences")
                continue
                
            self.start_img_ids.extend(start_img_ids_)
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(imgs)
            self.scenes.append(scene)
            self.scene_img_list.append(img_ids)
            self.intrinsics.extend(list(intrins))
            self.trajectories.extend(list(poses))
            
            offset += num_imgs
            j += 1

        # Update dataset length
        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"ScanNet {self.split} data loaded: {len(self.start_img_ids)} sequences and {len(self.images)} images from {len(self.scenes)} scenes.")
        

    def _show_stats(self):
        logging.info(f"ScanNet {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of sequences: {len(self.start_img_ids)}")
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

        # Get start_id and scene image list following Cut3R ScanNet pattern
        start_id = self.start_img_ids[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]

        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.6,
            block_shuffle=24,
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

        scene = self.scenes[scene_id]
        for image_idx in ids:
            scene_dir = osp.join(self.ScanNet_DIR, self.split_folder, scene)
            
            basename = self.images[image_idx]
            
            # Use preloaded camera parameters
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)
            intri = np.array(self.intrinsics[image_idx], dtype=np.float32)
            
            # make it w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB and depth images
            image_filepath = osp.join(scene_dir, "color", basename + ".jpg")
            depth_filepath = osp.join(scene_dir, "depth", basename + ".png")
            
            image = imread_cv2(image_filepath)
            depth_map = imread_cv2(depth_filepath, cv2.IMREAD_UNCHANGED)
            depth_map = depth_map.astype(np.float32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=98
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

        set_name = "scannet"
        seq_name = scene
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "scannet",
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

