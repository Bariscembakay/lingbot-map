# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import numpy as np
import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset



class DL3DVDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        DL3DV_DIR: str = "/home/haian/Dataset/dl3dv/",
        min_num_images: int = 24,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 20,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.DL3DV_DIR = DL3DV_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = False
        self.video_prob = video_prob    
        logging.info(f"DL3DV_DIR is {self.DL3DV_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._load_data()

    def _load_data(self):
        # Optional pre-merged dict for faster loading
        all_scenes_dict_path = osp.join(self.DL3DV_DIR, "all_scenes_dict.npz")
        all_scenes_dict_exists = osp.exists(all_scenes_dict_path)

        assert all_scenes_dict_exists, f"Missing scene dictionary at {all_scenes_dict_path}"
        all_scenes_dict = np.load(all_scenes_dict_path, allow_pickle=True)
        all_scenes = all_scenes_dict['scene_names']


        if self.debug:
            all_scenes = all_scenes[:50]


        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.scenes = []  # actually loaded scenes
        self.start_img_ids = []
        self.scene_img_list = []
        
        offset = 0
        j = 0

        for i, scene in enumerate(all_scenes):
            # Load per-scene metadata
            if scene not in all_scenes_dict:
                logging.warning(f"Scene {scene} not found in all_scenes_dict, skipping.")
                continue

            scene_data = all_scenes_dict[scene].item()
            imgs = scene_data["images"]
            intrins = scene_data["intrinsics"]
            traj = scene_data["trajectories"]
        

            num_imgs = len(imgs)
            if num_imgs < self.min_num_images:
                logging.debug(
                    f"Skipping {scene} because it has less than {self.min_num_images} images (num_imgs = {num_imgs})"
                )
                continue

            # Create sequence of all images for this scene
            img_ids = np.arange(num_imgs) + offset
            cut_off = self.min_num_images
            start_img_ids = img_ids[: max(0, num_imgs - cut_off + 1)]
            
            if len(start_img_ids) == 0:
                logging.debug(f"Skipping scene {scene} because it has no valid start positions")
                continue

            # Append scene data
            self.scenes.append(scene)
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(list(imgs))
            self.intrinsics.extend(list(intrins))
            self.trajectories.extend(list(traj))
            self.start_img_ids.extend(start_img_ids)
            self.scene_img_list.append(list(img_ids))
            offset += num_imgs
            j += 1

        # Finalization
        self.sequence_list = self.start_img_ids
        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"DL3DV {self.split} data loaded: {len(self.sequence_list)} sequences and {len(self.images)} images from {len(self.scenes)} scenes.")  

    def _show_stats(self):
        logging.info(f"DL3DV {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of sequences: {len(self.sequence_list)}")
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
            seq_index = np.random.randint(0, self.sequence_list_len)

        start_id = self.sequence_list[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]


        # Video-like sampling following the original pattern
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=self.max_interval,
            video_prob=self.video_prob,  
            fix_interval_prob=0.5,
            block_shuffle=25,
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
        scene_dir = osp.join(self.DL3DV_DIR, scene, "dense")

        for image_idx in ids:
            

            intri = np.float32(self.intrinsics[image_idx])
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to OpenCV w2c extrinsic [3x4]
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            rgb_path = self.images[image_idx]
            basename = rgb_path[:-4]
            image_filepath = osp.join(scene_dir, "rgb", rgb_path)
            depth_filepath = osp.join(scene_dir, "depth", basename + ".npy")
            sky_mask_filepath = osp.join(scene_dir, "sky_mask", rgb_path)
            outlier_mask_filepath = osp.join(scene_dir, "outlier_mask", rgb_path)

            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)
            depth_map[~np.isfinite(depth_map)] = 0.0
            
            # Apply sky and outlier masks if they exist
            if osp.exists(sky_mask_filepath):
                sky_mask = cv2.imread(sky_mask_filepath, cv2.IMREAD_UNCHANGED) >= 127
                depth_map[sky_mask] = 0.0
            
            if osp.exists(outlier_mask_filepath):
                outlier_mask = cv2.imread(outlier_mask_filepath, cv2.IMREAD_UNCHANGED)
                depth_map[outlier_mask >= 110] = 0.0
            
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=90)
 

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
                    f"Wrong shape for {scene}: expected {target_image_shape}, got {image.shape[:2]}"
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

        # seq_name = scene_dir
        seq_name = scene_dir.split("/")[-3] + "_"+ scene_dir.split("/")[-2]
        set_name = "dl3dv"

        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "dl3dv",
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


