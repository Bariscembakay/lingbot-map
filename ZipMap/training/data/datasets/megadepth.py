# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional


import os.path as osp
import logging
import random

import numpy as np
from data.dataset_util import *
from data.base_dataset import BaseDataset
import json


class MegaDepthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        MegaDepth_DIR: str = "/home/haian/Dataset/processed_megadepth/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        expand_ratio: int = 8,
    ):
        """
        Initialize the VKittiDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            MegaDepth_DIR (str): Directory path to MegaDepth data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            expand_range (int): Range for expanding nearby image selection.
            get_nearby_thres (int): Threshold for nearby image selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.expand_ratio = expand_ratio
        self.MegaDepth_DIR = MegaDepth_DIR
        self.min_num_images = min_num_images
        self.split = split
        logging.info(f"MegaDepth_DIR is {self.MegaDepth_DIR}")

        # read metadata
        with np.load(
            osp.join(self.MegaDepth_DIR, "megadepth_sets_64.npz"), allow_pickle=True
        ) as data:
            self.all_scenes = data["scenes"]
            self.all_images = data["images"]
            self.sets = data["sets"]
        if osp.exists(osp.join(self.MegaDepth_DIR, "megadepth_pose_metadata.json")):
            with open(osp.join(self.MegaDepth_DIR, "megadepth_pose_metadata.json"), "r") as f:
                self.pose_metadata = json.load(f)
        else:
            print(f"No pose metadata found in {osp.join(self.MegaDepth_DIR, 'megadepth_pose_metadata.json')}")  
            self.pose_metadata = None

        if split == "train":
            self.len_train = len_train
            # 'opposite= True' means we select all scenes except '0015' and '0022'
            self.select_scene(("0015", "0022"), opposite=True) 
        elif split == "test":
            self.len_train = len_test
            self.select_scene(("0015", "0022"))
        else:
            raise ValueError(f"Invalid split: {split}")
        self.sequence_list = self.sets
        self.sequence_list_len = len(self.sequence_list)

        logging.info(f"MegaDepth {self.split} data loaded: {len(self.sequence_list)} sequences")

    def _show_stats(self):
        logging.info(f"MegaDepth {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(np.unique(self.sets[:, 0]))}")
        logging.info(f"  Number of sequences: {len(self.sequence_list)}")
        # Statistics on sequence level
        sequence_image_counts = [np.count_nonzero(seq[1:65]) for seq in self.sets]
        if sequence_image_counts:
            min_count = min(sequence_image_counts)
            max_count = max(sequence_image_counts)
            avg_count = sum(sequence_image_counts) / len(sequence_image_counts)
            logging.info(f"  Image count per sequence: min={min_count}, max={max_count}, avg={avg_count:.2f}")
        # Scene distribution
        scene_counts = {}
        for scene_id in self.sets[:, 0]:
            scene = self.all_scenes[scene_id]
            scene_counts[scene] = scene_counts.get(scene, 0) + 1
        logging.info(f"  Top 5 scenes by sequence count: {sorted(scene_counts.items(), key=lambda x: x[1], reverse=True)[:5]}")

        


    def select_scene(self, scene, *instances, opposite=False):
        scenes = (scene,) if isinstance(scene, str) else tuple(scene)
        scene_id = [s.startswith(scenes) for s in self.all_scenes]
        assert any(scene_id), "no scene found"
        valid = np.in1d(self.sets[:, 0], np.nonzero(scene_id)[0])
        if instances:
            raise NotImplementedError("selecting instances not implemented")
        if opposite:
            valid = ~valid
        assert valid.any()
        self.sets = self.sets[valid]
        self.sequence_list = self.sets
        self.sequence_list_len = len(self.sequence_list)


    def get_data(
        self,
        seq_index = None,
        img_per_seq = 24,
        seq_name = None,
        ids = None,
        aspect_ratio = 1.0,
    ) -> dict:
        """
        Retrieve data for a specific sequence.

        Args:
            seq_index (int): Index of the sequence to retrieve.
            img_per_seq (int): Number of images per sequence.
            seq_name (str): Name of the sequence.
            ids (list): Specific IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)
        scene_id = self.sequence_list[seq_index][0]
        image_idxs = self.sets[seq_index][1:65]
        scene, subscene = self.all_scenes[scene_id].split()
        if seq_name is None:
            seq_name = f"{scene}_{subscene}"
        seq_path = osp.join(self.MegaDepth_DIR, scene, subscene)
        if ids is None:
            ids = np.random.choice(image_idxs, img_per_seq, replace=self.allow_duplicate_img).tolist()
        # if self.get_nearby:
        #     ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio).tolist()

        target_image_shape = self.get_target_shape(aspect_ratio)

        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []

        for image_idx in ids:
            img = self.all_images[image_idx]
            try:
                image_filepath = osp.join(seq_path, img + ".jpg")
                image = imread_cv2(image_filepath)
                depth_map = imread_cv2(osp.join(seq_path, img + ".exr"))
                depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=95
                )
                if self.pose_metadata is None:
                    # read from npz file
                    camera_params = np.load(osp.join(seq_path, img + ".npz"))
                    intri_opencv = np.float32(camera_params["intrinsics"])
                    c2w_camera_pose = np.float32(camera_params["cam2world"])  # 4x4
                    w2c_camera_pose = closed_form_inverse_se3(c2w_camera_pose[None])[0]
                else:
                    # Preferred; Save IO overhead
                    camera_params = self.pose_metadata[str(image_idx)]
                    intri_opencv = np.array(camera_params["intrinsics"])
                    w2c_camera_pose = np.array(camera_params["w2c"])
            except Exception as e:
                raise OSError(f"cannot load {img}, got exception {e}")


            extri_opencv = w2c_camera_pose[:3, :] 
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
                intri_opencv,
                original_size,
                target_image_shape,
                filepath=image_filepath,
            )

            if not np.array_equal(image.shape[:2], target_image_shape):
                logging.error(f"Wrong shape for {seq_name}: expected {target_image_shape}, got {image.shape[:2]}")
                continue

            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        set_name = "megadepth"
        batch = {
            "seq_name": set_name + "_" + seq_name,
            "dataset_name": "megadepth",
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
    



