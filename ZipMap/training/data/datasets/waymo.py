# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import numpy as np

from data.dataset_util import *
from data.base_dataset import BaseDataset
from zipmap.utils.geometry import closed_form_inverse_se3



class WaymoDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        Waymo_DIR: str = "/home/haian/Dataset/waymo/",
        min_num_images: int = 24,
        len_train: int = 50000,
        len_test: int = 5000,
        max_interval: int = 6,
        video_prob: float = 0.9,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.Waymo_DIR = Waymo_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        self.video_prob = video_prob
        logging.info(f"Waymo_DIR is {self.Waymo_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._load_data()

    def _load_data(self):
        # Load from precomputed metadata file
        metadata_file = osp.join(self.Waymo_DIR, "waymo_metadata.npz")
        if not osp.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Please run preprocessing first.")
        
        logging.info("Waymo loading data from precomputed waymo_metadata.npz")
        all_sequences_dict = np.load(metadata_file, allow_pickle=True)
        all_sequences = list(all_sequences_dict.files)

        if self.debug:
            all_sequences = all_sequences[:20]

        self.sequence_list = []
        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.scenes = []  # actually loaded sequences
        self.scene_img_list = []
        self.start_img_ids = []

        offset = 0
        j = 0

        for i, sequence_key in enumerate(all_sequences):
            sequence_data = all_sequences_dict[sequence_key].item()
            imgs = sequence_data["images"]
            c2ws = sequence_data["c2ws"]
            intrins = sequence_data["intrinsics"]
            scene = sequence_data["scene"]
            seq_id = sequence_data["seq_id"]

            num_imgs = len(imgs)
            if num_imgs < self.min_num_images:
                logging.debug(
                    f"Skipping {sequence_key} because it has less than {self.min_num_images} images (num_imgs = {num_imgs})"
                )
                continue

            # Store c2w poses directly, convert to w2c later in get_data
            c2w_poses = c2ws

            # Create image IDs for this sequence
            img_ids = list(np.arange(num_imgs) + offset)
            cut_off = self.min_num_images
            start_img_ids_ = img_ids[: max(0, num_imgs - cut_off + 1)]

            if len(start_img_ids_) == 0:
                logging.debug(f"Skipping sequence {sequence_key} because it has no valid starting positions")
                continue

            # Append sequence data
            self.scenes.append((scene, seq_id))
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(list(imgs))
            self.intrinsics.extend(list(intrins))
            self.trajectories.extend(list(c2w_poses))  # Store c2w instead of w2c
            self.start_img_ids.extend(start_img_ids_)
            self.scene_img_list.append(img_ids)

            offset += num_imgs
            j += 1

        # Finalization
        self.sequence_list = self.start_img_ids
        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"Waymo {self.split} data loaded: {len(self.sequence_list)} start positions from {len(self.scenes)} sequences")

    def _show_stats(self):
        logging.info(f"Waymo {self.split} dataset statistics:")
        logging.info(f"  Number of sequences: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of start positions: {len(self.sequence_list)}")
        # Statistics on sequence level
        sequence_image_counts = {}
        for scene_id in self.sceneids:
            sequence_image_counts[scene_id] = sequence_image_counts.get(scene_id, 0) + 1
        image_counts = list(sequence_image_counts.values())
        if image_counts:
            min_count = min(image_counts)
            max_count = max(image_counts)
            avg_count = sum(image_counts) / len(image_counts)
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
            seq_index = np.random.randint(0, self.sequence_list_len)

        start_id = self.start_img_ids[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        scene, seq_id = self.scenes[scene_id]
        
        # Adjust max_interval based on seq_id (following original logic)
        max_interval = self.max_interval // 2 if seq_id == "4" else self.max_interval
        
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.9,
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

        scene, seq_id = self.scenes[scene_id]
        scene_dir = osp.join(self.Waymo_DIR, scene)

        for image_idx in ids:

            intri = np.float32(self.intrinsics[image_idx])
            # Convert c2w to w2c at runtime
            c2w_pose = np.array(self.trajectories[image_idx])  # c2w from stored data
            extri_opencv = closed_form_inverse_se3(c2w_pose[None])[0, :3] # w2c

            impath = self.images[image_idx]
            image_filepath = osp.join(scene_dir, impath + ".jpg")
            depth_filepath = osp.join(scene_dir, impath + ".exr")

            image = imread_cv2(image_filepath)
            depth_map = imread_cv2(depth_filepath)
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98, max_depth=80.0)
            
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
                    f"Wrong shape for {scene}_{seq_id}: expected {target_image_shape}, got {image.shape[:2]}"
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

        set_name = "waymo"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else f"{scene}_{seq_id}_{seq_index}"),
            "dataset_name": "waymo",
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

