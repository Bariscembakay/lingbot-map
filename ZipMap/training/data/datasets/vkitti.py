# Code for ZipMap (CVPR 2026); created by Haian Jin

import os.path as osp
import logging
import random
import glob
import cv2
import numpy as np
from data.dataset_util import *
from data.base_dataset import BaseDataset
from typing import Optional


class VKittiDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        VKitti_DIR: str = "/home/haian/Dataset/vkitti",
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
            VKitti_DIR (str): Directory path to VKitti data.
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
        self.VKitti_DIR = VKitti_DIR
        self.min_num_images = min_num_images
        self.video = True
        self.is_metric = True

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        logging.info(f"VKitti_DIR is {self.VKitti_DIR}")


        # Try to load metadata npz if it exists
        metadata_path = osp.join(self.VKitti_DIR, "vkitti_metadata.npz")
        if osp.exists(metadata_path):
            metadata = np.load(metadata_path, allow_pickle=True)
            self.sequence_list = list(metadata["sequence_list"])
            self.sequence_list_len = len(self.sequence_list)
            self.extrinsics_dict = metadata["extrinsics_dict"].item() if "extrinsics_dict" in metadata else None
            self.intrinsics_dict = metadata["intrinsics_dict"].item() if "intrinsics_dict" in metadata else None
            logging.info(f"Loaded VKitti metadata from {metadata_path}")
        else:
            # Load or generate sequence list as before
            txt_path = osp.join(self.VKitti_DIR, "sequence_list.txt")
            if osp.exists(txt_path):
                with open(txt_path, 'r') as f:
                    sequence_list = [line.strip() for line in f.readlines()]
            else:
                # Generate sequence list and save to txt            
                sequence_list = glob.glob(osp.join(self.VKitti_DIR, "*/*/*/rgb/*"))            
                sequence_list = [file_path.split(self.VKitti_DIR)[-1].lstrip('/') for file_path in sequence_list]
                sequence_list = sorted(sequence_list)

                # Save to txt file
                with open(txt_path, 'w') as f:
                    f.write('\n'.join(sequence_list))

            self.sequence_list = sequence_list
            self.sequence_list_len = len(self.sequence_list)
            self.extrinsics_dict = None
            self.intrinsics_dict = None

        self.depth_max = 80

        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: VKitti Data has {self.sequence_list_len} sequences")
    

    def _show_stats(self):
        # for each sequence, count the image num, and plot a histogram
        image_num = dict()
        for seq_name in self.sequence_list:
            num_images = len(glob.glob(osp.join(self.VKitti_DIR, seq_name, "rgb_*.jpg")))
            image_num[seq_name] = num_images
        from matplotlib import pyplot as plt
        # Plot histogram of image counts
        plt.figure(figsize=(10, 5))
        plt.bar(image_num.keys(), image_num.values())
        plt.xlabel("Sequence Name")
        plt.ylabel("Number of Images")
        plt.title("VKitti Dataset: Number of Images per Sequence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # save
        plt.savefig("vkitti_image_count_histogram.png")
        mean = np.mean(list(image_num.values()))
        max = np.max(list(image_num.values()))
        min = np.min(list(image_num.values()))
        logging.info(f"Mean number of images per sequence: {mean}")
        total_image_num  = np.sum(list(image_num.values()))
        logging.info(f"Total number of images in dataset: {total_image_num}")

    def get_data(
        self,
        seq_index: Optional[int] = None,
        img_per_seq: Optional[int] = None,
        seq_name: Optional[str] = None,
        ids: Optional[list] = None,
        aspect_ratio: float = 1.0,
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

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]


        camera_id = int(seq_name[-1])

        # Use metadata if available
        if self.extrinsics_dict is not None and seq_name in self.extrinsics_dict:
            camera_parameters = self.extrinsics_dict[seq_name]
            if self.intrinsics_dict is not None and seq_name in self.intrinsics_dict:
                camera_intrinsic = self.intrinsics_dict[seq_name]
            else:
                camera_intrinsic = None
        else:
            # Fallback to old behavior
            try:
                camera_parameters = np.loadtxt(
                    osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "extrinsic.txt"), 
                    delimiter=" ", 
                    skiprows=1
                )
                camera_parameters = camera_parameters[camera_parameters[:, 1] == camera_id]

                camera_intrinsic = np.loadtxt(
                    osp.join(self.VKitti_DIR, "/".join(seq_name.split("/")[:2]), "intrinsic.txt"), 
                    delimiter=" ", 
                    skiprows=1
                )
                camera_intrinsic = camera_intrinsic[camera_intrinsic[:, 1] == camera_id]
            except Exception as e:
                logging.error(f"Error loading camera parameters for {seq_name}: {e}")
                raise

        num_images = len(camera_parameters) 
        if ids is None:
            ids = np.random.choice(num_images, img_per_seq, replace=self.allow_duplicate_img)

        if self.get_nearby:
            ids = self.get_nearby_ids(ids, num_images, expand_ratio=self.expand_ratio)

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
            image_filepath = osp.join(self.VKitti_DIR, seq_name, f"rgb_{image_idx:05d}.jpg")
            depth_filepath = osp.join(self.VKitti_DIR, seq_name, f"depth_{image_idx:05d}.png").replace("/rgb", "/depth")

            image = read_image_cv2(image_filepath)
            depth_map = cv2.imread(depth_filepath, cv2.IMREAD_ANYCOLOR | cv2.IMREAD_ANYDEPTH)
            depth_map = depth_map / 100
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(depth_map, max_percentile=-1, min_percentile=-1, max_depth=self.depth_max)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            # Process camera matrices
            extri_opencv = camera_parameters[image_idx][-16:].reshape(4, 4)
            extri_opencv = extri_opencv[:3]

            intri_opencv = np.eye(3)
            intri_opencv[0, 0] = camera_intrinsic[image_idx][-4]
            intri_opencv[1, 1] = camera_intrinsic[image_idx][-3]
            intri_opencv[0, 2] = camera_intrinsic[image_idx][-2]
            intri_opencv[1, 2] = camera_intrinsic[image_idx][-1]

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

            if (image.shape[:2] != target_image_shape).any():
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

        set_name = "virtual_kitti"
        batch = {
            "seq_name": set_name + "_" + seq_name.replace("/", "_"),
            "dataset_name": "virtual_kitti",
            "ids": ids,
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



