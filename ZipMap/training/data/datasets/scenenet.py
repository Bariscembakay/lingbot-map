# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import random
from PIL import Image
import numpy as np
import pickle
import math

from data.dataset_util import *
from data.base_dataset import BaseDataset
from typing import Optional


def read_pkl_array(filename):
    """Read pickle array from file."""
    with open(filename, 'rb') as f:
        array = pickle.load(f)
    return array


def load_depth_map_in_m(file_name):
    """Load depth map from PNG file and convert to meters."""
    image = Image.open(file_name)
    pixel = np.array(image)
    return (pixel * 0.001)





def pixel_to_ray(pixel,vfov=45,hfov=60,pixel_width=320,pixel_height=240):
    x, y = pixel
    x_vect = math.tan(math.radians(hfov/2.0)) * ((2.0 * ((x+0.5)/pixel_width)) - 1.0)
    y_vect = math.tan(math.radians(vfov/2.0)) * ((2.0 * ((y+0.5)/pixel_height)) - 1.0)
    return (x_vect,y_vect,1.0)

def normalize(v):
    return v/np.linalg.norm(v)

def normalised_pixel_to_ray_array(width=320,height=240):
    pixel_to_ray_array = np.zeros((height,width,3))
    for y in range(height):
        for x in range(width):
            pixel_to_ray_array[y,x] = normalize(np.array(pixel_to_ray((x,y),pixel_height=height,pixel_width=width)))
    return pixel_to_ray_array

class SceneNetDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        SceneNet_DIR: str = "/path/to/scenergbd_net/",
        min_num_images: int = 24,
        len_train: int = 200000,
        len_test: int = 1000,
        max_interval: int = 6,
        video_prob: float = 0.6
    ):
        """
        Initialize the SceneNetDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            SceneNet_DIR (str): Directory path to SceneNet data.
            meta (str): Path to metadata npz file.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
            max_interval (int): Maximum interval for view selection.
        """
        super().__init__(common_conf=common_conf)

        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img

        self.max_interval = max_interval
        self.SceneNet_DIR = SceneNet_DIR
        self.min_num_images = min_num_images
        self.video_prob = video_prob

        self.video = True
        self.is_metric = True

        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self.split = split

        logging.info(f"SceneNet_DIR is {self.SceneNet_DIR}")

        meta_data_path = osp.join(self.SceneNet_DIR, 'SceneNet_meta.npz')

        meta_path = osp.abspath(meta_data_path)
        if not osp.exists(meta_path):
            raise FileNotFoundError(f"Metadata file not found: {meta_path}")

        logging.info(f"Loading SceneNet metadata from {meta_path}")
        metadata = np.load(meta_path, allow_pickle=True)
        self.meta = metadata['arr_0']
        self.scenes = self.meta.item()[split]

        # Build sequence list and image indices
        self.sequence_list = []
        self.image_indices = []

        for scene_id in sorted(self.scenes.keys()):
            scene_data = self.scenes[scene_id]
            # Get all available frame IDs (exclude 'intrinsic_matrix' key)
            frame_ids = [k for k in scene_data.keys() if k != 'intrinsic_matrix']

            if len(frame_ids) < self.min_num_images:
                continue

            self.sequence_list.append(scene_id)
            self.image_indices.append(sorted(frame_ids))

        self.sequence_list_len = len(self.sequence_list)
        self.cached_pixel_to_ray_array = normalised_pixel_to_ray_array().astype(np.float32)


        status = "Training" if self.training else "Testing"
        logging.info(f"{status}: SceneNet Data has {self.sequence_list_len} scenes")

    def _show_stats(self):
        """Show statistics about the dataset based on loaded metadata."""
        # Calculate number of images per sequence from metadata
        image_num = dict()
        for i, scene_name in enumerate(self.sequence_list):
            num_images = len(self.image_indices[i])
            image_num[scene_name] = num_images

        from matplotlib import pyplot as plt

        plt.figure(figsize=(10, 5))
        plt.bar(range(len(image_num)), list(image_num.values()))
        plt.xlabel("Sequence Index")
        plt.ylabel("Number of Images")
        plt.title("SceneNet Dataset: Number of Images per Sequence")
        plt.tight_layout()
        plt.savefig("scenenet_image_count_histogram.png")

        mean = np.mean(list(image_num.values()))
        max_val = np.max(list(image_num.values()))
        min_val = np.min(list(image_num.values()))
        logging.info(f"Mean number of images per sequence: {mean:.2f}")
        logging.info(f"Max number of images per sequence: {max_val}")
        logging.info(f"Min number of images per sequence: {min_val}")
        total_image_num = np.sum(list(image_num.values()))
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
            seq_name (str): Name of the sequence (scene_id).
            ids (list): Specific frame IDs to retrieve.
            aspect_ratio (float): Aspect ratio for image processing.

        Returns:
            dict: A batch of data including images, depths, and other metadata.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, self.sequence_list_len - 1)

        if seq_name is None:
            seq_name = self.sequence_list[seq_index]

        # Get available frame IDs for this sequence
        image_idxs = self.image_indices[seq_index]

        # Use get_seq_from_start_id for sampling (similar to omniworld_game)
        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]

        start_id = np.random.choice(start_image_idxs)

        # SceneNet is video-like; use get_seq_from_start_id
        pos, _ = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            image_idxs,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
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

        seq_path = osp.join(self.SceneNet_DIR, seq_name)
        scene_data = self.scenes[seq_name]

        for frame_id in ids:
            # Load image
            image_filepath = osp.join(seq_path, 'photo', f'{frame_id}.jpg')
            image = read_image_cv2(image_filepath)

            # Load depth
            depth_filepath = osp.join(seq_path, 'depth', f'{frame_id}.png')
            depth_map = load_depth_map_in_m(depth_filepath).astype(np.float32)

            # Convert depth using pixel-to-ray array (SceneNet specific)
            # This converts from ray distance to Z-depth
            depth_map = depth_map * self.cached_pixel_to_ray_array[:, :, 2]
            depth_map[np.isinf(depth_map)] = 0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=99)

            assert image.shape[:2] == depth_map.shape, f"Image and depth shape mismatch: {image.shape[:2]} vs {depth_map.shape}"

            original_size = np.array(image.shape[:2])

            # Get camera parameters from metadata
            # The extrinsics in metadata are world2cam, need to invert to cam2world
            world2cam = scene_data[frame_id].astype(np.float32)
            world2cam = world2cam[:3]  # Take first 3 rows (3x4)
            extri_opencv = closed_form_inverse_se3(world2cam[None])[0][:3]  # cam2world

            intri_opencv = scene_data['intrinsic_matrix'].astype(np.float32)
            intri_opencv = intri_opencv[:3, :3]  # Take first 3 rows and columns (3x3)

            # Process image, depth, and camera parameters
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

        set_name = "scenenet"
        batch = {
            "seq_name": set_name + "_" + seq_name.replace("/", "_"),
            "dataset_name": "scenenet",
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

