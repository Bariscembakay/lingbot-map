# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import numpy as np
import random

from data.dataset_util import *
from data.base_dataset import BaseDataset


class BlendedMVSDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        BlendedMVS_DIR: str = "/home/haian/Dataset/blendedmvs/",
        min_num_images: int = 24,
        len_train: int = 20000,
        len_test: int = 1000,
    ):
        """
        Initialize the BlendedMVSDataset.

        Args:
            common_conf: Configuration object with common settings.
            split (str): Dataset split, either 'train' or 'test'.
            BlendedMVS_DIR (str): Directory path to BlendedMVS data.
            min_num_images (int): Minimum number of images per sequence.
            len_train (int): Length of the training dataset.
            len_test (int): Length of the test dataset.
        """
        super().__init__(common_conf=common_conf)
        
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        
        self.BlendedMVS_DIR = BlendedMVS_DIR
        self.min_num_images = min_num_images
        self.split = split
        self.is_metric = False
        
        logging.info(f"BlendedMVS_DIR is {self.BlendedMVS_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self._load_data()

    def _load_data(self):
        """Load BlendedMVS data from precomputed metadata."""
        metadata_file = osp.join(self.BlendedMVS_DIR, "blendedmvs_metadata.npz")
        if not osp.exists(metadata_file):
            raise FileNotFoundError(f"Metadata file not found: {metadata_file}. Please run preprocessing first.")
        
        logging.info("BlendedMVS loading data from precomputed blendedmvs_metadata.npz")
        self._load_from_metadata(metadata_file)
        
        self.invalid_scenes = []
        self.is_reachable_cache = {scene: {} for scene in self.data_dict.keys()}
        
        logging.info(f"BlendedMVS {self.split} data loaded: {self.num_imgs} ref images from {self.num_scenes} scenes")

    def _load_from_metadata(self, metadata_file):
        """Load from precomputed metadata file."""
        data = np.load(metadata_file, allow_pickle=True)
        
        # Initialize data structures following Cut3R pattern
        self.data_dict = {}
        self.all_ref_imgs = []
        self.scenes = []
        self.sceneids = []
        self.images = []
        self.trajectories = []
        self.intrinsics_list = []
        self.adjacency_lists = []
        self.id_ranges = []
        
        offset = 0
        j = 0
        for scene_dir in data.files:
            if self.debug and j >= 1:
                break
                
            scene_data = data[scene_dir].item()
            images = scene_data["images"]
            c2ws = scene_data["c2ws"]
            intrinsics = scene_data["intrinsics"]
            score_matrix = scene_data["score_matrix"]
            
            if len(images) < self.min_num_images:
                continue
            
            # Convert c2w to w2c
            w2c_poses = closed_form_inverse_se3(np.array(c2ws))[:, :3]
            
            # Store in Cut3R format
            self.data_dict[scene_dir] = {
                "basenames": images,
                "score_matrix": score_matrix,  # adjacency list
                "c2ws": c2ws,
                "intrinsics": intrinsics,
            }
            
            # Add all possible ref views following Cut3R pattern
            self.all_ref_imgs.extend(
                [(scene_dir, b) for b in range(len(images))]
            )
            
            
            self.scenes.append(scene_dir)
            self.sceneids.extend([j] * len(images))
            self.images.extend(images)
            self.trajectories.extend(w2c_poses)
            self.intrinsics_list.extend(intrinsics)
            self.adjacency_lists.append(score_matrix)
            self.id_ranges.append((offset, offset + len(images)))
            
            offset += len(images)
            j += 1
        
        self.num_imgs = len(self.all_ref_imgs)
        self.num_scenes = len(self.data_dict.keys())

    def _show_stats(self):
        logging.info(f"BlendedMVS {self.split} dataset statistics:")
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
    


    @staticmethod
    def is_reachable(adjacency_list, start_index, k):
        """Check if we can reach k nodes from start_index."""
        visited = set()
        stack = [start_index]
        while stack and len(visited) < k:
            node = stack.pop()
            if node not in visited:
                visited.add(node)
                for neighbor in adjacency_list[node]:
                    if neighbor[0] not in visited:
                        stack.append(neighbor[0])
        return len(visited) >= k


    @staticmethod
    def random_sequence_with_optional_repeats(
        adjacency_list,
        k,
        start_index,
        rng: np.random.Generator,
        max_k=None,
        max_attempts=100,
    ):
        """Cut3R's sequence generation with optional repeats."""
        if max_k is None:
            max_k = k
        path = [start_index]
        visited = set([start_index])
        current_index = start_index
        attempts = 0

        while len(path) < max_k and attempts < max_attempts:
            attempts += 1
            neighbors = adjacency_list[current_index]
            neighbor_idxs = [n[0] for n in neighbors]
            neighbor_weights = [n[1] for n in neighbors]

            if not neighbor_idxs:
                # No neighbors, cannot proceed further
                break

            # Try to find unvisited neighbors
            unvisited_neighbors = [
                (idx, wgt)
                for idx, wgt in zip(neighbor_idxs, neighbor_weights)
                if idx not in visited
            ]
            if unvisited_neighbors:
                # Select among unvisited neighbors
                unvisited_idxs = [idx for idx, _ in unvisited_neighbors]
                unvisited_weights = [wgt for _, wgt in unvisited_neighbors]
                probabilities = np.array(unvisited_weights) / np.sum(unvisited_weights)
                next_index = rng.choice(unvisited_idxs, p=probabilities)
                visited.add(next_index)
            else:
                # All neighbors visited, but we need to reach length max_k
                # So we can revisit nodes
                probabilities = np.array(neighbor_weights) / np.sum(neighbor_weights)
                next_index = rng.choice(neighbor_idxs, p=probabilities)

            path.append(next_index)
            current_index = next_index

        # Require no more unique nodes than we can possibly have
        required_unique = min(k, max_k)
        if len(set(path)) >= required_unique:
            while len(path) < max_k:
                next_index = rng.choice(path)
                path.append(next_index)
            return path
        else:
            return None

    def generate_sequence(
        self, scene, adj_list, num_views, start_index, rng
    ):
        """Cut3R's sequence generation method with repeats allowed."""
        # Do not require more unique views than total views requested
        cutoff_unique = min(num_views, max(num_views // 5, 3))
        if start_index in self.is_reachable_cache[scene]:
            if not self.is_reachable_cache[scene][start_index]:
                return None
        else:
            self.is_reachable_cache[scene][start_index] = self.is_reachable(
                adj_list, start_index, cutoff_unique
            )
            if not self.is_reachable_cache[scene][start_index]:
                return None
        
        sequence = self.random_sequence_with_optional_repeats(
            adj_list, cutoff_unique, start_index, rng, max_k=num_views
        )
        if not sequence:
            self.is_reachable_cache[scene][start_index] = False
        return sequence


    def get_data(
        self,
        seq_index=None,
        img_per_seq=None,
        seq_name=None,
        ids=None,
        aspect_ratio=1.0,
    ):
        """
        Retrieve data following Cut3R's ref_view-first approach.
        """
        if self.inside_random and self.training:
            seq_index = random.randint(0, len(self.all_ref_imgs) - 1)
        
        # Get ref view info (Cut3R approach)
        scene_info, ref_img_idx = self.all_ref_imgs[seq_index]
        rng = np.random.default_rng()
        
        invalid_seq = True
        while invalid_seq:
            basenames = self.data_dict[scene_info]["basenames"]
            # Check if too many views are unreachable in this scene
            if (
                sum(
                    [
                        (1 - int(x))
                        for x in list(self.is_reachable_cache[scene_info].values())
                    ]
                )
                > len(basenames) - img_per_seq
            ):
                self.invalid_scenes.append(scene_info)
            
            # Skip invalid scenes
            while scene_info in self.invalid_scenes:
                seq_index = rng.integers(low=0, high=len(self.all_ref_imgs))
                scene_info, ref_img_idx = self.all_ref_imgs[seq_index]
                basenames = self.data_dict[scene_info]["basenames"]

            score_matrix = self.data_dict[scene_info]["score_matrix"]
            imgs_idxs = self.generate_sequence(
                scene_info, score_matrix, img_per_seq, ref_img_idx, rng
            )

            if imgs_idxs is None:
                # Try different ref views in the same scene
                random_direction = 2 * rng.choice(2) - 1
                for offset in range(1, len(basenames)):
                    tentative_im_idx = (
                        ref_img_idx + (random_direction * offset)
                    ) % len(basenames)
                    if (
                        tentative_im_idx not in self.is_reachable_cache[scene_info]
                        or self.is_reachable_cache[scene_info][tentative_im_idx]
                    ):
                        ref_img_idx = tentative_im_idx
                        break
            else:
                invalid_seq = False
                
        scene_dir = scene_info
        scene_path = osp.join(self.BlendedMVS_DIR, scene_dir)
        basenames = self.data_dict[scene_info]["basenames"]
        c2ws = self.data_dict[scene_info]["c2ws"]
        intris = self.data_dict[scene_info]["intrinsics"]
        
        target_image_shape = self.get_target_shape(aspect_ratio)
        images = []
        depths = []
        cam_points = []
        world_points = []
        point_masks = []
        extrinsics = []
        intrinsics = []
        original_sizes = []
        

        for view_idx in imgs_idxs:
            basename = basenames[view_idx]
            intri = np.array(intris[view_idx], dtype=np.float32)
            c2w = c2ws[view_idx]
            
            # Convert c2w to w2c
            w2c = closed_form_inverse_se3(c2w[None])[0]
            extri_opencv = w2c[:3]
            
            # Load RGB image and depth map
            image_filepath = osp.join(scene_path, basename + ".jpg")
            depth_filepath = osp.join(scene_path, basename + ".exr")
            
            image = imread_cv2(image_filepath)
            depth_map = imread_cv2(depth_filepath)
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            # Process depth map
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=97
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
                logging.error(f"Wrong shape for {scene_info}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
            
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        
        set_name = "blendedmvs"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_info.replace("/", "_")),
            "dataset_name": "blendedmvs",
            "ids": np.array(imgs_idxs),
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



