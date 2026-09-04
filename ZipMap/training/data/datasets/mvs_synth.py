# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import numpy as np
import os
import random
import logging
from data.dataset_util import *
from data.base_dataset import BaseDataset


class MVSSynthDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        MVS_Synth_DIR: str = "/path/to/mvs_synth/",
        min_num_images: int = 24,
        len_train: int = 10000,
        len_test: int = 1000,
        max_interval: int = 4,
        video_prob: float = 0.6
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.MVS_Synth_DIR = MVS_Synth_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.video_prob = video_prob
        self.is_metric = False  # MVS_Synth is not metric
        logging.info(f"MVS_Synth_DIR is {self.MVS_Synth_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def _load_data(self):
        # Check for precomputed metadata (similar to HyperSim structure)
        all_scenes_dict_exists = False
        if os.path.exists(osp.join(self.MVS_Synth_DIR, "all_scenes_dict.npz")):
            all_scenes_dict_exists = True
            all_scenes_dict = np.load(osp.join(self.MVS_Synth_DIR, "all_scenes_dict.npz"), allow_pickle=True)
        else:
            all_scenes_dict_exists = False

        
        # Initialize data structures
        self.scenes = []  # All scene paths; len = number of scenes
        self.images = []  # All Image names; len = total images
        self.trajectories = []  # Camera poses; len = total images
        self.intrinsics_list = []  # Camera intrinsics; len = total images
        self.sequence_list = []  # list of list of image indices; len = number of scenes
        
        offset = 0
        j = 0
        
        if all_scenes_dict_exists:
            # Fast path: use precomputed metadata
            logging.info("MVS_Synth loading metadata from precomputed all_scenes_dict.npz")
            
            for scene_name in all_scenes_dict.keys():
                scene_data = all_scenes_dict[scene_name].item()  # Convert numpy array back to dict
                
                # Extract data from precomputed metadata
                valid_images = scene_data["images"]  # These are RGB filenames like "000000.jpg"
                sequence_trajectories = scene_data["c2ws"]  # Camera poses (cam2world)
                sequence_intrinsics = scene_data["intrinsics"]  # Camera intrinsics
                
                if len(valid_images) < self.min_num_images:
                    logging.debug(f"Skipping {scene_name} - insufficient images ({len(valid_images)})")
                    continue
                
                # Add this scene to our dataset
                img_ids = list(np.arange(len(valid_images)) + offset)
                
                self.scenes.append(scene_name)
                self.sequence_list.append(img_ids)
                self.images.extend(valid_images)
                self.trajectories.extend(sequence_trajectories)
                self.intrinsics_list.extend(sequence_intrinsics)
                
                offset += len(valid_images)
                j += 1
                
                if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                    break

        else:
            # Slow path: scan directories and load metadata
            logging.info("all_scenes_dict.npz not found, scanning directories...")
            
            # Get all scenes (top-level directories)
            all_scenes = sorted([f for f in os.listdir(self.MVS_Synth_DIR)])
            
            if self.debug:
                all_scenes = all_scenes[:1]
            
            for scene in all_scenes:
                scene_dir = osp.join(self.MVS_Synth_DIR, scene)
                rgb_dir = osp.join(scene_dir, "rgb")
                
                # Get all RGB JPG files
                rgb_paths = sorted([
                    f[:-4] for f in os.listdir(rgb_dir) 
                    if f.endswith(".jpg") or f.endswith(".png")
                ])
                
                if len(rgb_paths) < self.min_num_images:
                    logging.debug(f"Skipping {scene} - insufficient images ({len(rgb_paths)})")
                    continue
                
                # Load camera data for this scene
                sequence_trajectories = []
                sequence_intrinsics = []
                valid_images = []
                
                cam_dir = osp.join(scene_dir, "cam")
                depth_dir = osp.join(scene_dir, "depth")
                
                for basename in rgb_paths:
                    try:
                        cam_file_path = osp.join(cam_dir, basename + ".npz")
                        
                        # Load camera parameters
                        cam_file = np.load(cam_file_path)
                        intrinsics = cam_file["intrinsics"].astype(np.float32)
                        camera_pose = cam_file["pose"].astype(np.float32)  # cam2world
                        
                        sequence_trajectories.append(camera_pose)
                        sequence_intrinsics.append(intrinsics)
                        valid_images.append(basename)
                        
                    except Exception as e:
                        logging.debug(f"Error loading {basename} from {scene}: {e}")
                        continue
                
                if len(valid_images) < self.min_num_images:
                    logging.debug(f"Skipping {scene} - insufficient valid images ({len(valid_images)})")
                    continue
                
                # Add this scene to our dataset
                img_ids = list(np.arange(len(valid_images)) + offset)
                
                self.scenes.append(scene)
                self.sequence_list.append(img_ids)
                self.images.extend(valid_images)
                self.trajectories.extend(sequence_trajectories)
                self.intrinsics_list.extend(sequence_intrinsics)
                
                offset += len(valid_images)
                j += 1
                
                if self.debug and j >= 1:  # Limit to 1 scene in debug mode
                    break

        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"MVS_Synth {self.split} data loaded: {len(self.sequence_list)} scene trajectories")

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
            
        # Get image indices for this sequence
        image_idxs = self.sequence_list[seq_index]
        
        # Sample images from the sequence using similar logic as HyperSim
        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]
        start_id = np.random.choice(start_image_idxs)
        
        pos, ordered_video = self.get_seq_from_start_id(
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
        scene_path = self.scenes[seq_index]
        
        for image_idx in ids:
            basename = self.images[image_idx]
            # Construct full paths
            scene_dir = osp.join(self.MVS_Synth_DIR, scene_path)
            rgb_dir = osp.join(scene_dir, "rgb")
            depth_dir = osp.join(scene_dir, "depth")
            
            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image and depth map
            image_filepath = osp.join(rgb_dir, basename + ".jpg")
            depth_filepath = osp.join(depth_dir, basename + ".npy")
            
            image = imread_cv2(image_filepath)
            depth_map = np.load(depth_filepath).astype(np.float32)
            
            # Process depth map similar to original MVS_Synth processing
            depth_map[~np.isfinite(depth_map)] = 0  # invalid
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)

            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=98, max_depth=1000 # sky mask
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
                logging.error(f"Wrong shape for {scene_path}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
                
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        
        set_name = "mvs_synth"
        scene_path = scene_path.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else scene_path),
            "dataset_name": "mvs_synth",
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


