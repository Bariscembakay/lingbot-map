# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

from data.dataset_util import *
from data.base_dataset import BaseDataset
import os
import os.path as osp
import logging
import numpy as np
import random


EXCLUDED_SCENE = {
    "ai_003_001": ["cam_00"], # black pixels
    "ai_004_005": ["cam_00", "cam_01"],
    "ai_004_007": "all",
    "ai_010_005": ["cam_00"],
    "ai_010_008": ["cam_00"],
    "ai_010_009": ["cam_00", "cam_01", "cam_02"],
    "ai_010_009": ["cam_02"],
    "ai_011_000": ["cam_00"],
    "ai_011_001": ["cam_00"],
    "ai_014_006": ["cam_00"],
    "ai_016_001": ["cam_01"],
    "ai_016_003": ["cam_00", "cam_01"],
    "ai_017_001": ["cam_00"],
    "ai_026_015": ["cam_00"],
    "ai_027_001": ["cam_00", "cam_01","cam_02", "cam_03"],
    "ai_027_003": "all",
    "ai_027_005": "all",
    "ai_028_003": "all",
    "ai_029_004": "all",
    "ai_031_009": ["cam_00"],
    "ai_038_002": "all",
    "ai_038_004": "all",
    "ai_038_007": "all",
    "ai_039_002": ["cam_00"],
    "ai_039_003": ["cam_00"],
    "ai_039_005": ["cam_00"],
    "ai_039_006": "all",
    "ai_039_010": ["cam_00"],
    "ai_041_001": ["cam_00"],
    "ai_041_008": ["cam_00"],
    "ai_042_002": "all",
    "ai_042_003": ["cam_01"],
    "ai_042_005": "all",
    "ai_043_003": ["cam_00"],
    "ai_044_009": "all",
    "ai_045_005": "all",
    "ai_046_003": "all",
    "ai_048_001": "all",
    "ai_048_003": "all",
    "ai_054_001": "all",
    "ai_054_009": "all",
}


def check_excluded(scene_name, camera_name):
    if scene_name not in EXCLUDED_SCENE:
        return False
    if EXCLUDED_SCENE[scene_name] == "all":
        return True
    if camera_name in EXCLUDED_SCENE[scene_name]:
        return True
    return False


class HyperSimDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        HyperSim_DIR: str = "/home/haian/Dataset/hypersim/",
        min_num_images: int = 24,
        len_train: int = 20000,
        len_test: int = 1000,
        max_interval: int = 4,
        video_prob: float = 0.5,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.HyperSim_DIR = HyperSim_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        self.video_prob = video_prob
        logging.info(f"HyperSim_DIR is {self.HyperSim_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def _load_data(self):
        all_scenes_dict_exists = False
        if os.path.exists(osp.join(self.HyperSim_DIR, "all_scenes_dict.npz")):
            all_scenes_dict_exists = True
            all_scenes_dict = np.load(osp.join(self.HyperSim_DIR, "all_scenes_dict.npz"), allow_pickle=True)
        else:
            all_scenes_dict_exists = False

        # Initialize data structures similar to ARKitScenes
        self.scenes = []  # All subscene paths; len = number of subscenes
        self.sceneids = []  # Scene id for each image; len = total images
        self.images = []  # All Image names; len = total images
        self.trajectories = []  # Camera poses; len = total images
        self.intrinsics_list = []  # Camera intrinsics; len = total images
        self.sequence_list = []  # list of list of image indices; len = number of subscenes
        self.id_ranges = []  # image id ranges for each sequence; len = number of subscenes
        
        offset = 0
        j = 0
        
        if all_scenes_dict_exists:
            # Fast path: use precomputed metadata
            logging.info("Loading data from precomputed all_scenes_dict.npz")
            
            for scene_name in all_scenes_dict.keys():
                scene_data = all_scenes_dict[scene_name].item()  # Convert numpy array back to dict
                
                for subscene_name, subscene_data in scene_data.items():
                    if check_excluded(scene_name, subscene_name):
                        continue
                    # Construct full subscene path
                    subscene_path = osp.join(scene_name, subscene_name)
                    
                    # Extract data from precomputed metadata
                    valid_images = subscene_data["images"]  # These are RGB filenames like "000000_rgb.png"
                    sequence_trajectories = subscene_data["c2ws"]  # Camera poses (cam2world)
                    sequence_intrinsics = subscene_data["intrinsics"]  # Camera intrinsics
                    
                    if len(valid_images) < self.min_num_images:
                        logging.debug(f"Skipping {subscene_path} - insufficient images ({len(valid_images)})")
                        continue
                    
                    # Add this subscene to our dataset
                    img_ids = list(np.arange(len(valid_images)) + offset)
                    
                    self.scenes.append(subscene_path)
                    self.sequence_list.append(img_ids)
                    self.sceneids.extend([j] * len(valid_images))
                    self.images.extend(valid_images)
                    self.trajectories.extend(sequence_trajectories)
                    self.intrinsics_list.extend(sequence_intrinsics)
                    self.id_ranges.append((offset, offset + len(valid_images)))
                    
                    offset += len(valid_images)
                    j += 1
                    
                    if self.debug and j >= 1:  # Limit to 1 subscene in debug mode
                        break
                
                if self.debug and j >= 1:
                    break

        else:
            # Slow path: scan directories and load metadata
            logging.info("all_scenes_dict.npz not found, scanning directories...")
            
            # Get all scenes (top-level directories)
            all_scenes = sorted([
                f for f in os.listdir(self.HyperSim_DIR) 
                if os.path.isdir(osp.join(self.HyperSim_DIR, f))
            ])
            
            if self.debug:
                all_scenes = all_scenes[:1]
            
            for scene in all_scenes:
                scene_dir = osp.join(self.HyperSim_DIR, scene)
                
                # Get all subscenes (subdirectories within each scene)
                subscenes = []
                for f in os.listdir(scene_dir):
                    subscene_dir = osp.join(scene_dir, f)
                    if os.path.isdir(subscene_dir) and len(os.listdir(subscene_dir)) > 0:
                        subscenes.append(f)
                
                for subscene in subscenes:
                    subscene_dir = osp.join(scene_dir, subscene)
                    subscene_path = osp.join(scene, subscene)
                    
                    # Get all RGB PNG files
                    rgb_paths = sorted([
                        f for f in os.listdir(subscene_dir) 
                        if f.endswith("rgb.png")
                    ])
                    
                    if len(rgb_paths) < self.min_num_images:
                        logging.debug(f"Skipping {subscene_path} - insufficient images ({len(rgb_paths)})")
                        continue
                    
                    # Load camera data for this subscene
                    sequence_trajectories = []
                    sequence_intrinsics = []
                    valid_images = []
                    
                    for rgb_path in rgb_paths:
                        try:
                            # Load camera parameters
                            cam_path = rgb_path.replace("rgb.png", "cam.npz")
                            cam_file_path = osp.join(subscene_dir, cam_path)
                            
                            if not osp.exists(cam_file_path):
                                continue
                                
                            cam_file = np.load(cam_file_path)
                            intrinsics = cam_file["intrinsics"].astype(np.float32)
                            camera_pose = cam_file["pose"].astype(np.float32)  # cam2world
                            
                            sequence_trajectories.append(camera_pose)
                            sequence_intrinsics.append(intrinsics)
                            valid_images.append(rgb_path)
                            
                        except Exception as e:
                            logging.warning(f"Failed to load camera data for {rgb_path}: {e}")
                            continue
                    
                    if len(valid_images) < self.min_num_images:
                        logging.debug(f"Skipping {subscene_path} - insufficient valid images ({len(valid_images)})")
                        continue
                    
                    # Add this subscene to our dataset
                    img_ids = list(np.arange(len(valid_images)) + offset)
                    
                    self.scenes.append(subscene_path)
                    self.sequence_list.append(img_ids)
                    self.sceneids.extend([j] * len(valid_images))
                    self.images.extend(valid_images)
                    self.trajectories.extend(sequence_trajectories)
                    self.intrinsics_list.extend(sequence_intrinsics)
                    self.id_ranges.append((offset, offset + len(valid_images)))
                    
                    offset += len(valid_images)
                    j += 1
                    
                    if self.debug and j >= 1:  # Limit to 1 subscene in debug mode
                        break
                
                if self.debug and j >= 1:
                    break

        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"HyperSim {self.split} data loaded: {len(self.sequence_list)} subscene trajectories")


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
        
        # Sample images from the sequence using similar logic as ARKitScenes
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
            # fix_interval_prob=0.8,
            block_shuffle=24,
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
        subscene_path = None
        for image_idx in ids:
            scene_id = self.sceneids[image_idx]
            subscene_path = self.scenes[scene_id]
            rgb_filename = self.images[image_idx]
            
            # Construct full paths
            subscene_dir = osp.join(self.HyperSim_DIR, subscene_path)
            
            # Load intrinsics and camera pose
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image and depth map
            image_filepath = osp.join(subscene_dir, rgb_filename)
            depth_filepath = osp.join(subscene_dir, rgb_filename.replace("rgb.png", "depth.npy"))
            
            try:
                image = imread_cv2(image_filepath)
                depth_map = np.load(depth_filepath).astype(np.float32)
                
                # Process depth map similar to original HyperSim processing
                depth_map[~np.isfinite(depth_map)] = 0  # invalid
                depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
                
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=99
                )
            except Exception as e:
                raise OSError(f"Cannot load {rgb_filename} from {subscene_path}, got exception {e}")
            
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
                logging.error(f"Wrong shape for {subscene_path}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
                
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)
        
        set_name = "hypersim"

        subscene_path = subscene_path.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else subscene_path),
            "dataset_name": "hypersim",
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


