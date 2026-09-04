# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os
import os.path as osp
import logging
import random
import numpy as np
from data.dataset_util import *
from data.base_dataset import BaseDataset
from PIL import Image

# Selected scenes for training/testing
SELECTED_SCENES = [
    "Kite_training",
    "Kite_test", 
    "PLE_training",
    "PLE_test",
    "VO_test",
]

# Selected conditions
SELECTED_CONDITIONS = [
    "cloudy",
    "foggy", 
    "sunny",
    "sunset",
    "fall",
    "spring",
    "winter",
]

def open_float16(image_path):
    """Load float16 depth image from MidAir dataset
    
    MidAir depth maps encode the Euclidean distance from the camera focal point
    to each pixel, not the standard Z-depth. This requires special handling
    during 3D point projection.
    
    Note: MidAir uses a specific encoding where uint16 PNG data is reinterpreted
    as float16. This is the official format specification from the dataset authors.
    """
    pic = Image.open(image_path)
    img = np.asarray(pic, np.uint16)
    img.dtype = np.float16
    # Convert to float32 for further processing
    img = img.clip(max=31740).astype(np.float32)
    return img

def convert_euclidean_to_z_depth(euclidean_depth, focal_length, image_height=None, image_width=None):
    """Convert MidAir Euclidean distance depth to standard OpenCV Z-depth
    
    MidAir depth maps encode Euclidean distance from focal point to each pixel.
    We need to convert this to standard Z-depth (perpendicular distance to image plane).
    
    From MidAir documentation:
    rxy = euclidean_distance / sqrt((xi - h/2)^2 + (yi - h/2)^2 + f^2)
    zc = rxy * f = euclidean_distance * f / sqrt((xi - h/2)^2 + (yi - h/2)^2 + f^2)
    
    The Z-depth is simply zc.
    """
    h, w = euclidean_depth.shape
    
    # Handle invalid depths first to avoid overflow issues
    invalid_mask = (euclidean_depth <= 0) | np.isnan(euclidean_depth) | np.isinf(euclidean_depth)

    # # Clip extremely large values to prevent overflow (e.g., > 1000m is unrealistic), and make it zero
    # euclidean_depth_filtered = np.where(euclidean_depth > 1000.0, 0, euclidean_depth)

    # Create pixel coordinate grids
    yi, xi = np.meshgrid(np.arange(h), np.arange(w), indexing='ij')
    
    # Convert to camera coordinate system (centered at principal point)
    x_centered = xi - w/2
    y_centered = yi - h/2
    
    # Calculate denominator: sqrt((xi - h/2)^2 + (yi - h/2)^2 + f^2)
    distance_squared = x_centered**2 + y_centered**2 + focal_length**2
    denominator = np.sqrt(distance_squared)
    
    # Convert Euclidean distance to Z-depth with overflow protection
    # Use double precision for intermediate calculation to prevent overflow
    euclidean_double = euclidean_depth.astype(np.float64)
    focal_double = np.float64(focal_length)
    denominator_double = denominator.astype(np.float64)
    
    z_depth_double = euclidean_double * focal_double / denominator_double
    
    # Convert back to float32 and handle any remaining overflow/underflow
    z_depth = np.clip(z_depth_double, 0, 1000.0).astype(np.float32)
    
    # Restore original invalid values
    z_depth[invalid_mask] = euclidean_depth[invalid_mask]
    
    return z_depth





class MidAirDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        MidAir_DIR: str = "/home/haian/codebase/zipmap_code/raw_data/mid_air/",
        min_num_images: int = 24,
        len_train: int = 33600,
        len_test: int = 1000,
        max_interval: int = 20,
        video_prob: float = 0.8,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.MidAir_DIR = MidAir_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.video_prob = video_prob
        logging.info(f"MidAir_DIR is {self.MidAir_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def _load_data(self):
        metadata_file = osp.join(self.MidAir_DIR, "midair_all_scenes_dict.npz")
        assert os.path.exists(metadata_file), f"midair_all_scenes_dict.npz not found in {self.MidAir_DIR}, please run save_midair_metadata.py first."
        all_scenes_dict = np.load(metadata_file, allow_pickle=True)
        # Filter scenes based on split
        if self.split == "train":
            scene_filter = ["_training"]
        elif self.split == "test":
            scene_filter = ["_test"]
        else:
            scene_filter = ["_training", "_test"]  # Use both for validation or other splits

        scene_dirs = []
        for scene in SELECTED_SCENES:
            if any(suffix in scene for suffix in scene_filter):
                scene_dirs.append(scene)
        
        if self.debug:
            scene_dirs = scene_dirs[:1]
        
        # Initialize data structures 
        self.scenes = []  # All scene directories
        self.images = []  # All image basenames
        self.trajectories = []  # Camera poses
        self.sceneids = []  # Maps image index to scene ID
        self.scene_img_list = []  # List of image indices for each scene
        self.start_img_ids = []  # Valid start positions for sequences
        
        # MidAir camera intrinsics (from the paper/documentation)
        # Field of view: 90 degrees, image size: 1024x1024
        # For MidAir: f = h/2 = w/2, so f = 512 for 1024x1024 images
        # Principal point at image center
        self.focal_length = 512.0  # f = h/2 = w/2 for MidAir
        self.image_height = 1024
        self.image_width = 1024
        self.intrinsics = np.array([[512.0,   0.0, 512.0],
                                   [  0.0, 512.0, 512.0],  
                                   [  0.0,   0.0,   1.0]], dtype=np.float32)
        
        offset = 0
        j = 0
        
        logging.info("MidAir loading data from precomputed midair_all_scenes_dict.npz")
        
        for scene in scene_dirs:
            if scene not in all_scenes_dict:
                logging.warning(f"Scene {scene} not found in midair_all_scenes_dict.npz")
                continue
                
            scene_data = all_scenes_dict[scene].item()  # Convert numpy array back to dict
            
            for condition in SELECTED_CONDITIONS:
                if condition not in scene_data:
                    logging.debug(f"Condition {condition} not found for scene {scene}")
                    continue
                    
                condition_data = scene_data[condition]
                condition_dir = os.path.join(self.MidAir_DIR, scene, condition)
                
                for seq_name, seq_data in condition_data.items():
                    seq_dir = condition_dir  # MidAir stores all data in the condition directory
                    
                    # Extract data from precomputed metadata
                    valid_images = seq_data["images"]  # These are full relative paths like "color_left/trajectory_3000/00000000.png"
                    sequence_trajectories = seq_data["c2ws"]  # Camera poses (cam2world)
                    
                    num_imgs = len(valid_images)
                    cut_off = self.min_num_images
                    
                    if num_imgs < cut_off:
                        logging.debug(f"Skipping {seq_dir}/{seq_name} - insufficient images ({num_imgs})")
                        continue
                    
                    # Add this sequence to our dataset
                    self.scenes.append(seq_dir)
                    img_ids = list(range(offset, offset + num_imgs))
                    start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
                    
                    self.scene_img_list.append(img_ids)
                    self.sceneids.extend([j] * num_imgs)
                    
                    # Store the full relative paths for proper loading
                    self.images.extend(valid_images)
                    self.trajectories.extend(sequence_trajectories)
                    self.start_img_ids.extend(start_img_ids_)
                    
                    offset += num_imgs
                    j += 1

        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"MidAir {self.split} data loaded: {len(self.images)} images from {len(self.scenes)} sequences")

    def _show_stats(self):
        logging.info(f"MidAir {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of start positions: {len(self.start_img_ids)}")
        # Statistics on scene level
        scene_image_counts = [len(scene_imgs) for scene_imgs in self.scene_img_list]
        if scene_image_counts:
            min_count = min(scene_image_counts)
            max_count = max(scene_image_counts)
            avg_count = sum(scene_image_counts) / len(scene_image_counts)
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
        
        # Use cut3r-style sampling logic
        start_id = self.start_img_ids[seq_index]
        scene_id = self.sceneids[start_id]
        all_image_ids = self.scene_img_list[scene_id]
        
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            all_image_ids,
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.8,
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
        scene_dir = self.scenes[scene_id]
        
        for image_idx in ids:
            image_path = self.images[image_idx]  # Full relative path like "color_left/trajectory_3000/00000000.png"
            intri = self.intrinsics
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image using full relative path
            image_filepath = osp.join(scene_dir, image_path)
            # Create depth path by replacing color_left with depth
            depth_path = image_path.replace("color_left", "depth").replace(".JPEG", ".PNG")
            depth_filepath = osp.join(scene_dir, depth_path)
            
            if not os.path.exists(image_filepath):
                # switch JPEG and PNG
                if image_filepath.endswith(".JPEG"):
                    image_filepath = image_filepath[:-5] + ".PNG"
                elif image_filepath.endswith(".PNG"):
                    image_filepath = image_filepath[:-4] + ".JPEG"
            image = imread_cv2(image_filepath)
            euclidean_depth = open_float16(depth_filepath)
            
            # # Convert MidAir Euclidean depth to standard Z-depth
            depth_map = convert_euclidean_to_z_depth(
                euclidean_depth, self.focal_length, self.image_height, self.image_width
            )

            # Process depth map - MidAir depth is in meters
            # Clip extreme values and handle invalid depths
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            
            # Apply reasonable depth range for outdoor drone scenes
            depth_map = threshold_depth_map(
                depth_map, min_percentile=0.1, max_percentile=98, max_depth=200)  

            depth_map = depth_map.astype(np.float32)
            
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
                logging.error(f"Wrong shape for {scene_dir}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
                
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)

        # Create sequence name from scene directory structure
        scene_parts = scene_dir.replace(self.MidAir_DIR, '').strip('/').split('/')
        trajectory_part = "_".join(image_path.split('/')[:2])

        seq_name = "_".join(scene_parts) + "_" + trajectory_part
        set_name = "midair"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "dataset_name": "midair",
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


