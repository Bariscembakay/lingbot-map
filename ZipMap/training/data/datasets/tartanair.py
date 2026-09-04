# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os
import os.path as osp
import logging
import random
import numpy as np
from data.dataset_util import *
from data.base_dataset import BaseDataset



SELECTED_SCENE_NAMES = [
    "abandonedfactory", 
    # "amusement", 
    "endofworld", 
    "hospital", 
    "neighborhood", 
    "office", 
    "oldtown", 
    "seasonsforest", 
    "soulcity",
    # "abandonedfactory_night", 
    "carwelding", 
    "gascola", 
    "japanesealley", 
    # "ocean", 
    # "office2", 
    "seasidetown", 
    "seasonsforest_winter", 
    "westerndesert"
]




class TartanAirDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = 'train',
        TartanAir_DIR: str = "/home/haian/Dataset/tartanair/",
        min_num_images: int = 24,
        len_train: int = 33600,
        len_test: int = 1000,
        max_interval: int = 20,
        video_prob=0.8,
        
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.TartanAir_DIR = TartanAir_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        self.video_prob = video_prob
        logging.info(f"TartanAir_DIR is {self.TartanAir_DIR}")
        
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        
        self.split = split
        self._load_data()

    def _load_data(self):
        assert os.path.exists(osp.join(self.TartanAir_DIR, "all_scenes_dict.npz")), f"all_scenes_dict.npz not found in {self.TartanAir_DIR}"
        all_scenes_dict = np.load(osp.join(self.TartanAir_DIR, "all_scenes_dict.npz"), allow_pickle=True)

        # abandonedfactory, neighborhood, etc
        scene_dirs = sorted(SELECTED_SCENE_NAMES)
        if self.debug:
            scene_dirs = scene_dirs[:1]
        
        # Initialize data structures 
        self.scenes = []  # All scene directories; 
        self.images = []  # All image basenames; 
        self.trajectories = []  # Camera poses; 
        self.sceneids = []  # Maps image index to scene ID
        self.scene_img_list = []  # List of image indices for each scene
        self.start_img_ids = []  # Valid start positions for sequences 
        
        # For this dataset, we use a fixed intrinsics matrix
        self.intrinsics = np.array([[320.,   0., 320.],
                                    [  0., 320., 240.],
                                    [  0.,   0.,   1.]], dtype=np.float32)
        
        offset = 0
        j = 0
        # Fast path: use precomputed metadata
        logging.info("TartanAir loading metadata from precomputed all_scenes_dict.npz")
        
        for scene in scene_dirs:
            if scene not in all_scenes_dict:
                logging.warning(f"Scene {scene} not found in all_scenes_dict.npz")
                continue
                
            scene_data = all_scenes_dict[scene].item()  # Convert numpy array back to dict
            
            for mode in ["Easy", "Hard"]:
                if mode not in scene_data:
                    logging.debug(f"Mode {mode} not found for scene {scene}")
                    continue
                    
                mode_data = scene_data[mode]
                mode_dir = os.path.join(self.TartanAir_DIR, scene, mode)
                
                for seq_name, seq_data in mode_data.items():
                    seq_dir = os.path.join(mode_dir, seq_name)
                    
                    # Extract data from precomputed metadata
                    valid_images = seq_data["images"]  # These are full filenames like "000000_rgb.png"
                    sequence_trajectories = seq_data["c2ws"]  # Camera poses (cam2world)
                    
                    # Convert full filenames back to basenames for consistency
                    basenames = [img[:-8] for img in valid_images if img.endswith("_rgb.png")]
                    
                    num_imgs = len(basenames)
                    cut_off = self.min_num_images
                    
                    if num_imgs < cut_off:
                        logging.debug(f"Skipping {seq_dir} - insufficient images ({num_imgs})")
                        continue
                    
                    # Add this sequence to our dataset
                    self.scenes.append(seq_dir)
                    img_ids = list(range(offset, offset + num_imgs))
                    start_img_ids_ = img_ids[: num_imgs - cut_off + 1]
                    
                    self.scene_img_list.append(img_ids)
                    self.sceneids.extend([j] * num_imgs)
                    self.images.extend(basenames)
                    self.trajectories.extend(sequence_trajectories)
                    self.start_img_ids.extend(start_img_ids_)
                    
                    offset += num_imgs
                    j += 1
    

        self.sequence_list_len = len(self.start_img_ids)
        logging.info(f"TartanAir {self.split} data loaded: {len(self.images)} images from {len(self.scenes)} sequences")

    def _show_stats(self):
        logging.info(f"TartanAir {self.split} dataset statistics:")
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
            basename = self.images[image_idx]
            intri = self.intrinsics
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            
            # Load RGB image
            image_filepath = osp.join(scene_dir, basename + "_rgb.png")
            # Load depth map
            depth_filepath = osp.join(scene_dir, basename + "_depth.npy")
            
            try:
                image = imread_cv2(image_filepath)
                depth_map = np.load(depth_filepath)
                
                # Process depth map similar to original TartanAir processing

                depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
                depth_map = threshold_depth_map(
                    depth_map, min_percentile=-1, max_percentile=99, max_depth=80 # sky mask
                )

                depth_map = depth_map.astype(np.float32)
                
            except Exception as e:
                raise OSError(f"Cannot load {basename}, got exception {e}")
            
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

        # Use the scene directory from the first image for seq_name

        seq_name = scene_dir.split("/")[-3] + "_" + scene_dir.split("/")[-2] + "_" + scene_dir.split("/")[-1]
        set_name = "tartanair"

        batch = {
            "seq_name": set_name + "_" + seq_name,
            "dataset_name": "tartanair",
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

