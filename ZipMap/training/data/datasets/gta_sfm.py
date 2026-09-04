# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os.path as osp
import logging
import random
import numpy as np
from typing import Dict, Any
from data.dataset_util import *  # utilities and geometry helpers
from data.base_dataset import BaseDataset




class GTASfMDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        GTA_SfM_DIR: str = "/path/to/processed_GTA-SfM/",
        min_num_images: int = 48,
        len_train: int = 20000,
        len_test: int = 1000,
        max_interval: int = 4,
        video_prob: float = 0.6,
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.GTA_SfM_DIR = GTA_SfM_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.video_prob = video_prob
        # GTA-SfM is synthetic, but we don't rely on metric scale here
        self.is_metric = False
        logging.info(f"GTA_SfM_DIR is {self.GTA_SfM_DIR}")

        if self.split == "train":
            self.split_list = ["train", 'test']
        else:
            self.split_list = [self.split]
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._load_data()

    def _load_cameras_dict(self, cameras_path: str) -> Dict[str, Any]:
        """
        Load the cameras dictionary saved by the preprocessor.
        The script may have saved to either cameras.npy (np.save with a dict)
        or cameras.npz. We support both.
        """
        if not osp.exists(cameras_path):
            raise FileNotFoundError(f"Cameras file not found: {cameras_path}")

        cams = np.load(cameras_path, allow_pickle=True)
        # If saved via np.save(dict, ...), np.load returns a 0d-object array; .item() yields the dict
        if isinstance(cams, np.ndarray) and cams.dtype == object and cams.shape == ():
            cams = cams.item()
        # If saved as np.savez, 'cams' is an NpzFile and we cannot directly .item(); but our preprocessor used np.save
        if not isinstance(cams, dict):
            # Try common key if saved strangely, but fall back to error
            try:
                cams = dict(cams)
            except Exception as e:
                raise TypeError(f"Unexpected cameras file format at {cameras_path}: {type(cams)}") from e
        return cams

    def _load_data(self):
        # Prefer cameras.npy, fallback to cameras.npz
        cameras_npy = osp.join(self.GTA_SfM_DIR, "cameras.npy")
        # cameras_npz = osp.join(self.GTA_SfM_DIR, "cameras.npz")
        # cameras_path = cameras_npy if osp.exists(cameras_npy) else cameras_npz
        cameras_path = cameras_npy
        logging.info(f"Loading GTA-SfM cameras from: {cameras_path}")
        cameras_dict = self._load_cameras_dict(cameras_path)

        # Check that all splits in split_list exist
        for split in self.split_list:
            if split not in cameras_dict:
                raise KeyError(f"Split '{split}' not found in cameras file at {cameras_path}")

        # Initialize containers similar to MVSSynthDataset
        self.scenes = []            # sequence (HDF5 stem) names
        self.scene_splits = []      # track which split each scene came from
        self.images = []            # list of basenames (e.g., '000123') across all sequences
        self.trajectories = []      # list of 4x4 cam2world poses aligned with images
        self.intrinsics_list = []   # list of 3x3 intrinsics aligned with images
        self.sequence_list = []     # per-sequence list of indices into self.images

        offset = 0
        j = 0

        # Iterate through all splits in split_list
        for split in self.split_list:
            split_cams = cameras_dict[split]
            # Iterate sequences within the split
            for seq_name in sorted(split_cams.keys()):
                entry = split_cams[seq_name]
                basenames = entry.get("basenames", [])
                Ks = entry.get("K", None)
                poses = entry.get("pose", None)

                if Ks is None or poses is None:
                    logging.warning(f"Missing K/pose for sequence {seq_name}, skipping")
                    continue

                # Ensure aligned lengths
                n = len(basenames)
                if n == 0 or Ks.shape[0] != n or poses.shape[0] != n:
                    logging.warning(
                        f"Sequence {seq_name} has inconsistent lengths: basenames={n}, K={getattr(Ks, 'shape', None)}, pose={getattr(poses, 'shape', None)}; skipping"
                    )
                    continue

                if n < self.min_num_images:
                    logging.debug(f"Skipping {seq_name} - insufficient images ({n})")
                    continue

                img_ids = list(np.arange(n) + offset)

                self.scenes.append(seq_name)
                self.scene_splits.append(split)  # Track which split this scene came from
                self.sequence_list.append(img_ids)
                self.images.extend(list(basenames))
                self.trajectories.extend([p.astype(np.float32) for p in poses])
                self.intrinsics_list.extend([k.astype(np.float32) for k in Ks])

                offset += n
                j += 1

                if self.debug and j >= 1:
                    break

        self.sequence_list_len = len(self.scenes)
        splits_str = "+".join(self.split_list) if len(self.split_list) > 1 else self.split
        logging.info(f"GTA-SfM {splits_str} data loaded: {len(self.sequence_list)} sequences and {len(self.images)} images from {len(self.scenes)} scenes.")

    def _show_stats(self):
        # for each sequence, count the image num, and plot a histogram
        image_num = dict()
        import glob
        for i, seq_name in enumerate(self.scenes):
            seq_split = self.scene_splits[i]  # Get the correct split for this sequence
            num_images = len(glob.glob(osp.join(self.GTA_SfM_DIR, seq_split, seq_name, "images", "*.png")))
            image_num[seq_name] = num_images
        from matplotlib import pyplot as plt
        # Plot histogram of image counts
        plt.figure(figsize=(10, 5))
        plt.bar(image_num.keys(), image_num.values())
        plt.xlabel("Sequence Name")
        plt.ylabel("Number of Images")
        plt.title("GTA-SfM Dataset: Number of Images per Sequence")
        plt.xticks(rotation=45)
        plt.tight_layout()
        # save
        plt.savefig("gta_sfm_image_count_histogram.png")
        mean = np.mean(list(image_num.values()))
        max = np.max(list(image_num.values()))
        min = np.min(list(image_num.values()))
        logging.info(f"Mean number of images per sequence: {mean}")
        logging.info(f"Max number of images per sequence: {max}")
        logging.info(f"Min number of images per sequence: {min}")
        total_image_num  = np.sum(list(image_num.values()))
        logging.info(f"Total number of images in dataset: {total_image_num}")
        
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

        image_idxs = self.sequence_list[seq_index]

        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]

        start_id = np.random.choice(start_image_idxs)
        # GTA-SfM is sequence/video-like; use fixed-interval sampling by default
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

        seq_name_cur = self.scenes[seq_index]
        seq_split_cur = self.scene_splits[seq_index]  # Get the original split for this sequence
        seq_root = osp.join(self.GTA_SfM_DIR, seq_split_cur, seq_name_cur)
        rgb_dir = osp.join(seq_root, "images")
        depth_dir = osp.join(seq_root, "depths")

        for image_idx in ids:
            basename = self.images[image_idx]

            # Load intrinsics and camera pose (pose is cam2world; convert to world2cam/OpenCV extrinsic)
            intri = np.array(self.intrinsics_list[image_idx], dtype=np.float32)
            cam2world = np.array(self.trajectories[image_idx], dtype=np.float32)
            world2cam = closed_form_inverse_se3(cam2world[None])[0]
            extri_opencv = world2cam[:3]

            # Filepaths
            image_filepath = osp.join(rgb_dir, f"{basename}.png")
            depth_filepath_exr = osp.join(depth_dir, f"{basename}.exr")

            image = imread_cv2(image_filepath)
            # Depth can be .exr (preferred) or 16bit .png fallback
            depth_map = imread_cv2(depth_filepath_exr)
            depth_map[~np.isfinite(depth_map)] = 0.0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            # Conservative percentile thresholding; no absolute clamp by default for GTA-SfM
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=99, max_depth=1000
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
                logging.error(
                    f"Wrong shape for {seq_name_cur}: expected {target_image_shape}, got {image.shape[:2]}"
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

        set_name = "gta_sfm"
        seq_name_tag = seq_name if seq_name is not None else seq_name_cur.replace("/", "_")
        batch = {
            "seq_name": set_name + "_" + seq_name_tag,
            "dataset_name": "gta_sfm",
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

