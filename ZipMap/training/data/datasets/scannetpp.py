# Code for ZipMap (CVPR 2026); created by Haian Jinfrom typing import Optional

import os
import os.path as osp
import logging
import numpy as np
import cv2

from data.dataset_util import *
from data.base_dataset import BaseDataset



class ScanNetppDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ScanNetpp_DIR: str = "/home/haian/Dataset/processed_scannetpp/",
        min_num_images: int = 48,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 3,
        video_prob: float = 0.8
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.ScanNetpp_DIR = ScanNetpp_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.is_metric = True
        self.video_prob = video_prob
        logging.info(f"ScanNetpp_DIR is {self.ScanNetpp_DIR}")
        
        # ScanNet++ packaged here has no explicit train/test subfolders in the processed layout.
        # Keep API parity with other datasets by setting an effective length.
        if split == "train":
            self.len_train = len_train
        elif split == "test":
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")

        self._load_data()

    def _load_data(self):
        # Optional pre-merged dict for faster loading
        all_scenes_dict_path = osp.join(self.ScanNetpp_DIR, "all_scenes_dict.npz")
        all_scenes_dict_exists = osp.exists(all_scenes_dict_path)
        if all_scenes_dict_exists:
            all_scenes_dict = np.load(all_scenes_dict_path, allow_pickle=True)
            all_scenes = all_scenes_dict['scene_names']
        else:
            # Load all scene names
            with np.load(osp.join(self.ScanNetpp_DIR, "all_metadata.npz")) as data:
                all_scenes = data["scenes"]

        if self.debug:
            all_scenes = all_scenes[:200]

        self.groups = []
        self.images = []
        self.intrinsics = []
        self.trajectories = []
        self.sceneids = []
        self.id_ranges = []
        self.scenes = []  # actually loaded scenes
        self.sequence_list = []

        offset = 0
        j = 0

        for i, scene in enumerate(all_scenes):
            scene_dir = osp.join(self.ScanNetpp_DIR, scene)

            # Load per-scene metadata
            if all_scenes_dict_exists and scene in all_scenes_dict:
                scene_data = all_scenes_dict[scene].item()
                imgs = scene_data["images"]
                intrins = scene_data["intrinsics"]
                traj = scene_data["trajectories"]
                collections = scene_data["image_collection"]
                imgs_on_disk = scene_data.get("imgs_on_disk", set())
            else:
                scene_dir = osp.join(self.ScanNetpp_DIR, scene)
                npz_path = osp.join(scene_dir, "new_scene_metadata.npz")
                if not osp.exists(npz_path):
                    logging.debug(f"Skipping {scene}, missing new_scene_metadata.npz")
                    continue
                with np.load(npz_path, allow_pickle=True) as data:
                    imgs = data["images"]
                    intrins = data["intrinsics"]
                    traj = data["trajectories"]
                    assert "image_collection" in data, "Image collection not found"
                    collections = data["image_collection"].item()
                            # Ensure images on disk exist to avoid I/O errors later (from cut3r)
                imgs_on_disk = sorted(os.listdir(osp.join(scene_dir, "images"))) if osp.exists(osp.join(scene_dir, "images")) else []
                imgs_on_disk = set(map(lambda x: x[:-4], imgs_on_disk)) # can be smaller than img_ids 

            num_imgs = len(imgs)
            if num_imgs < self.min_num_images:
                logging.debug(
                    f"Skipping {scene} because it has less than {self.min_num_images} images (num_imgs = {num_imgs})"
                )
                continue
            
            # Partition IDs into DSLR and iPhone streams (by filename pattern), keeping only present ones
            img_ids = np.arange(num_imgs).tolist()
            dslr_ids = [i + offset for i in img_ids if imgs[i].startswith("DSC") and imgs[i] in imgs_on_disk]
            iphone_ids = [i + offset for i in img_ids if imgs[i].startswith("frame") and imgs[i] in imgs_on_disk]

            img_groups = []
            img_id_ranges = []

            # collections maps ref_id -> list[(id, score), ...]
            for ref_id, group in collections.items():
                if len(group) + 1 < self.min_num_images:
                    continue
                # Include the ref at highest score
                group = list(group)  # copy if it's a list of tuples
                group.insert(0, (ref_id, 1.0)) # insert itself
                # # Sort by score desc, then convert to global indices
                # group_sorted = sorted(group, key=lambda x: x[1], reverse=True) # score from high to low
                # group_ids = [int(x[0] + offset) for x in group_sorted]
                group_ids = [int(x[0] + offset) for x in group]
                img_groups.append(sorted(group_ids))

                # determine if the current collection is dslr- or iphone-based
                if imgs[ref_id].startswith("frame"):
                    img_id_ranges.append(dslr_ids)
                else:
                    img_id_ranges.append(iphone_ids)
                    
            if len(img_groups) == 0:
                logging.debug(f"Skipping scene {scene} because it has no valid sequences")
                continue
            # Append scene data
            self.scenes.append(scene)
            self.sceneids.extend([j] * num_imgs)
            self.images.extend(list(imgs))
            self.intrinsics.extend(list(intrins))
            self.trajectories.extend(list(traj))
            self.groups.extend(img_groups)
            self.id_ranges.extend(img_id_ranges)

            offset += num_imgs
            j += 1

        # Finalization
        self.sequence_list = self.groups
        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"ScanNet++ {self.split} data loaded: {len(self.sequence_list)} sequences")

    def _show_stats(self):
        logging.info(f"ScanNet++ {self.split} dataset statistics:")
        logging.info(f"  Number of scenes: {len(self.scenes)}")
        logging.info(f"  Number of images: {len(self.images)}")
        logging.info(f"  Number of sequences: {len(self.sequence_list)}")
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
        # Statistics on sequence level
        sequence_image_counts = [len(g) for g in self.sequence_list]
        if sequence_image_counts:
            min_count = min(sequence_image_counts)
            max_count = max(sequence_image_counts)
            avg_count = sum(sequence_image_counts) / len(sequence_image_counts)
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

        image_idxs = self.sequence_list[seq_index]


        rand_val = np.random.rand()
        image_idxs_video = self.id_ranges[seq_index]
        cut_off = img_per_seq
        if cut_off > len(image_idxs_video):
            start_image_idxs = image_idxs_video[:1]
        else:
            start_image_idxs = image_idxs_video[: len(image_idxs_video) - cut_off + 1]

        if (rand_val < 0.7 and len(start_image_idxs) > 0) or (len(image_idxs) <= img_per_seq):
            # Video-like sampling on the alternate stream
            start_id = np.random.choice(start_image_idxs)
            pos, ordered_video = self.get_seq_from_start_id(
                img_per_seq,
                start_id,
                image_idxs_video,
                max_interval=self.max_interval,
                video_prob=self.video_prob,
                fix_interval_prob=0.5,
                block_shuffle=24,
            )
            ids = np.array(image_idxs_video)[pos]
        else:
            # Ordered video with varying intervals from the collection
            ordered_video = True
            num_candidates = len(image_idxs)
            max_id = min(num_candidates, int(img_per_seq * (2 + 2 * np.random.rand())))
            # Ensure at least img_per_seq candidates
            max_id = max(max_id, min(num_candidates, img_per_seq))
            image_idxs_subset = image_idxs[:max_id]
            image_idxs_sel = np.random.permutation(image_idxs_subset)[:img_per_seq]
            image_idxs_sel = np.sort(image_idxs_sel)
            if rand_val > 0.75:
                ordered_video = False
                image_idxs_sel = np.random.permutation(image_idxs_sel)
            ids = image_idxs_sel
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
            scene_id = self.sceneids[image_idx]
            scene = self.scenes[scene_id]
            scene_dir = osp.join(self.ScanNetpp_DIR, scene)

            intri = np.float32(self.intrinsics[image_idx])
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)  # cam2world
            # Convert to OpenCV w2c extrinsic [3x4]
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]

            basename = self.images[image_idx]
            image_filepath = osp.join(scene_dir, "images", basename + ".jpg")
            depth_filepath = osp.join(scene_dir, "depth", basename + ".png")

            try:
                image = imread_cv2(image_filepath)
                depth_map = imread_cv2(depth_filepath, cv2.IMREAD_UNCHANGED)
                depth_map = depth_map.astype(np.float32) / 1000.0
                depth_map[~np.isfinite(depth_map)] = 0.0
                depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
                depth_map = threshold_depth_map(depth_map, min_percentile=-1, max_percentile=98)
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
                logging.error(
                    f"Wrong shape for {scene}: expected {target_image_shape}, got {image.shape[:2]}"
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

        set_name = "scannetpp"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "scannetpp",
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

