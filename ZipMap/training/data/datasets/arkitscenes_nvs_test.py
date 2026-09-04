import os
import os.path as osp
import logging
import random
import numpy as np
import cv2
from data.dataset_util import *
from data.base_dataset import BaseDataset
from omegaconf import OmegaConf

from tqdm import tqdm
from PIL import Image

# logging.basicConfig(level=logging.DEBUG)

class ARKitScenesDataset(BaseDataset):
    def __init__(
        self,
        common_conf,
        split: str = "train",
        ARKitScenes_DIR: str = "/home/haian/Dataset/processed_arkitscenes/",
        min_num_images: int = 24,
        len_train: int = 100800,
        len_test: int = 10000,
        max_interval: int = 8,
        video_prob: float = 1.0
    ):
        super().__init__(common_conf=common_conf)
        self.debug = common_conf.debug
        self.training = common_conf.training
        self.get_nearby = common_conf.get_nearby
        self.inside_random = common_conf.inside_random
        self.allow_duplicate_img = common_conf.allow_duplicate_img
        self.ARKitScenes_DIR = ARKitScenes_DIR
        self.max_interval = max_interval
        self.split = split
        self.min_num_images = min_num_images
        self.video_prob = video_prob
        self.is_metric = True
        logging.info(f"ARKitScenes_DIR is {self.ARKitScenes_DIR}")
        if split == "train":
            split_folder = "Training"
            self.len_train = len_train
        elif split == "test":
            split_folder = "Test"
            self.len_train = len_test
        else:
            raise ValueError(f"Invalid split: {split}")
        self.split = split
        self.split_folder = split_folder
        self._load_data()


    def _load_data(self):
        all_scenes_dict_path = osp.join(self.ARKitScenes_DIR, self.split_folder, "all_scenes_dict.npz")
        all_scenes_dict_exists = osp.exists(all_scenes_dict_path)
        if all_scenes_dict_exists:
            all_scenes_dict = np.load(all_scenes_dict_path, allow_pickle=True)
            all_scenes = all_scenes_dict['scene_names']
        else: 
            # Load metadata
            with np.load(osp.join(self.ARKitScenes_DIR, self.split_folder, "all_metadata.npz")) as data:
                all_scenes = data["scenes"] # num = 3344 when split = 'train'
                # self.debug = True
        if self.debug:
            all_scenes = all_scenes[:100]

        # Preload all scene metadata
        self.groups = []
        self.images = [] # image name sorted by global image index; len = 956k for train (without filtering)
        self.intrinsics = [] # intrinsic matrix for each image, len = total number of images, e.g. 956k for train; 
        self.trajectories = [] # camera pose for each image, len = total number of images, e.g. 956k for train; 
        self.sceneids = [] # scene id for each image, len = total number of images, e.g. 956k for train;; It's like 0, 0, 0, 1, 1, 1, 2, 2, 2, ...
        self.id_ranges = [] # image id range for each sequence, len = scene * sequence num per scene, (len(self.sequence_list )), it's 543k for train); e.g. [0, 385], [0, 385], [0, 385], [0, 385], [0, 385], [0, 385], ...
        self.scenes = []  # Track actually loaded scenes, len = actual number of scenes; e.g. 3344 for train
        self.sequence_list = [] # sequence list, which is the len of dataset; e.g. 923k for train; after filtering (>=24), it's 543k for train
        offset = 0
        j = 0


        for i, scene in enumerate(all_scenes):
            scene_data = None
            if all_scenes_dict_exists:
                scene_data = all_scenes_dict[scene].item()
                imgs = scene_data["images"] # list of image names; e.g. num = 385 when split = 'train'
                intrins = scene_data["intrinsics"]
                traj = scene_data["trajectories"]
                collections = scene_data["image_collection"] # collection * [num_imgs]; e.g. 381 * collections, each collection is a list of image indices like [44, 144, 141, ...]
            else:
                scene_dir = osp.join(self.ARKitScenes_DIR, self.split_folder, scene)
                with np.load(osp.join(scene_dir, "new_scene_metadata.npz"), allow_pickle=True) as data:
                    imgs = data["images"] # e.g. num = 385 when split = 'train'
                    intrins = data["intrinsics"]
                    traj = data["trajectories"]
                    collections = data["image_collection"] # collection * [num_imgs]; e.g. 381 * collections, each collection is a list of image indices like [44, 144, 141, ...]

            if not all_scenes_dict_exists:
                collections = collections.item()
            num_imgs = imgs.shape[0] # total number of images in the scene; e.g. 385
            if num_imgs < self.min_num_images:
                logging.debug(f"Skipping scene {scene} because it has less than {self.min_num_images} images (num_imgs = {num_imgs})")
                continue

            img_groups = []
            for ref_id, group in collections.items():
                if len(group) + 1 < 3 or len(group) + 1 < self.min_num_images: # len(group) = 0 or 1
                    continue
                group.insert(0, (ref_id, 1.0))
                group = [int(x[0] + offset) for x in group] # global image index
                img_groups.append(sorted(group))
            if len(img_groups) == 0:
                logging.debug(f"Skipping scene {scene} because it has no valid sequences")
                continue
            self.scenes.append(scene)  # Add to filtered scenes list
            self.sceneids.extend([j] * num_imgs) # scene id for each image
            self.id_ranges.extend([(offset, offset + num_imgs) for _ in range(len(img_groups))])
            self.images.extend(imgs)
            K = np.expand_dims(np.eye(3), 0).repeat(num_imgs, 0)
            K[:, 0, 0] = [fx for _, _, fx, _, _, _ in intrins]
            K[:, 1, 1] = [fy for _, _, _, fy, _, _ in intrins]
            K[:, 0, 2] = [cx for _, _, _, _, cx, _ in intrins]
            K[:, 1, 2] = [cy for _, _, _, _, _, cy in intrins]
            self.intrinsics.extend(list(K))
            self.trajectories.extend(list(traj))
            self.groups.extend(img_groups)
            offset += num_imgs
            j += 1

        # Update scenes to only include actually loaded scenes
        self.sequence_list = self.groups
        self.sequence_list_len = len(self.sequence_list)
        logging.info(f"ARKitScenes {self.split} data loaded: {len(self.sequence_list)} sequences")
        

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
            seq_index = random.randint(0, self.sequence_list_len - 1)

        image_idxs = self.sequence_list[seq_index]

        image_idxs = np.arange(self.id_ranges[seq_index][0], self.id_ranges[seq_index][1])
        cut_off = img_per_seq
        if len(image_idxs) < cut_off:
            start_image_idxs = image_idxs[:1]
        else:
            start_image_idxs = image_idxs[: len(image_idxs) - cut_off + 1]
        start_id = np.random.choice(start_image_idxs)
        pos, ordered_video = self.get_seq_from_start_id(
            img_per_seq,
            start_id,
            image_idxs.tolist(),
            max_interval=self.max_interval,
            video_prob=self.video_prob,
            fix_interval_prob=0.5,
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
        for image_idx in ids:
            scene_id = self.sceneids[image_idx]
            scene = self.scenes[scene_id]
            split_folder = "Training" if self.split == "train" else "Test"
            scene_dir = osp.join(self.ARKitScenes_DIR, split_folder, scene)
            intri = np.float32(self.intrinsics[image_idx])
            camera_pose = np.array(self.trajectories[image_idx], dtype=np.float32)
            # make it w2c
            camera_pose = closed_form_inverse_se3(camera_pose[None])[0]
            extri_opencv = camera_pose[:3]
            basename = self.images[image_idx]
            image_filepath = osp.join(scene_dir, "vga_wide", basename.replace(".png", ".jpg"))
            if not osp.exists(image_filepath):
                # Try .jpg if .png not found, or vice versa
                alt_ext = ".png" if image_filepath.endswith(".jpg") else ".jpg"
                image_filepath_alt = osp.join(scene_dir, "vga_wide", osp.splitext(basename)[0] + alt_ext)
                if osp.exists(image_filepath_alt):
                    image_filepath = image_filepath_alt
            depth_filepath = osp.join(scene_dir, "lowres_depth", basename)
            image = imread_cv2(image_filepath)
            
            depth_map = imread_cv2(depth_filepath, cv2.IMREAD_UNCHANGED)
            depth_map = depth_map.astype(np.float32) / 1000.0
            depth_map[~np.isfinite(depth_map)] = 0
            depth_map = np.nan_to_num(depth_map, nan=0, posinf=0, neginf=0)
            depth_map = threshold_depth_map(
                depth_map, min_percentile=-1, max_percentile=98
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
                logging.error(f"Wrong shape for {scene}: expected {target_image_shape}, got {image.shape[:2]}")
                continue
            images.append(image)
            depths.append(depth_map)
            extrinsics.append(extri_opencv)
            intrinsics.append(intri_opencv)
            cam_points.append(cam_coords_points)
            world_points.append(world_coords_points)
            point_masks.append(point_mask)
            original_sizes.append(original_size)


        set_name = "arkitscenes"
        batch = {
            "seq_name": set_name + "_" + (seq_name if seq_name is not None else str(seq_index)),
            "dataset_name": "arkitscenes",
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


if __name__ == "__main__":
    logging.basicConfig(level=logging.INFO)
    base_dir = "/home/storage-bucket/haian/zipmap_data/"
    spec_dir = "dust3r_data/processed_arkitscenes"
    dataset_dir = osp.join(base_dir, spec_dir)
    common_conf = OmegaConf.load("config/default_dataset_debug.yaml")
    dataset = ARKitScenesDataset(common_conf=common_conf, ARKitScenes_DIR=dataset_dir)

    # dataset._show_stats()

    VISUALIZE_ONLY_ONE_SAMPLE = True
    SAMPLE_NUM = 5
    IMAGES_PER_SEQ = 8

    if VISUALIZE_ONLY_ONE_SAMPLE:
        # Save per-sequence images and a fused PLY like other datasets
        for cur_id in tqdm(range(min(SAMPLE_NUM, dataset.sequence_list_len)), desc="Visualizing"):
            batch = dataset.get_data(seq_index=cur_id, img_per_seq=IMAGES_PER_SEQ)
            
            # Apply the same preprocessing used during training
            # batch = process_batch_for_training_consistency(batch)
            
            seq_name = batch["seq_name"]
            world_points = batch["world_points"]
            point_masks = batch["point_masks"]
            images = batch["images"]

            all_world_points = []
            all_world_point_colors = []

            for i in range(len(world_points)):
                pts = world_points[i]
                msk = point_masks[i]
                img = images[i]
                if pts is None or msk is None:
                    continue
                if msk.dtype != bool:
                    msk = msk.astype(bool)
                pts_valid = pts[msk]
                cols_valid = img[msk]
                if pts_valid.size == 0:
                    continue
                all_world_points.append(pts_valid)
                all_world_point_colors.append(cols_valid)

            if len(all_world_points) == 0:
                print(f"No valid points for {seq_name}")
                continue

            all_world_points = np.concatenate(all_world_points, axis=0)
            all_world_point_colors = np.concatenate(all_world_point_colors, axis=0)

            os.makedirs("debug", exist_ok=True)
            os.makedirs(f"debug/{seq_name}", exist_ok=True)

            # Save images of the sequence
            for i, img in enumerate(images):
                try:
                    Image.fromarray(img).save(f"debug/{seq_name}/{i:03d}.png")
                except Exception as e:
                    print(f"Failed saving image {i} for {seq_name}: {e}")

            # Save fused point cloud
            save_ply(
                all_world_points.reshape(-1, 3),
                all_world_point_colors.reshape(-1, 3),
                f"debug/{seq_name}.ply",
            )
    else:
        # Throughput-oriented DataLoader test
        batch_size = 1 * 2  # 2 batch per GPU and 8 GPU; adjust as needed
        num_workers = 72
        from torch.utils.data import DataLoader
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size,
            shuffle=True,
            num_workers=num_workers,
            pin_memory=True,
            drop_last=True,
            prefetch_factor=4,
        )

        import time
        print(f"Testing training loop with batch_size={batch_size}, num_workers={num_workers}...")
        print("Simulating a 2-second forward/backward pass per batch.")

        num_test_batches = 100

        # Warm-up phase
        print("Performing a brief warm-up...")
        start_time = time.time()
        for i, _ in enumerate(dataloader):
            print(f"Warm-up batch {i}..., time: {time.time() - start_time:.2f} seconds")
            start_time = time.time()
            if i >= 10:
                break
            time.sleep(0.5)  # Brief sleep to ensure workers are active

        print("Starting timed test run...")
        start_time = time.time()
        total_samples = 0

        pbar = tqdm(dataloader, total=num_test_batches, desc="[INFO] Simulating Training")

        for i, batch in enumerate(pbar):
            if i >= num_test_batches:
                break
            # Simulate model forward and backward pass
            time.sleep(2)
            total_samples += len(batch['ids'])

        end_time = time.time()

        total_duration = end_time - start_time
        avg_iteration_time = total_duration / num_test_batches
        samples_per_second = total_samples / total_duration if total_duration > 0 else 0

        print("\n" + "=" * 50)
        print("       Training Loop Simulation Summary")
        print("=" * 50)
        print(f"✔️  Simulated {num_test_batches} training steps.")
        print(f"⏱️  Total Time: {total_duration:.2f} seconds")
        print(f"⏱️  Average Iteration Time: {avg_iteration_time:.4f} seconds")
        print(f"   (Includes data loading + 2s simulated computation)")

        if avg_iteration_time > 2.05:  # small buffer for overhead
            data_bottleneck_time = avg_iteration_time - 2.0
            print(f"Warning: Dataloader appears to be a bottleneck.")
            print(f"The model is idle for ~{data_bottleneck_time:.3f}s each step waiting for data.")
        else:
            print(f"Dataloader is keeping up with the simulated model computation.")

        print(f"Overall Throughput: {samples_per_second:.2f} samples/second")
        print("=" * 50 + "\n")