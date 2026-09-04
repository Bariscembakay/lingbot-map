# Code for ZipMap (CVPR 2026); created by Haian Jin

import numpy as np
from PIL import Image, ImageFile

from torch.utils.data import Dataset
from .dataset_util import *
import itertools
Image.MAX_IMAGE_PIXELS = None
ImageFile.LOAD_TRUNCATED_IMAGES = True


class BaseDataset(Dataset):
    """
    Base dataset class for VGGT and VGGSfM training.

    This abstract class handles common operations like image resizing,
    augmentation, and coordinate transformations. Concrete dataset
    implementations should inherit from this class.

    Attributes:
        img_size: Target image size (typically the width)
        patch_size: Size of patches for vit
        augs.scales: Scale range for data augmentation [min, max]
        rescale: Whether to rescale images
        rescale_aug: Whether to apply augmentation during rescaling
        landscape_check: Whether to handle landscape vs portrait orientation
    """
    def __init__(
        self,
        common_conf,
    ):
        """
        Initialize the base dataset with common configuration.

        Args:
            common_conf: Configuration object with the following properties, shared by all datasets:
                - img_size: Default is 518
                - patch_size: Default is 14
                - augs.scales: Default is [0.8, 1.2]
                - rescale: Default is True
                - rescale_aug: Default is True
                - landscape_check: Default is True
        """
        super().__init__()
        self.img_size = common_conf.img_size
        self.patch_size = common_conf.patch_size
        self.aug_scale = common_conf.augs.scales
        self.rescale = common_conf.rescale
        self.rescale_aug = common_conf.rescale_aug
        self.landscape_check = common_conf.landscape_check

    def __len__(self):
        return self.len_train

    def __getitem__(self, idx_N):
        """
        Get an item from the dataset.

        Args:
            idx_N: Tuple containing (seq_index, img_per_seq, aspect_ratio)

        Returns:
            Dataset item as returned by get_data()
        """
        seq_index, img_per_seq, aspect_ratio = idx_N

        return self.get_data(
            seq_index=seq_index, img_per_seq=img_per_seq, aspect_ratio=aspect_ratio
        )

    def get_data(self, seq_index=None, seq_name=None, ids=None, aspect_ratio=1.0):
        """
        Abstract method to retrieve data for a given sequence.

        Args:
            seq_index (int, optional): Index of the sequence
            seq_name (str, optional): Name of the sequence
            ids (list, optional): List of frame IDs
            aspect_ratio (float, optional): Target aspect ratio.

        Returns:
            Dataset-specific data

        Raises:
            NotImplementedError: This method must be implemented by subclasses
        """
        raise NotImplementedError(
            "This is an abstract method and should be implemented in the subclass, i.e., each dataset should implement its own get_data method."
        )

    def get_target_shape(self, aspect_ratio):
        """
        Calculate the target shape based on the given aspect ratio.

        Args:
            aspect_ratio: Target aspect ratio

        Returns:
            numpy.ndarray: Target image shape [height, width]
        """
        short_size = int(self.img_size * aspect_ratio)
        small_size = self.patch_size

        # ensure the input shape is friendly to vision transformer
        if short_size % small_size != 0:
            short_size = (short_size // small_size) * small_size

        image_shape = np.array([short_size, self.img_size])
        return image_shape

    def process_one_image(
        self,
        image,
        depth_map,
        extri_opencv,
        intri_opencv,
        original_size,
        target_image_shape,
        track=None,
        filepath=None,
        safe_bound=4,
    ):
        """
        Process a single image and its associated data.

        This method handles image transformations, depth processing, and coordinate conversions.

        Args:
            image (numpy.ndarray): Input image array
            depth_map (numpy.ndarray): Depth map array
            extri_opencv (numpy.ndarray): Extrinsic camera matrix (OpenCV convention)
            intri_opencv (numpy.ndarray): Intrinsic camera matrix (OpenCV convention)
            original_size (numpy.ndarray): Original image size [height, width]
            target_image_shape (numpy.ndarray): Target image shape after processing
            track (numpy.ndarray, optional): Optional tracking information. Defaults to None.
            filepath (str, optional): Optional file path for debugging. Defaults to None.
            safe_bound (int, optional): Safety margin for cropping operations. Defaults to 4.

        Returns:
            tuple: (
                image (numpy.ndarray): Processed image,
                depth_map (numpy.ndarray): Processed depth map,
                extri_opencv (numpy.ndarray): Updated extrinsic matrix,
                intri_opencv (numpy.ndarray): Updated intrinsic matrix,
                world_coords_points (numpy.ndarray): 3D points in world coordinates,
                cam_coords_points (numpy.ndarray): 3D points in camera coordinates,
                point_mask (numpy.ndarray): Boolean mask of valid points,
                track (numpy.ndarray, optional): Updated tracking information
            )
        """
        # Make copies to avoid in-place operations affecting original data
        image = np.copy(image)
        depth_map = np.copy(depth_map)
        extri_opencv = np.copy(extri_opencv)
        intri_opencv = np.copy(intri_opencv)
        if track is not None:
            track = np.copy(track)

        # Apply random scale augmentation during training if enabled
        if self.training and self.aug_scale:
            random_h_scale, random_w_scale = np.random.uniform(
                self.aug_scale[0], self.aug_scale[1], 2
            )
            # Avoid random padding by capping at 1.0
            random_h_scale = min(random_h_scale, 1.0)
            random_w_scale = min(random_w_scale, 1.0)
            aug_size = original_size * np.array([random_h_scale, random_w_scale])
            aug_size = aug_size.astype(np.int32)
        else:
            aug_size = original_size

        # Move principal point to the image center and crop if necessary
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, aug_size, track=track, filepath=filepath,
        )

        original_size = np.array(image.shape[:2])  # update original_size
        target_shape = target_image_shape
        # Handle landscape vs. portrait orientation
        rotate_to_portrait = False
        if self.landscape_check:
            # Switch between landscape and portrait if necessary
            if original_size[0] > 1.25 * original_size[1]:
                if (target_image_shape[0] != target_image_shape[1]) and (np.random.rand() > 0.5):
                    target_shape = np.array([target_image_shape[1], target_image_shape[0]])
                    rotate_to_portrait = True

        # Resize images and update intrinsics
        if self.rescale:
            image, depth_map, intri_opencv, track = resize_image_depth_and_intrinsic(
                image, depth_map, intri_opencv, target_shape, original_size, track=track,
                safe_bound=safe_bound,
                rescale_aug=self.rescale_aug
            )
        else:
            print("Not rescaling the images")

        # Ensure final crop to target shape
        image, depth_map, intri_opencv, track = crop_image_depth_and_intrinsic_by_pp(
            image, depth_map, intri_opencv, target_shape, track=track, filepath=filepath, strict=True,
        )

        # Apply 90-degree rotation if needed
        if rotate_to_portrait:
            assert self.landscape_check
            clockwise = np.random.rand() > 0.5
            image, depth_map, extri_opencv, intri_opencv, track = rotate_90_degrees(
                image,
                depth_map,
                extri_opencv,
                intri_opencv,
                clockwise=clockwise,
                track=track,
            )
        
        # Convert depth to world and camera coordinates
        world_coords_points, cam_coords_points, point_mask = (
            depth_to_world_coords_points(depth_map, extri_opencv, intri_opencv)
        )

        return (
            image,
            depth_map,
            extri_opencv,
            intri_opencv,
            world_coords_points,
            cam_coords_points,
            point_mask,
            track,
        )

    def get_nearby_ids(self, ids, full_seq_num, expand_ratio=None, expand_range=None):
        """
        TODO: add the function to sample the ids by pose similarity ranking.

        Sample a set of IDs from a sequence close to a given start index.

        You can specify the range either as a ratio of the number of input IDs
        or as a fixed integer window.


        Args:
            ids (list): Initial list of IDs. The first element is used as the anchor.
            full_seq_num (int): Total number of items in the full sequence.
            expand_ratio (float, optional): Factor by which the number of IDs expands
                around the start index. Default is 2.0 if neither expand_ratio nor
                expand_range is provided.
            expand_range (int, optional): Fixed number of items to expand around the
                start index. If provided, expand_ratio is ignored.

        Returns:
            numpy.ndarray: Array of sampled IDs, with the first element being the
                original start index.

        Examples:
            # Using expand_ratio (default behavior)
            # If ids=[100,101,102] and full_seq_num=200, with expand_ratio=2.0,
            # expand_range = int(3 * 2.0) = 6, so IDs sampled from [94...106] (if boundaries allow).

            # Using expand_range directly
            # If ids=[100,101,102] and full_seq_num=200, with expand_range=10,
            # IDs are sampled from [90...110] (if boundaries allow).

        Raises:
            ValueError: If no IDs are provided.
        """
        if len(ids) == 0:
            raise ValueError("No IDs provided.")

        if expand_range is None and expand_ratio is None:
            expand_ratio = 2.0  # Default behavior

        total_ids = len(ids)
        start_idx = ids[0]

        # Determine the actual expand_range
        if expand_range is None:
            # Use ratio to determine range
            expand_range = int(total_ids * expand_ratio)

        # Calculate valid boundaries
        low_bound = max(0, start_idx - expand_range)
        high_bound = min(full_seq_num, start_idx + expand_range)

        # Create the valid range of indices
        valid_range = np.arange(low_bound, high_bound)

        # Sample 'total_ids - 1' items, because we already have the start_idx
        sampled_ids = np.random.choice(
            valid_range,
            size=(total_ids - 1),
            replace=True,   # we accept the situation that some sampled ids are the same
        )

        # Insert the start_idx at the beginning
        result_ids = np.insert(sampled_ids, 0, start_idx)

        return result_ids

    @staticmethod
    def blockwise_shuffle(x, block_shuffle):
        if block_shuffle is None:
            return np.random.permutation(x).tolist()
        else:
            assert block_shuffle > 0
            blocks = [x[i : i + block_shuffle] for i in range(0, len(x), block_shuffle)]
            shuffled_blocks = [np.random.permutation(block).tolist() for block in blocks]
            shuffled_list = [item for block in shuffled_blocks for item in block]
            return shuffled_list

    def get_seq_from_start_id(
        self,
        num_views,
        id_ref,
        ids_all,
        min_interval=1,
        max_interval=25,
        video_prob=0.5,
        fix_interval_prob=0.5,
        block_shuffle=None,
    ):
        """
        args:
            num_views: number of views to return
            id_ref: the reference id (first id)
            ids_all: all the ids
            max_interval: maximum interval between two views
        returns:
            pos: list of positions of the views in ids_all, i.e., index for ids_all
            is_video: True if the views are consecutive
        """
        assert min_interval > 0, f"min_interval should be > 0, got {min_interval}"
        assert (
            min_interval <= max_interval
        ), f"min_interval should be <= max_interval, got {min_interval} and {max_interval}"
        assert id_ref in ids_all
        pos_ref = ids_all.index(id_ref)
        all_possible_pos = np.arange(pos_ref, len(ids_all))
        remaining_sum = len(ids_all) - 1 - pos_ref

        if remaining_sum >= num_views - 1:
            if remaining_sum == num_views - 1:
                assert ids_all[-num_views] == id_ref
                return [pos_ref + i for i in range(num_views)], True
            max_interval = min(max_interval, 2 * remaining_sum // (num_views - 1)) # E(uniform_sample(2N)) = N
            intervals = [
                np.random.choice(range(min_interval, max_interval + 1))
                for _ in range(num_views - 1)
            ]

            # if video or collection
            if np.random.rand() < video_prob:
                # if fixed interval or random
                if np.random.rand() < fix_interval_prob:
                    # regular interval
                    fixed_interval = np.random.choice(
                        range(
                            1,
                            min(remaining_sum // (num_views - 1) + 1, max_interval + 1),
                        )
                    )
                    intervals = [fixed_interval for _ in range(num_views - 1)]
                is_video = True
            else:
                is_video = False

            pos = list(itertools.accumulate([pos_ref] + intervals))
            pos = [p for p in pos if p < len(ids_all)]
            pos_candidates = [p for p in all_possible_pos if p not in pos]
            pos = (
                pos
                + np.random.choice(
                    pos_candidates, num_views - len(pos), replace=False
                ).tolist()
            )

            pos = (
                sorted(pos)
                if is_video
                else self.blockwise_shuffle(pos, block_shuffle)
            )
        else:
            uniq_num = remaining_sum
            new_pos_ref = np.random.choice(np.arange(pos_ref + 1))
            new_remaining_sum = len(ids_all) - 1 - new_pos_ref
            new_max_interval = min(max_interval, new_remaining_sum // (uniq_num - 1))
            new_intervals = [
                np.random.choice(range(1, new_max_interval + 1)) for _ in range(uniq_num - 1)
            ]

            revisit_random = np.random.rand()
            video_random = np.random.rand()

            if np.random.rand() < fix_interval_prob and video_random < video_prob:
                # regular interval
                fixed_interval = np.random.choice(range(1, new_max_interval + 1))
                new_intervals = [fixed_interval for _ in range(uniq_num - 1)]
            pos = list(itertools.accumulate([new_pos_ref] + new_intervals))

            is_video = False
            if revisit_random < 0.5 or video_prob == 1.0:  # revisit, video / collection
                is_video = video_random < video_prob
                pos = (
                    self.blockwise_shuffle(pos, block_shuffle)
                    if not is_video
                    else pos
                )
                num_full_repeat = num_views // uniq_num
                pos = (
                    pos * num_full_repeat
                    + pos[: num_views - len(pos) * num_full_repeat]
                )
            elif revisit_random < 0.9:  # random
                pos = np.random.choice(pos, num_views, replace=True)
            else:  # ordered
                pos = sorted(np.random.choice(pos, num_views, replace=True))
        assert len(pos) == num_views
        return pos, is_video