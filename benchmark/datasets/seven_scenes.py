"""7Scenes dataset loader.

Dataset format:
  Source format:
    {raw_data_root}/
      {scene_name}/
        seq-{XX}/
          frame-{NNNNNN}.color.png    # RGB image (640x480)
          frame-{NNNNNN}.depth.png    # Depth map (16-bit PNG, millimeters)
          frame-{NNNNNN}.pose.txt     # 4x4 C2W matrix
        TrainSplit.txt                # Training sequence list
        TestSplit.txt                 # Testing sequence list

Reference:
  https://www.microsoft.com/en-us/research/project/rgb-d-dataset-7-scenes/
"""

import numpy as np
from pathlib import Path
from PIL import Image
from typing import List, Dict, Any, Optional
from benchmark.core.loader import BSSLoader
from benchmark.dataset.base import BaseDataset

# Estimated Kinect calibration, from zinsmatt/7-Scenes-Calibration (intrinsics
# from vanishing points, depth->RGB extrinsics from point correspondences; the
# two cameras are assumed to share orientation, so the extrinsic is a pure
# translation).  Microsoft ship no official parameters -- "the RGB and depth
# camera have not been calibrated and we can't provide calibration parameters".
DEPTH_INTRINSICS_7SCENES = {'fx': 598.84, 'fy': 587.62, 'cx': 320.0, 'cy': 240.0}
DEPTH_TO_COLOR_T = np.array([0.023449, 0.006177, 0.010525], dtype=np.float64)

# 7Scenes fixed intrinsics
INTRINSICS_7SCENES = {
    'fx': 525.0,
    'fy': 525.0,
    'cx': 320.0,
    'cy': 240.0,
    'width': 640,
    'height': 480
}


class SevenScenesDataset(BaseDataset):
    """7Scenes dataset loader."""

    def __init__(self, raw_data_root: str, split: str = 'test',
                 register_depth: bool = False, logger=None):
        """Initialize 7Scenes dataset loader.

        Args:
            raw_data_root: Dataset root directory
            split: 'train' or 'test' (default: 'test')
            register_depth: Warp raw depth into the colour camera frame before
                use. The raw `.depth.png` is in the DEPTH camera frame, but the
                whole pipeline unprojects it with the colour intrinsics -- a
                radial error (0 at the principal point, ~15 cm at the corners
                at 2 m) that a global Umeyama+ICP cannot absorb. Upstream reads
                `.depth.proj.png` instead, i.e. already-registered depth that
                the Pi3 data prep is supposed to emit and that the raw
                Microsoft distribution does not ship.
            logger: Optional logger instance
        """
        super().__init__(raw_data_root, logger=logger)
        self.split = split
        self.register_depth = register_depth

    def get_scenes(self) -> List[str]:
        """Get all sequence names (per sequence, not per scene).

        Returns:
            List of sequence identifiers in format 'scene_name/seq-XX'

        Note:
            Each sequence is treated as a separate scene in the benchmark.
        """
        scenes = []
        for scene_dir in sorted(self.raw_data_root.iterdir()):
            if scene_dir.is_dir() and not scene_dir.name.startswith('.'):
                scene_name = scene_dir.name
                # Get all sequences for this scene
                scene_dir = self.raw_data_root / scene_name
                split_file = scene_dir / f"{self.split.capitalize()}Split.txt"

                # raw_data_root can be shared with unrelated projects (e.g. a
                # stray eval_gt/ dir) -- skip anything that isn't a real scene
                # rather than crashing the whole run.
                if not split_file.exists():
                    continue

                # Read split file
                with open(split_file, 'r') as f:
                    sequences = [line.strip() for line in f if line.strip()]

                # Convert 'sequence1' to 'seq-01'
                for seq in sequences:    
                    scenes.append(f"{scene_name}/seq-{int(seq.replace('sequence', '')):02d}")
                
        return scenes

    def get_frame_list(self, scene: str) -> List[int]:
        """Get all frame identifiers for sequence.

        Args:
            scene: Sequence name (format: 'chess/seq-01')

        Returns:
            List of frame_ids (integers)
        """
        # Parse scene and sequence names
        scene_name, seq = scene.split('/')
        seq_dir = self.raw_data_root / scene_name / seq

        # Collect all frames for this sequence
        all_frames = []
        for file in sorted(seq_dir.glob('frame-*.color.png')):
            frame_id = int(file.stem.split('-')[1].split('.')[0])
            all_frames.append(frame_id)

        return sorted(all_frames)

    def load_frame_data(self, scene: str, frame_id: int) -> Dict[str, Any]:
        """Load single frame data.

        Args:
            scene: Sequence name (format: 'chess/seq-01')
            frame_id: Frame number (integer)

        Returns:
            Dictionary containing:
            - timestamp: float (using frame_id as timestamp)
            - rgb: HxWx3 RGB image (uint8, 0-255)
            - depth: HxW depth map (float32, meters)
            - pose: 4x4 C2W transformation matrix
            - intrinsics: [fx, fy, cx, cy] tuple
        """
        # Parse scene and sequence names
        scene_name, seq = scene.split('/')
        seq_dir = self.raw_data_root / scene_name / seq

        # File paths
        color_file = seq_dir / f"frame-{frame_id:06d}.color.png"
        depth_file = seq_dir / f"frame-{frame_id:06d}.depth.png"
        pose_file = seq_dir / f"frame-{frame_id:06d}.pose.txt"

        # Check if files exist
        if not all([color_file.exists(), depth_file.exists(), pose_file.exists()]):
            raise FileNotFoundError(f"Missing files for {scene}/frame-{frame_id:06d}")

        # Load data
        rgb = self._load_rgb(color_file)
        depth = self._load_depth(depth_file)
        if self.register_depth:
            depth = self._register_depth_to_color(depth)
        c2w = self._load_pose(pose_file)

        return {
            'rgb': rgb,
            'depth': depth,
            'pose': c2w,
            'intrinsics': np.array([
                INTRINSICS_7SCENES['fx'],
                INTRINSICS_7SCENES['fy'],
                INTRINSICS_7SCENES['cx'],
                INTRINSICS_7SCENES['cy']
            ])
        }

    def load_global_data(self, scene: str) -> Dict[str, Any]:
        """Load global scene-level data.

        Args:
            scene: Sequence name

        Returns:
            Empty dictionary (7Scenes doesn't provide global point clouds)

        Note:
            7Scenes dataset doesn't include pre-computed point clouds.
            Point clouds can be generated from depth maps during evaluation.
        """
        return {}

    @staticmethod
    def _load_rgb(color_file: Path) -> np.ndarray:
        """Load RGB image.

        Args:
            color_file: Path to RGB image file

        Returns:
            HxWx3 RGB image (uint8, 0-255)
        """
        img = Image.open(color_file).convert('RGB')
        return np.array(img, dtype=np.uint8)
    
    @staticmethod
    def _load_depth(depth_file: Path) -> np.ndarray:
        """Load depth map.

        7Scenes depth maps are 16-bit PNG in millimeters, need conversion to meters.

        Args:
            depth_file: Path to depth image file

        Returns:
            HxW depth map (float32, meters)
        """
        depthmap = np.array(Image.open(depth_file), dtype=np.float32)

        # Validate shape
        expected_shape = (INTRINSICS_7SCENES['height'], INTRINSICS_7SCENES['width'])
        if depthmap.shape != expected_shape:
            raise ValueError(
                f"Depth map shape {depthmap.shape} does not match "
                f"expected {expected_shape}"
            )

        # Convert from millimeters to meters
        depthmap[depthmap == 65535] = 0  # Invalid depth marker
        depthmap = np.nan_to_num(depthmap, 0.0) / 1000.0

        # Filter invalid depth values
        depthmap[depthmap > 10.0] = 0    # Too far (>10m)
        depthmap[depthmap < 1e-3] = 0    # Too near (<1mm)

        return depthmap

    @staticmethod
    def _register_depth_to_color(depth: np.ndarray) -> np.ndarray:
        """Forward-warp depth from the depth camera into the colour camera frame.

        Unproject with the DEPTH intrinsics, translate into the colour frame,
        reproject with the COLOUR intrinsics. Occlusions are resolved by writing
        far points first so nearer ones overwrite them.

        The translation sign was fixed empirically, not guessed: see
        `.agents/scratch/reproduction/check_depth_registration.py`, which scores
        RGB gradient magnitude at depth discontinuities. Negating the published
        translation improved that score on every sequence tested; the opposite
        sign was indistinguishable from doing nothing.
        """
        H, W = depth.shape
        v, u = np.nonzero(depth > 0)
        if v.size == 0:
            return np.zeros_like(depth)
        d = depth[v, u].astype(np.float64)

        x = (u - DEPTH_INTRINSICS_7SCENES['cx']) / DEPTH_INTRINSICS_7SCENES['fx'] * d
        y = (v - DEPTH_INTRINSICS_7SCENES['cy']) / DEPTH_INTRINSICS_7SCENES['fy'] * d
        x, y, z = (x - DEPTH_TO_COLOR_T[0],
                   y - DEPTH_TO_COLOR_T[1],
                   d - DEPTH_TO_COLOR_T[2])

        ok = z > 1e-6
        x, y, z = x[ok], y[ok], z[ok]
        uc = np.rint(INTRINSICS_7SCENES['fx'] * x / z + INTRINSICS_7SCENES['cx']).astype(np.int32)
        vc = np.rint(INTRINSICS_7SCENES['fy'] * y / z + INTRINSICS_7SCENES['cy']).astype(np.int32)

        inb = (uc >= 0) & (uc < W) & (vc >= 0) & (vc < H)
        uc, vc, z = uc[inb], vc[inb], z[inb]

        order = np.argsort(-z)
        out = np.zeros((H, W), dtype=np.float32)
        out[vc[order], uc[order]] = z[order].astype(np.float32)
        return out
    
    @staticmethod
    def _load_pose(pose_file: Path) -> np.ndarray:
        """Load pose file (4x4 C2W matrix).

        Args:
            pose_file: Path to pose text file

        Returns:
            4x4 C2W transformation matrix
        """
        pose = np.loadtxt(pose_file).reshape(4, 4)
        return pose.astype(np.float32)
    
    @staticmethod
    def evaluate_pointcloud(
        gt_loader: BSSLoader,
        pred_loader: BSSLoader,
        logger,
        options: Optional[Dict[str, Any]] = None
    ) -> Optional[Dict[str, Any]]:
        """Evaluate point cloud reconstruction for 7Scenes.

        Alignment uses align_gt_loader (gt resized to pred resolution) paired
        with pred_loader so that pixel-wise correspondences are valid.
        Final evaluation compares the aligned pred points against the full-
        resolution gt point cloud from gt_loader.

        Args:
            gt_loader:   BSSLoader for ground truth (native resolution, no resize)
            pred_loader: BSSLoader for method output
            logger:      Logger instance
            options:     Optional dict; supported keys:
                           icp_threshold (float, default 0.1)

        Returns:
            Dictionary with point cloud metrics and point clouds, or None on failure
        """
        from benchmark.geometry.registration import umeyama_registration, icp_registration, apply_transform, voxel_downsample
        from benchmark.evaluation.points import evaluate_pointcloud as eval_pc

        icp_threshold  = (options or {}).get('icp_threshold', 0.1)
        voxel_size     = (options or {}).get('voxel_size', 4.0 / 512.0)
        conf_threshold = (options or {}).get('conf_threshold', 0.0)

        # --- Load point clouds for Umeyama (pixel-wise correspondences required) ---
        gt_xyzrgb_for_umeyama, gt_mask_for_umeyama = gt_loader.load_point_cloud_grid()
        pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid(confidence_threshold=conf_threshold)
        # pred_xyzrgb, pred_mask = pred_loader.load_point_cloud_grid()

        # Subset GT frame axis to pred's keyframe indices so shapes match.
        # For sparse SLAM, get_frame_indices() returns e.g. [5, 12, 18, ...];
        # for dense methods it returns [0, 1, ..., N-1] (identity, no-op).
        pred_frame_indices = pred_loader.get_frame_indices()
        gt_xyzrgb_for_umeyama = gt_xyzrgb_for_umeyama[pred_frame_indices]
        gt_mask_for_umeyama = gt_mask_for_umeyama[pred_frame_indices]

        # Pixel-wise correspondence: only pixels valid in both grids
        common_mask = gt_mask_for_umeyama & pred_mask
        gt_pts_for_umeyama = gt_xyzrgb_for_umeyama[common_mask][:, :3]
        pred_pts_for_umeyama = pred_xyzrgb[common_mask][:, :3]

        logger.info(f"Umeyama alignment with {len(gt_pts_for_umeyama)} corresponding points")
        T_umeyama = umeyama_registration(
            source_points=pred_pts_for_umeyama,
            target_points=gt_pts_for_umeyama,
        )
        logger.info(f"Umeyama transform:\n{T_umeyama}")

        # --- Load full GT point cloud for ICP and evaluation ---
        gt_xyzrgb, gt_mask = gt_loader.load_point_cloud_grid()
        gt_pts = gt_xyzrgb[gt_mask][:, :3]

        # ICP uses all pred points (after Umeyama) vs full GT cloud.
        # No pixel-wise correspondence needed; denser target improves convergence.
        # NOTE: We also filter out gt_mask from the predicted point clouds for better accuracy metrics
        # (consistent with Pi3 evaluation). This applies to all methods, ensuring fair comparison.
        all_pred_pts = pred_xyzrgb[common_mask][:, :3]
        all_pred_after_umeyama = apply_transform(all_pred_pts, T_umeyama)

        # Voxel downsample once — used for both ICP and eval_pc (same as DA3 bench)
        if voxel_size > 0:
            logger.info(
                f"Voxel downsampling at {voxel_size:.6f}m "
                f"(pred: {len(all_pred_after_umeyama):,}, gt: {len(gt_pts):,})"
            )
            pred_ds = voxel_downsample(all_pred_after_umeyama, voxel_size)
            gt_ds   = voxel_downsample(gt_pts, voxel_size)
            logger.info(f"After downsampling: pred={len(pred_ds):,}, gt={len(gt_ds):,}")
        else:
            pred_ds = all_pred_after_umeyama
            gt_ds   = gt_pts

        logger.info(f"ICP alignment with threshold {icp_threshold}")
        T_icp = icp_registration(
            source_points=pred_ds,
            target_points=gt_ds,
            icp_threshold=icp_threshold,
        )
        logger.info(f"ICP transform:\n{T_icp}")

        pred_pts_eval = apply_transform(pred_ds, T_icp)
        gt_pts_eval   = gt_ds

        logger.info(
            f"Evaluating: {len(pred_pts_eval)} pred points "
            f"vs {len(gt_pts_eval)} gt points"
        )
        results = eval_pc(
            source_points=pred_pts_eval,
            target_points=gt_pts_eval,
            thresholds=[0.05]
        )

        return results