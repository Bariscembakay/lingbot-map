#!/usr/bin/env python3
"""
Preprocess OmniWorld-Game dataset by reading already-extracted data and organizing camera information.

This script:
1. Reads split_info.json and camera/*.json files from already-extracted scene directories
2. Organizes all camera poses and intrinsics into a single cameras.npy file
3. Maintains the same structure as GTA-SfM preprocessing for compatibility

Example:
    python preprocess_omniworld_game.py \
        --src /home/storage-bucket/haian/zipmap_data/OmniWorld-11k/OmniWorld-Game \
        --out /home/storage-bucket/haian/zipmap_data/OmniWorld-11k/OmniWorld-Game \

Requires: numpy, scipy, tqdm
    pip install numpy scipy tqdm
"""
from __future__ import annotations
import argparse
import json
import csv
from pathlib import Path
from typing import Dict, Any, Tuple
import logging
import numpy as np
from scipy.spatial.transform import Rotation as R
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed

logging.basicConfig(
    level=logging.INFO,
    format='[%(levelname)s] %(message)s'
)


def load_split_info(scene_dir: Path) -> Dict[str, Any]:
    """Load split_info.json from scene directory."""
    split_info_path = scene_dir / "split_info.json"
    if not split_info_path.exists():
        raise FileNotFoundError(f"split_info.json not found at {split_info_path}")

    with open(split_info_path, "r", encoding="utf-8") as f:
        return json.load(f)


def load_camera_poses(scene_dir: Path, split_idx: int, frame_count: int) -> Tuple[np.ndarray, np.ndarray]:
    """
    Load camera intrinsics and extrinsics for a given split.

    Returns:
        intrinsics: (S, 3, 3) array, pixel-space K matrices
        extrinsics: (S, 4, 4) array, OpenCV world-to-camera matrices
    """
    cam_file = scene_dir / "camera" / f"split_{split_idx}.json"
    if not cam_file.exists():
        raise FileNotFoundError(f"Camera file not found: {cam_file}")

    with open(cam_file, "r", encoding="utf-8") as f:
        cam = json.load(f)

    # Build intrinsics
    intrinsics = np.repeat(np.eye(3)[None, ...], frame_count, axis=0)
    intrinsics[:, 0, 0] = cam["focals"]  # fx
    intrinsics[:, 1, 1] = cam["focals"]  # fy
    intrinsics[:, 0, 2] = cam["cx"]      # cx
    intrinsics[:, 1, 2] = cam["cy"]      # cy

    # Build extrinsics (world-to-camera)
    extrinsics = np.repeat(np.eye(4)[None, ...], frame_count, axis=0)

    # Convert quaternions from (w, x, y, z) to (x, y, z, w) for scipy
    quat_wxyz = np.array(cam["quats"])  # (S, 4) (w,x,y,z)
    quat_xyzw = np.concatenate([quat_wxyz[:, 1:], quat_wxyz[:, :1]], axis=1)

    rotations = R.from_quat(quat_xyzw).as_matrix()  # (S, 3, 3)
    translations = np.array(cam["trans"])           # (S, 3)

    extrinsics[:, :3, :3] = rotations
    extrinsics[:, :3, 3] = translations

    return intrinsics, extrinsics


def process_scene(scene_dir: Path, scene_id: str) -> Tuple[str, Dict[str, Any]]:
    """
    Process a single scene: read split_info.json and all camera data.

    Returns:
        (scene_id, scene_data_dict)
        where scene_data_dict contains per-split camera data
    """
    try:
        split_info = load_split_info(scene_dir)
        split_num = split_info["split_num"]

        scene_data = {}
        for split_idx in range(split_num):
            try:
                frame_indices = split_info["split"][split_idx]
                frame_count = len(frame_indices)

                intrinsics, extrinsics = load_camera_poses(scene_dir, split_idx, frame_count)

                # Create basenames from frame indices (zero-padded to 6 digits)
                basenames = [f"{idx:06d}" for idx in frame_indices]

                world2cam = extrinsics

                scene_data[f"split_{split_idx}"] = {
                    "basenames": np.array(basenames),
                    "K": intrinsics,
                    "pose": world2cam,  # world2cam poses
                    "frame_indices": np.array(frame_indices, dtype=np.int32),
                }
            except Exception as e:
                logging.warning(f"Failed to load split {split_idx} for scene {scene_id}: {e}")
                continue

        return scene_id, scene_data
    except Exception as e:
        logging.error(f"Failed to process scene {scene_id}: {e}")
        return scene_id, {}


def main():
    ap = argparse.ArgumentParser(
        description="Preprocess OmniWorld-Game dataset: read camera data from extracted scenes"
    )
    ap.add_argument(
        "--src",
        required=True,
        type=Path,
        help="Root directory of extracted OmniWorld-Game data (contains scene folders)"
    )
    ap.add_argument(
        "--out",
        required=True,
        type=Path,
        help="Output directory for processed data (will save cameras.npy here)"
    )
    ap.add_argument(
        "--workers",
        type=int,
        default=96,
        help="Number of parallel workers (default: 96)"
    )
    ap.add_argument(
        "--metadata-csv",
        type=Path,
        default=None,
        help="Path to OmniWorld-Game_metadata.csv (optional, for train/test split assignment)"
    )
    args = ap.parse_args()

    src_root: Path = args.src
    out_root: Path = args.out
    out_root.mkdir(parents=True, exist_ok=True)

    # Load metadata if provided (to determine test splits)
    scene_test_splits = {}  # scene_id -> list of test split indices
    if args.metadata_csv and args.metadata_csv.exists():
        logging.info(f"Loading metadata from {args.metadata_csv}")
        with open(args.metadata_csv, "r", encoding="utf-8") as f:
            reader = csv.DictReader(f)
            for row in reader:
                scene_id = row["UID"]
                test_split_str = row.get("Test Split Index", "").strip()
                if test_split_str:
                    # Parse comma-separated indices
                    test_splits = [int(x.strip()) for x in test_split_str.split(",")]
                    scene_test_splits[scene_id] = test_splits

    # Find all scene directories
    scene_dirs = sorted([d for d in src_root.iterdir() if d.is_dir() and (d / "split_info.json").exists()])
    
    # Data source errors, OmniWorld-Game may have fixed this issue in the future
    for scene_dir in scene_dirs:
        if scene_dir.name == "05017c153ac9" or scene_dir.name == "05346c2f4648":
            print(f"Skipping problematic scene: {scene_dir.name}")
            scene_dirs.remove(scene_dir)
    
    for scene_dir in scene_dirs:
        if scene_dir.name == "05017c153ac9" or scene_dir.name == "05346c2f4648":
            print(f"Failed to remove problematic scene: {scene_dir.name}")

    if not scene_dirs:
        logging.error(f"No scene directories with split_info.json found in {src_root}")
        return

    logging.info(f"Found {len(scene_dirs)} scenes to process")

    # Dictionary to store all camera data: dataset_split -> scene_id -> split_data
    all_cameras_data = {
        "train": {},
        "test": {}
    }

    # Process scenes in parallel
    scene_results = {}
    with ProcessPoolExecutor(max_workers=args.workers) as executor:
        futures = {executor.submit(process_scene, scene_dir, scene_dir.name): scene_dir.name
                   for scene_dir in scene_dirs}

        with tqdm(total=len(futures), desc="Processing scenes", unit="scene") as pbar:
            for future in as_completed(futures):
                scene_id = futures[future]
                try:
                    scene_id_ret, scene_data = future.result()
                    if scene_data:
                        scene_results[scene_id_ret] = scene_data
                        pbar.set_postfix_str(f"{scene_id_ret}: {len(scene_data)} splits")
                except Exception as e:
                    logging.error(f"Error processing scene {scene_id}: {e}")
                    pbar.set_postfix_str(f"ERROR: {e}")
                finally:
                    pbar.update(1)

    # Organize scenes into train/test based on metadata
    # If no metadata or no test split info for a scene, put all splits in train
    for scene_id, scene_data in scene_results.items():
        test_split_indices = scene_test_splits.get(scene_id, [])

        if test_split_indices:
            # Split this scene's data into train and test
            train_splits = {}
            test_splits = {}

            for split_key, split_data in scene_data.items():
                split_idx = int(split_key.split("_")[1])
                if split_idx in test_split_indices:
                    test_splits[split_key] = split_data
                else:
                    train_splits[split_key] = split_data

            if train_splits:
                all_cameras_data["train"][scene_id] = train_splits
            if test_splits:
                all_cameras_data["test"][scene_id] = test_splits
        else:
            # No test split info, put everything in train
            all_cameras_data["train"][scene_id] = scene_data

    # Save all camera data to a single npz file
    cameras_path = out_root / "omniworld_game_metadata.npz"
    logging.info(f"Saving camera data to {cameras_path}")
    np.savez(cameras_path, data=all_cameras_data)

    # Print summary
    total_train_scenes = len(all_cameras_data["train"])
    total_test_scenes = len(all_cameras_data["test"])
    total_train_splits = sum(len(scene_data) for scene_data in all_cameras_data["train"].values())
    total_test_splits = sum(len(scene_data) for scene_data in all_cameras_data["test"].values())

    logging.info("[DONE]")
    logging.info(f"  Train scenes: {total_train_scenes} ({total_train_splits} scene-splits)")
    logging.info(f"  Test scenes: {total_test_scenes} ({total_test_splits} scene-splits)")
    logging.info(f"  Output directory: {out_root}")
    logging.info(f"  Camera data: {cameras_path}")


if __name__ == "__main__":
    main()
