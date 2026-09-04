#!/usr/bin/env python3
"""
Extract images, depths, and camera data from GTA-SfM HDF5 files **with multiprocessing**.

- Images are saved as PNG
- Depths are saved as EXR (32-bit float)
- All camera intrinsics (K) and poses are saved into a single NPZ

Example:
    python extract_gta_sfm_hdf5.py \
        --src /share/phoenix/nfs06/S9/hj453/DATA/GTA-SfM/GTA-SfM \
        --out /share/phoenix/nfs06/S9/hj453/DATA/GTA-SfM/GTA-SfM_extracted \
        --splits train test \
        --workers 8

Requires: h5py, numpy, opencv-python (or opencv-python-headless), tqdm
    pip install h5py numpy opencv-python tqdm

Notes:
- This script assumes each HDF5 file has datasets named image_{i}, depth_{i}, K_{i}, pose_{i}.
- image_{i} is assumed to be an encoded image buffer (e.g., JPEG/PNG) that OpenCV can decode.
- We parallelize **per file** (safer with h5py). Each process opens one HDF5 and writes its frames to disk.
- If your node is heavily threaded, consider limiting OpenCV/BLAS threads via env vars, e.g. OMP_NUM_THREADS=1.
"""
from __future__ import annotations
import argparse
from pathlib import Path
from typing import List, Dict, Any, Tuple
import os
os.environ["OPENCV_IO_ENABLE_OPENEXR"] = "1"
import numpy as np
import cv2
import h5py
from tqdm import tqdm
from concurrent.futures import ProcessPoolExecutor, as_completed


def find_frame_indices(h5: h5py.File) -> List[int]:
    idxs = []
    for k in h5.keys():
        if k.startswith("image_"):
            try:
                idxs.append(int(k.split("_")[1]))
            except (IndexError, ValueError):
                pass
    idxs = sorted(set(idxs))
    return idxs


def ensure_dir(p: Path) -> None:
    p.mkdir(parents=True, exist_ok=True)


def decode_image(buf: np.ndarray) -> np.ndarray:
    # Ensure uint8 1-D for cv2.imdecode
    if buf.dtype != np.uint8:
        buf = buf.astype(np.uint8)
    buf = buf.reshape(-1)
    img = cv2.imdecode(buf, cv2.IMREAD_COLOR)
    if img is None:
        raise ValueError("cv2.imdecode returned None (unsupported/invalid image buffer)")
    return img


def save_exr(path: Path, depth: np.ndarray) -> None:
    depth = np.asarray(depth)
    if depth.dtype != np.float32:
        depth = depth.astype(np.float32)
    # Replace NaNs/Infs with zeros to avoid writer issues
    bad = ~np.isfinite(depth)
    if bad.any():
        depth = depth.copy()
        depth[bad] = 0.0
    params: List[int] = []
    if hasattr(cv2, "IMWRITE_EXR_TYPE") and hasattr(cv2, "IMWRITE_EXR_TYPE_FLOAT"):
        params = [int(cv2.IMWRITE_EXR_TYPE), int(cv2.IMWRITE_EXR_TYPE_FLOAT)]
    ok = cv2.imwrite(str(path), depth, params)
    if not ok:
        raise IOError(f"Failed to write EXR: {path}")


def _process_hdf5_file(h5_path: Path, out_split_dir: Path, rel_split: str) -> Tuple[List[str], List[str], List[np.ndarray], List[np.ndarray], List[str], List[str], List[int], str, int]:
    """Worker: process a single HDF5 file, return per-frame lists.
    Returns (img_paths, depth_paths, Ks, poses, seq_names, splits, frame_ids, filename, num_frames_written)
    """
    seq_name = h5_path.stem
    out_img_dir = out_split_dir / seq_name / "images"
    out_depth_dir = out_split_dir / seq_name / "depths"
    ensure_dir(out_img_dir)
    ensure_dir(out_depth_dir)

    img_paths: List[str] = []
    depth_paths: List[str] = []
    Ks: List[np.ndarray] = []
    poses: List[np.ndarray] = []
    seq_names: List[str] = []
    splits: List[str] = []
    frame_ids: List[int] = []
    frames_written = 0

    with h5py.File(h5_path, "r") as h5:
        idxs = find_frame_indices(h5)
        for i in idxs:
            img_key = f"image_{i}"
            depth_key = f"depth_{i}"
            K_key = f"K_{i}"
            pose_key = f"pose_{i}"
            if img_key not in h5 or depth_key not in h5 or K_key not in h5 or pose_key not in h5:
                print(f"[WARN] Missing datasets for frame {i} in {h5_path}, skipping")
                continue
            # image
            img_buf = np.array(h5[img_key][:])
            img = decode_image(img_buf)
            img_fname = f"{i:06d}.png"
            img_out_path = out_img_dir / img_fname
            if not cv2.imwrite(str(img_out_path), img):
                raise IOError(f"Failed to write PNG: {img_out_path}")
            # depth
            depth = np.array(h5[depth_key][:])
            depth_fname = f"{i:06d}.exr"
            depth_out_path = out_depth_dir / depth_fname
            save_exr(depth_out_path, depth)
            # camera
            K = np.array(h5[K_key][:]).astype(np.float32)
            pose = np.array(h5[pose_key][:]).astype(np.float32)

            img_rel = str((Path(rel_split) / seq_name / "images" / img_fname).as_posix())
            depth_rel = str((Path(rel_split) / seq_name / "depths" / depth_fname).as_posix())

            img_paths.append(img_rel)
            depth_paths.append(depth_rel)
            Ks.append(K)
            poses.append(pose)
            seq_names.append(seq_name)
            splits.append(rel_split)
            frame_ids.append(i)
            frames_written += 1

    return img_paths, depth_paths, Ks, poses, seq_names, splits, frame_ids, h5_path.name, frames_written


def main():
    ap = argparse.ArgumentParser(description="Extract GTA-SfM HDF5 to PNG/EXR and NPZ cameras (multiprocessing)")
    ap.add_argument("--src", required=True, type=Path, help="Root dir containing train/ and/or test/ with .hdf5 files")
    ap.add_argument("--out", required=True, type=Path, help="Output root directory")
    ap.add_argument("--splits", nargs="*", default=["train", "test"], help="Which splits to process (default: train test)")
    ap.add_argument("--workers", type=int, default=os.cpu_count(), help="Number of worker processes (default: cpu_count)")
    args = ap.parse_args()

    src: Path = args.src
    out_root: Path = args.out
    ensure_dir(out_root)

    all_img_paths: List[str] = []
    all_depth_paths: List[str] = []
    all_Ks: List[np.ndarray] = []
    all_poses: List[np.ndarray] = []
    all_seq_names: List[str] = []
    all_splits: List[str] = []
    all_frame_ids: List[int] = []

    total_h5 = 0
    submitted = 0

    with ProcessPoolExecutor(max_workers=args.workers) as ex:
        futures = []
        for split in args.splits:
            split_dir = src / split
            if not split_dir.is_dir():
                print(f"[WARN] Skip missing split dir: {split_dir}")
                continue
            out_split_dir = out_root / split
            ensure_dir(out_split_dir)

            h5_files = sorted(split_dir.glob("*.hdf5"))
            total_h5 += len(h5_files)
            for h5_path in h5_files:
                futures.append(
                    ex.submit(_process_hdf5_file, h5_path, out_split_dir, split)
                )
                submitted += 1

        with tqdm(total=submitted, desc="HDF5 files", unit="file") as pbar:
            for fut in as_completed(futures):
                try:
                    (img_paths, depth_paths, Ks, poses, seq_names, splits, frame_ids, fname, nframes) = fut.result()
                    all_img_paths.extend(img_paths)
                    all_depth_paths.extend(depth_paths)
                    all_Ks.extend(Ks)
                    all_poses.extend(poses)
                    all_seq_names.extend(seq_names)
                    all_splits.extend(splits)
                    all_frame_ids.extend(frame_ids)
                    pbar.set_postfix_str(f"{fname}: {nframes} frames")
                except Exception as e:
                    pbar.set_postfix_str(f"ERROR: {e}")
                finally:
                    pbar.update(1)

    if len(all_img_paths) == 0:
        print("[WARN] No frames extracted. Nothing to save.")
        return

    # Organize data as: split -> seq_name -> {basenames, K, pose}
    data_dict = {}
    for img_path, depth_path, K, pose, seq_name, split, frame_id in zip(
        all_img_paths, all_depth_paths, all_Ks, all_poses, all_seq_names, all_splits, all_frame_ids
    ):
        # Extract basename (without extension)
        basename = Path(img_path).stem
        if split not in data_dict:
            data_dict[split] = {}
        if seq_name not in data_dict[split]:
            data_dict[split][seq_name] = {
                "basenames": [],
                "K": [],
                "pose": [],
            }
        data_dict[split][seq_name]["basenames"].append(basename)
        data_dict[split][seq_name]["K"].append(K)
        data_dict[split][seq_name]["pose"].append(pose)

    # Convert lists to arrays for K and pose
    for split in data_dict:
        for seq_name in data_dict[split]:
            data_dict[split][seq_name]["K"] = np.stack(data_dict[split][seq_name]["K"], axis=0)
            data_dict[split][seq_name]["pose"] = np.stack(data_dict[split][seq_name]["pose"], axis=0)
            data_dict[split][seq_name]["basenames"] = np.array(data_dict[split][seq_name]["basenames"])

    npz_path = out_root / "cameras.npy"
    # Save as a single dict (np.savez_compressed flattens dicts, so use np.save)
    np.save(npz_path, data_dict, allow_pickle=True)

    print("[DONE]")
    print(f"  HDF5 files processed: {total_h5}")
    print(f"  Frames extracted:     {len(all_img_paths)}")
    print(f"  Output root:          {out_root}")
    print(f"  Camera NPZ:           {npz_path}")


if __name__ == "__main__":
    main()