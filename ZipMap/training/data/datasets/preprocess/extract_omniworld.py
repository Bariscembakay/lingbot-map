#!/usr/bin/env python3
"""

Download Cmd:

# Download RGB videos
hf download InternRobotics/OmniWorld \                                                                             
    --repo-type dataset \
    --include "videos/*" \
    --local-dir ./OmniWorld-11k/

# Download Depth videos
hf download InternRobotics/OmniWorld \                                                                             
    --repo-type dataset \
    --include "annotations/OmniWorld-Game/*"  \
    --local-dir ./OmniWorld-11k/

  
python /home/haian/codebase/zipmap_code/Zimmap/training/data/datasets/preprocess/extract_omniworld.py --all --num-workers 192

Script to extract OmniWorld dataset files.
Handles extraction of RGB, depth, flow, and other annotations for specified scenes.
"""

import os
import tarfile
import argparse
from pathlib import Path
from typing import List, Optional
import glob
from tqdm import tqdm
from multiprocessing import Pool, cpu_count
from functools import partial


def extract_scene(scene_id: str, root_path: str, output_base: Optional[str] = None,
                  modalities: List[str] = None, dataset: str = 'OmniWorld-Game') -> None:
    """
    Extract all data for a specific scene.

    Args:
        scene_id: Scene identifier (e.g., 'b04f88d1f85a')
        root_path: Root path to OmniWorld dataset
        output_base: Base output directory
        modalities: List of modalities to extract (rgb, depth, flow, others)
        dataset: Dataset name
    """
    if modalities is None:
        modalities = ['rgb', 'depth', 'flow', 'others']

    # Create scene-specific output directory
    if output_base is None:
        output_dir = scene_id
    else:
        output_dir = os.path.join(output_base, dataset, scene_id)

    os.makedirs(output_dir, exist_ok=True)

    # Define paths
    videos_path = Path(root_path) / 'videos' / dataset / scene_id
    annotations_path = Path(root_path) / 'annotations' / dataset / scene_id

    # Extract RGB frames
    if 'rgb' in modalities:
        rgb_files = sorted(glob.glob(str(videos_path / f'{scene_id}_rgb_*.tar.gz')))
        for rgb_tar in tqdm(rgb_files, desc='Extracting RGB'):
            print(f"Extracting {os.path.basename(rgb_tar)}...")
            with tarfile.open(rgb_tar, 'r:gz') as tar:
                tar.extractall(output_dir)

    # Extract Depth
    if 'depth' in modalities:
        depth_files = sorted(glob.glob(str(annotations_path / f'{scene_id}_depth_*.tar.gz')))
        for depth_tar in tqdm(depth_files, desc='Extracting Depth'):
            print(f"Extracting {os.path.basename(depth_tar)}...")
            with tarfile.open(depth_tar, 'r:gz') as tar:
                tar.extractall(output_dir)

    # Extract Flow
    if 'flow' in modalities:
        flow_files = sorted(glob.glob(str(annotations_path / f'{scene_id}_flow_*.tar.gz')))
        for flow_tar in tqdm(flow_files, desc='Extracting Flow'):
            print(f"Extracting {os.path.basename(flow_tar)}...")
            with tarfile.open(flow_tar, 'r:gz') as tar:
                tar.extractall(output_dir)

    # Extract other annotations
    if 'others' in modalities:
        others_tar = annotations_path / f'{scene_id}_others.tar.gz'
        if others_tar.exists():
            print(f"Extracting {others_tar.name}...")
            with tarfile.open(others_tar, 'r:gz') as tar:
                tar.extractall(output_dir)

    print(f"✓ Extraction complete for scene {scene_id}")


def extract_scene_wrapper(scene_id: str, root_path: str, output_base: Optional[str],
                          modalities: List[str], dataset: str) -> str:
    """
    Wrapper function for multiprocessing that returns the scene_id when done.
    """
    try:
        extract_scene(scene_id, root_path, output_base, modalities, dataset)
        return f"✓ {scene_id}"
    except Exception as e:
        return f"✗ {scene_id}: {str(e)}"


def extract_all_scenes(root_path: str, dataset: str = 'OmniWorld-Game',
                       output_base: Optional[str] = None,
                       modalities: List[str] = None,
                       num_workers: int = None) -> None:
    """
    Extract all scenes from a dataset using multiprocessing.

    Args:
        root_path: Root path to OmniWorld dataset
        dataset: Dataset name (e.g., 'OmniWorld-Game')
        output_base: Base output directory
        modalities: List of modalities to extract
        num_workers: Number of parallel workers (defaults to CPU count)
    """
    videos_path = Path(root_path) / 'videos' / dataset

    if not videos_path.exists():
        print(f"Error: Dataset path {videos_path} does not exist")
        return

    # Get all scene directories
    scene_dirs = sorted([d.name for d in videos_path.iterdir() if d.is_dir()])

    print(f"Found {len(scene_dirs)} scenes in {dataset}")

    if num_workers is None:
        num_workers = min(cpu_count(), len(scene_dirs))

    print(f"Using {num_workers} workers for parallel extraction\n")

    # Create partial function with fixed arguments
    extract_func = partial(
        extract_scene_wrapper,
        root_path=root_path,
        output_base=output_base,
        modalities=modalities,
        dataset=dataset
    )

    # Process scenes in parallel with progress bar
    with Pool(processes=num_workers) as pool:
        results = list(tqdm(
            pool.imap(extract_func, scene_dirs),
            total=len(scene_dirs),
            desc="Extracting scenes"
        ))

    # Print summary
    print("\n=== Extraction Summary ===")
    for result in results:
        print(result)


def main():
    parser = argparse.ArgumentParser(description='Extract OmniWorld dataset files')
    # parser.add_argument('--root', type=str, default='/home/haian/Dataset/OmniWorld',
    #                     help='Root path to OmniWorld dataset')
    parser.add_argument('--root', type=str, default='/mnt/data/OmniWorld-11k',
                        help='Root path to OmniWorld dataset')
    parser.add_argument('--scene-id', type=str, default=None,
                        help='Specific scene ID to extract (e.g., b04f88d1f85a)')
    parser.add_argument('--dataset', type=str, default='OmniWorld-Game',
                        help='Dataset name (default: OmniWorld-Game)')
    parser.add_argument('--output', type=str, default='/home/storage-bucket/haian/zipmap_data/OmniWorld-11k',
                        help='Output directory (defaults to scene_id)')
    parser.add_argument('--modalities', type=str, nargs='+',
                        default=['rgb', 'depth','others'],
                        choices=['rgb', 'depth', 'flow', 'others'],
                        help='Modalities to extract')
    parser.add_argument('--all', action='store_true',
                        help='Extract all scenes in the dataset')
    parser.add_argument('--num-workers', type=int, default=None,
                        help='Number of parallel workers (default: CPU count)')

    args = parser.parse_args()

    if args.all:
        extract_all_scenes(args.root, args.dataset, args.output, args.modalities, args.num_workers)
    elif args.scene_id:
        extract_scene(args.scene_id, args.root, args.output, args.modalities, args.dataset)
    else:
        parser.print_help()
        print("\nError: Please specify either --scene-id or --all")


if __name__ == '__main__':
    main()
