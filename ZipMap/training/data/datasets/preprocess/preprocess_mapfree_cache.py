import os
import os.path as osp
import numpy as np
import pickle
import h5py
from tqdm import tqdm
import argparse

def path2imgid(subscene, filename):
    """Convert subscene and filename to image id"""
    first_seq_id = int(subscene[5:])  # Remove 'dense' prefix
    first_frame_id = int(filename[6:-4])  # Remove 'frame_' prefix and '.jpg' suffix
    return [first_seq_id, first_frame_id]

def generate_mapfree_cache(root_dir, num_views=50):
    """Generate cached metadata for MapFree dataset
    
    Args:
        root_dir: Root directory of MapFree dataset
        num_views: Minimum number of views required for image groups
    """
    cache_file = f"{root_dir}/cached_metadata_{num_views}_col_only.h5"
    
    if os.path.exists(cache_file):
        print(f"Cache file {cache_file} already exists. Remove it first if you want to regenerate.")
        return cache_file
    
    print(f"Generating MapFree cache at {cache_file}")
    print(f"Root directory: {root_dir}")
    
    # Get all scene directories
    scene_dirs = sorted([
        d for d in os.listdir(root_dir) 
        if os.path.isdir(os.path.join(root_dir, d)) and not d.startswith('.')
    ])
    
    if not scene_dirs:
        raise ValueError(f"No scene directories found in {root_dir}")
    
    print(f"Found {len(scene_dirs)} scenes: {scene_dirs}")
    
    scenes = []
    sceneids = []
    groups = []
    scope = []
    images = []
    id_ranges = []
    is_video = []
    start = 0
    j = 0
    offset = 0

    for scene in tqdm(scene_dirs, desc="Processing scenes"):
        scene_path = os.path.join(root_dir, scene)
        scenes.append(scene)
        
        # Get video sequences (dense* directories)
        subscenes = sorted([
            d for d in os.listdir(scene_path) 
            if d.startswith("dense") and os.path.isdir(os.path.join(scene_path, d))
        ])
        
        if not subscenes:
            print(f"Warning: No dense* subdirectories found in {scene}")
            continue
            
        print(f"Scene {scene}: found {len(subscenes)} subscenes")
        
        id_range_subscenes = []
        for subscene in subscenes:
            rgb_dir = os.path.join(scene_path, subscene, "rgb")
            if not os.path.exists(rgb_dir):
                print(f"Warning: RGB directory not found: {rgb_dir}")
                continue
                
            rgb_paths = sorted([
                d for d in os.listdir(rgb_dir) 
                if d.endswith(".jpg")
            ])
            
            if len(rgb_paths) == 0:
                print(f"Warning: No RGB images found in {rgb_dir}")
                continue
                
            num_imgs = len(rgb_paths)
            # print(f"  Subscene {subscene}: {num_imgs} images")
            
            # Add image IDs
            images.extend([path2imgid(subscene, rgb_path) for rgb_path in rgb_paths])
            id_range_subscenes.append((offset, offset + num_imgs))
            offset += num_imgs

        # Process image collections using metadata.pkl
        metadata_path = os.path.join(scene_path, "metadata.pkl")
        if not os.path.exists(metadata_path):
            print(f"Warning: metadata.pkl not found in {scene_path}")
            continue
            
        print(f"  Loading metadata from {metadata_path}")
        try:
            with open(metadata_path, "rb") as f:
                metadata = pickle.load(f)
        except Exception as e:
            print(f"Error loading metadata for {scene}: {e}")
            continue
            
        ref_imgs = list(metadata.keys())
        img_groups = []
        
        for ref_img in ref_imgs:
            other_imgs = metadata[ref_img]
            if len(other_imgs) + 1 < num_views:
                continue
                
            # Create group: [seq_id, frame_id, overlap_score]
            group = [(*other_img[0], other_img[1]) for other_img in other_imgs]
            group.insert(0, (*ref_img, 1))  # Reference image has overlap score of 1
            img_groups.append(np.array(group))
            
            # Store the ID range for this reference image's subscene
            if ref_img[0] < len(id_range_subscenes):
                id_ranges.append(id_range_subscenes[ref_img[0]])
            else:
                print(f"Warning: ref_img sequence {ref_img[0]} out of range for scene {scene}")
                continue
                
            scope.append(start)
            start = start + len(group)

        num_groups = len(img_groups)
        print(f"  Created {num_groups} image groups")
        
        sceneids.extend([j] * num_groups)
        groups.extend(img_groups)
        is_video.extend([False] * num_groups)  # All are image collections
        j += 1

    # Convert to numpy arrays
    scenes_array = np.array(scenes, dtype=object)
    sceneids_array = np.array(sceneids)
    scope_array = np.array(scope)
    video_flags_array = np.array(is_video)
    groups_array = np.concatenate(groups, 0) if groups else np.array([])
    id_ranges_array = np.array(id_ranges)
    images_array = np.array(images)

    print(f"\nDataset statistics:")
    print(f"  Scenes: {len(scenes_array)}")
    print(f"  Scene IDs: {len(sceneids_array)}")
    print(f"  Image groups: {len(scope_array)}")
    print(f"  Total images: {len(images_array)}")
    print(f"  Groups array shape: {groups_array.shape}")
    print(f"  ID ranges: {len(id_ranges_array)}")

    # Save to HDF5 file
    print(f"\nSaving cache to {cache_file}")
    os.makedirs(os.path.dirname(cache_file), exist_ok=True)
    
    with h5py.File(cache_file, "w") as h5f:
        h5f.create_dataset(
            "scenes",
            data=scenes_array,
            dtype=h5py.string_dtype(encoding="utf-8"),
            compression="lzf",
            chunks=True,
        )
        h5f.create_dataset(
            "sceneids", data=sceneids_array, compression="lzf", chunks=True
        )
        h5f.create_dataset(
            "scope", data=scope_array, compression="lzf", chunks=True
        )
        h5f.create_dataset(
            "video_flags", data=video_flags_array, compression="lzf", chunks=True
        )
        h5f.create_dataset(
            "groups", data=groups_array, compression="lzf", chunks=True
        )
        h5f.create_dataset(
            "id_ranges", data=id_ranges_array, compression="lzf", chunks=True
        )
        h5f.create_dataset(
            "images", data=images_array, compression="lzf", chunks=True
        )
    
    print(f"Cache file generated successfully: {cache_file}")
    return cache_file

def main():
    parser = argparse.ArgumentParser(description="Generate MapFree dataset cache")
    parser.add_argument(
        "--root_dir", 
        type=str, 
        default="/home/storage-bucket/haian/zipmap_data/processed_mapfree/",
        help="Root directory of MapFree dataset"
    )
    parser.add_argument(
        "--num_views", 
        type=int, 
        default=48,
        help="Minimum number of views required for image groups (default: 50)"
    )
    
    args = parser.parse_args()
    
    if not os.path.exists(args.root_dir):
        raise ValueError(f"Root directory does not exist: {args.root_dir}")
    
    generate_mapfree_cache(args.root_dir, args.num_views)

if __name__ == "__main__":
    main()