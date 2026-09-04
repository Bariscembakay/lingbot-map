#!/usr/bin/env python3
"""
    Read all CO3D annotation files and save into a single metadata file.
"""

import os
import os.path as osp
import gzip
import json
import logging
import numpy as np
from tqdm import tqdm

SEEN_CATEGORIES = [
    "apple", "backpack", "banana", "baseballbat", "baseballglove", "bench",
    "bicycle", "bottle", "bowl", "broccoli", "cake", "car", "carrot", "cellphone",
    "chair", "cup", "donut", "hairdryer", "handbag", "hydrant", "keyboard",
    "laptop", "microwave", "motorcycle", "mouse", "orange", "parkingmeter",
    "pizza", "plant", "stopsign", "teddybear", "toaster", "toilet", "toybus",
    "toyplane", "toytrain", "toytruck", "tv", "umbrella", "vase", "wineglass",
]

def merge_co3d_annotations(annotation_dir: str, output_path: str):
    """
    Read all CO3D annotation files and save into a single file.
    Uses original basename as key.
    """
    merged_data = {}
    
    # Find all .jgz files
    jgz_files = []
    for file in os.listdir(annotation_dir):
        if file.endswith('.jgz'):
            jgz_files.append(file)
    
    logging.info(f"Found {len(jgz_files)} annotation files")
    
    for file in tqdm(jgz_files, desc="Processing files"):
        file_path = osp.join(annotation_dir, file)
        basename = file[:-4]  # remove .jgz extension
        
        try:
            with gzip.open(file_path, 'r') as fin:
                data = json.loads(fin.read())
                merged_data[basename] = data
                logging.info(f"Loaded {file}: {len(data)} sequences")
        except Exception as e:
            logging.error(f"Failed to load {file}: {e}")
    
    # Save merged data as npz (compressed)
    os.makedirs(osp.dirname(output_path), exist_ok=True)
    np.savez_compressed(output_path, **merged_data)
    
    logging.info(f"Saved merged annotations to {output_path}")
    logging.info(f"Total files merged: {len(merged_data)}")

if __name__ == "__main__":
    annotation_dir = "/home/haian/codebase/zipmap_code/raw_data/co3d_anno"
    output_path = "/home/haian/codebase/zipmap_code/raw_data/co3d_anno/co3d_merged_annotations.npz"
    
    logging.basicConfig(level=logging.INFO)
    merge_co3d_annotations(annotation_dir, output_path)