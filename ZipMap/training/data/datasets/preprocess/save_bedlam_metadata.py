import os
import os.path as osp
import logging
import numpy as np
from tqdm import tqdm
import argparse
from multiprocessing import Pool, cpu_count
from functools import partial

logging.basicConfig(level=logging.DEBUG)

# Minimum number of images required per scene
MIN_NUM_IMAGES = 10

# Invalid sequences to skip (from original dataset)
INVALID_SEQS = [
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000042",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000059",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000079",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000978",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000081",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000268",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000089",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000189",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000034",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000889",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000293",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000067",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000904",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000434",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000044",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000396",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000012",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000082",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000120",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000324",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000038",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000486",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000421",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000226",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000012",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000149",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000311",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000080",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000122",
    "20221012_3-10_500_batch01hand_zoom_highSchoolGym_seq_000079",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000077",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000095",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000062",
    "20221013_3_250_batch01hand_static_bigOffice_seq_000015",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000095",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000119",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000297",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000011",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000196",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000316",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000283",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000085",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000287",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000163",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000804",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000842",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000027",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000182",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000982",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000029",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000031",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000025",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000250",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000785",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000069",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000122",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000246",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000352",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000425",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000192",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000900",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000043",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000063",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000096",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000091",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000013",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000309",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000114",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000969",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000361",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000267",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000083",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000383",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000890",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000003",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000045",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000317",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000076",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000082",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000907",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000279",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000076",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000004",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000061",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000811",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000800",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000841",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000794",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000308",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000064",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000284",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000752",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000269",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000036",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000419",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000290",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000322",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000818",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000327",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000326",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000002",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000060",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000348",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000059",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000016",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000817",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000332",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000094",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000193",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000779",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000177",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000368",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000023",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000024",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000310",
    "20221014_3_250_batch01hand_orbit_archVizUI3_time15_seq_000086",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000038",
    "20221024_10_100_batch01handhair_zoom_suburb_d_seq_000071",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000768",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000017",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000053",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000097",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000856",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000827",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000161",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000084",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000106",
    "20221013_3_250_batch01hand_orbit_bigOffice_seq_000207",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000007",
    "20221024_3-10_100_batch01handhair_static_highSchoolGym_seq_000013",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000251",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000796",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000105",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000251",
    "20221019_3-8_250_highbmihand_orbit_stadium_seq_000046",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000334",
    "20221019_3-8_1000_highbmihand_static_suburb_d_seq_000453",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000373",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000283",
    "20221010_3-10_500_batch01hand_zoom_suburb_d_seq_000249",
]

# HDRI scene prefixes to skip (from original dataset)
HDRI_SCENES = [
    "20221010_3_1000_batch01hand",
    "20221017_3_1000_batch01hand",
    "20221018_3-8_250_batch01hand",
    "20221019_3_250_highbmihand",
]

def process_single_scene(scene, BEDLAM_DIR):
    """
    Process a single scene and return its metadata
    Returns: (scene_name, scene_data_dict) or (scene_name, None) if skipped
    """
    # Skip invalid sequences
    if scene in INVALID_SEQS:
        return (scene, None)

    # Skip HDRI scenes
    if any([scene.startswith(x) for x in HDRI_SCENES]):
        return (scene, None)

    # Skip closeup scenes
    if "closeup" in scene:
        return (scene, None)

    scene_dir = osp.join(BEDLAM_DIR, scene)
    rgb_dir = osp.join(scene_dir, "rgb")
    cam_dir = osp.join(scene_dir, "cam")

    if not osp.exists(rgb_dir) or not osp.exists(cam_dir):
        return (scene, None)

    # Get all RGB image basenames (without extension), sorted
    try:
        rgb_files = sorted([
            f for f in os.listdir(rgb_dir)
            if f.endswith(".png")
        ])
    except Exception as e:
        logging.warning(f"Failed to list files in {rgb_dir}: {e}")
        return (scene, None)

    basenames = [osp.splitext(f)[0] for f in rgb_files]
    if len(basenames) < MIN_NUM_IMAGES:
        return (scene, None)

    images = []
    c2ws = []
    intrinsics = []

    for basename in basenames:
        cam_path = osp.join(cam_dir, basename + ".npz")
        if not osp.exists(cam_path):
            continue
        try:
            cam = np.load(cam_path)
            c2ws.append(cam["pose"].astype(np.float32))
            intrinsics.append(cam["intrinsics"].astype(np.float32))
            images.append(basename)
        except Exception as e:
            logging.warning(f"Failed to load camera data for {basename} in {scene}: {e}")
            continue

    if len(images) < MIN_NUM_IMAGES:
        return (scene, None)

    scene_data = {
        "images": np.array(images),
        "c2ws": np.array(c2ws),
        "intrinsics": np.array(intrinsics),
    }

    return (scene, scene_data)

def fetch_and_save_bedlam_metadata(BEDLAM_DIR, output_file, num_workers=None):
    """
    Process BEDLAM dataset and save all metadata in a single npz file

    Args:
        BEDLAM_DIR: Root directory of BEDLAM dataset
        output_file: Output path for the npz file
        num_workers: Number of worker processes (default: cpu_count())
    """
    os.makedirs(os.path.dirname(output_file), exist_ok=True)

    # Get all scenes (top-level directories)
    all_scenes = sorted([
        f for f in os.listdir(BEDLAM_DIR)
        if os.path.isdir(osp.join(BEDLAM_DIR, f))
    ])

    logging.info(f"Found {len(all_scenes)} scenes to process")

    # Determine number of workers
    if num_workers is None:
        num_workers = cpu_count()
    logging.info(f"Using {num_workers} worker processes")

    # Create partial function with BEDLAM_DIR bound
    process_func = partial(process_single_scene, BEDLAM_DIR=BEDLAM_DIR)

    # Process scenes in parallel
    scene_dict = {}
    processed_scenes = 0
    skipped_scenes = 0

    with Pool(processes=num_workers) as pool:
        # Use imap for progress bar
        results = list(tqdm(
            pool.imap(process_func, all_scenes),
            total=len(all_scenes),
            desc="Processing scenes"
        ))

    # Collect results
    for scene_name, scene_data in results:
        if scene_data is not None:
            scene_dict[scene_name] = scene_data
            processed_scenes += 1
            logging.info(f"Processed {scene_name}: {len(scene_data['images'])} images")
        else:
            skipped_scenes += 1

    # Save as a single npz with dict-of-arrays
    np.savez_compressed(
        output_file,
        **scene_dict
    )
    logging.info(f"BEDLAM processing complete: {processed_scenes} scenes processed, {skipped_scenes} scenes skipped")
    logging.info(f"All metadata saved to {output_file}")

def main():
    parser = argparse.ArgumentParser(description="Process BEDLAM dataset and save metadata")
    parser.add_argument("--input_dir", type=str, default="/home/storage-bucket/haian/zipmap_data/processed_bedlam", help="Input BEDLAM directory")
    parser.add_argument("--output_file", type=str, help="Output npz file for all processed metadata")
    parser.add_argument("--num_workers", type=int, default=None, help="Number of worker processes (default: cpu_count())")

    args = parser.parse_args()

    if args.output_file is None:
        args.output_file = osp.join(args.input_dir, "all_scenes_dict.npz")

    fetch_and_save_bedlam_metadata(args.input_dir, args.output_file, num_workers=args.num_workers)

if __name__ == "__main__":
    main()
