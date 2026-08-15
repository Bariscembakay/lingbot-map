"""One-off script: download the Oxford Spires raw data needed to reproduce
paper Table 2 (sparse setting, benchmark/configs/oxford.yaml).

Fetches only what benchmark/../preprocess/oxford.py actually reads:
  - sequences/<seq>/raw/images.zip
  - sequences/<seq>/processed/trajectory/*
  - ground_truth_map/<site>/*         (TLS point clouds)
  - calibration/*
for the 10 scenes in preprocess/oxford.py's PROCESS_SCENE list.

Does NOT fetch rosbags/ros2bag, colmap, lidar-clouds.zip, vilens-slam,
lidar-depths, lidar-undistortion — nothing in this repo reads them.

Zone note: LOCAL_DIR below is sof1's /group (the system-of-record zone).
Once downloaded, the data is registered as the `oxford_spires` cluster
dataset (`dataset create oxford_spires /group/compact-3dmem/datasets/oxford_spires`),
so msp3 should get it via `dataset pull oxford_spires` — NOT by re-running
this script there. msp3's public-internet inbound is ~1 MB/s, so a fresh
HuggingFace download of this size there would be extremely slow; this
script only belongs on sof1 (or wherever a dataset isn't registered yet).
"""

from huggingface_hub import snapshot_download

REPO_ID = "ori-drs/oxford_spires_dataset"
LOCAL_DIR = "/group/compact-3dmem/datasets/oxford_spires"

SEQUENCES = [
    "2024-03-12-keble-college-02",
    "2024-03-12-keble-college-03",
    "2024-03-12-keble-college-04",
    "2024-03-12-keble-college-05",
    "2024-03-13-observatory-quarter-01",
    "2024-03-13-observatory-quarter-02",
    "2024-03-18-christ-church-02",
    "2024-03-18-christ-church-03",
    "2024-03-20-christ-church-05",  # NOT 2024-03-18 -- upstream PROCESS_SCENE had this wrong
    "2024-05-20-bodleian-library-02",
]

GT_MAP_SITES = [
    "keble-college",
    "observatory-quarter",
    "christ-church",
    "bodleian-library",
]

patterns = ["calibration/*"]
for seq in SEQUENCES:
    patterns.append(f"sequences/{seq}/raw/images.zip")
    patterns.append(f"sequences/{seq}/processed/trajectory/*")
for site in GT_MAP_SITES:
    patterns.append(f"ground_truth_map/{site}/*")

if __name__ == "__main__":
    print(f"Downloading {len(patterns)} patterns to {LOCAL_DIR} ...")
    snapshot_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        local_dir=LOCAL_DIR,
        allow_patterns=patterns,
    )
    print("Done.")
