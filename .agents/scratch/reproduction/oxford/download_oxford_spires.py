"""Download the Oxford Spires raw data needed by preprocess/oxford.py, for
the 10 scenes in its PROCESS_SCENE list. Only fetches images/trajectory/
GT-map/calibration -- not the rosbags/COLMAP/lidar-clouds nothing here
reads. Run on sof1 (fast public internet), then register as a cluster
dataset for msp3 to pull instead of re-running this there.
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
