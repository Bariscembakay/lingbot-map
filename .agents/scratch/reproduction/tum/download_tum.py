"""Download the full TUM RGB-D benchmark (all non-calibration sequences --
no canonical subset is documented anywhere for this repo, and the paper
doesn't evaluate on TUM at all) for benchmark/datasets/tum.py.

Source: https://cvg.cit.tum.de/data/datasets/rgbd-dataset/download. Each
.tgz extracts directly to the layout the adapter auto-discovers, no reorg
needed. Run on sof1, then register as a cluster dataset for msp3 to pull.
"""

import subprocess
from pathlib import Path

BASE_URL = "https://cvg.cit.tum.de/rgbd/dataset"
OUTPUT_DIR = Path("/group/compact-3dmem/datasets/TUM-RGBD")

# freiburg_version, sequence_name -- excludes *_calibration / calibration_*
# / checkerboard_large entries (not trajectory sequences).
SEQUENCES = [
    ("freiburg1", "xyz"), ("freiburg1", "rpy"),
    ("freiburg2", "xyz"), ("freiburg2", "rpy"),
    ("freiburg1", "360"), ("freiburg1", "floor"), ("freiburg1", "desk"),
    ("freiburg1", "desk2"), ("freiburg1", "room"),
    ("freiburg2", "360_hemisphere"), ("freiburg2", "360_kidnap"),
    ("freiburg2", "desk"), ("freiburg2", "large_no_loop"),
    ("freiburg2", "large_with_loop"),
    ("freiburg3", "long_office_household"),
    ("freiburg2", "pioneer_360"), ("freiburg2", "pioneer_slam"),
    ("freiburg2", "pioneer_slam2"), ("freiburg2", "pioneer_slam3"),
    ("freiburg3", "nostructure_notexture_far"),
    ("freiburg3", "nostructure_notexture_near_withloop"),
    ("freiburg3", "nostructure_texture_far"),
    ("freiburg3", "nostructure_texture_near_withloop"),
    ("freiburg3", "structure_notexture_far"),
    ("freiburg3", "structure_notexture_near"),
    ("freiburg3", "structure_texture_far"),
    ("freiburg3", "structure_texture_near"),
    ("freiburg2", "desk_with_person"),
    ("freiburg3", "sitting_static"), ("freiburg3", "sitting_xyz"),
    ("freiburg3", "sitting_halfsphere"), ("freiburg3", "sitting_rpy"),
    ("freiburg3", "walking_static"), ("freiburg3", "walking_xyz"),
    ("freiburg3", "walking_halfsphere"), ("freiburg3", "walking_rpy"),
    ("freiburg1", "plant"), ("freiburg1", "teddy"),
    ("freiburg2", "coke"), ("freiburg2", "dishes"),
    ("freiburg2", "flowerbouquet"),
    ("freiburg2", "flowerbouquet_brownbackground"),
    ("freiburg2", "metallic_sphere"), ("freiburg2", "metallic_sphere2"),
    ("freiburg3", "cabinet"), ("freiburg3", "large_cabinet"),
    ("freiburg3", "teddy"),
    ("freiburg1", "xyz_validation"), ("freiburg1", "rpy_validation"),
    ("freiburg1", "desk_validation"), ("freiburg1", "desk2_validation"),
    ("freiburg1", "360_validation"), ("freiburg1", "room_validation"),
    ("freiburg1", "plant_validation"),
    ("freiburg2", "xyz_validation"), ("freiburg2", "rpy_validation"),
    ("freiburg2", "360_hemisphere_validation"),
    ("freiburg2", "360_kidnap_validation"), ("freiburg2", "desk_validation"),
    ("freiburg2", "desk_with_person_validation"),
    ("freiburg2", "pioneer_360_validation"),
    ("freiburg3", "cabinet_validation"),
    ("freiburg3", "large_cabinet_validation"),
    ("freiburg3", "long_office_household_validation"),
    ("freiburg3", "nostructure_notexture_far_validation"),
    ("freiburg3", "nostructure_notexture_near_withloop_validation"),
    ("freiburg3", "nostructure_texture_far_validation"),
    ("freiburg3", "nostructure_texture_near_withloop_validation"),
    ("freiburg3", "structure_notexture_far_validation"),
    ("freiburg3", "structure_notexture_near_validation"),
    ("freiburg3", "structure_texture_far_validation"),
    ("freiburg3", "structure_texture_near_validation"),
    ("freiburg3", "sitting_static_validation"),
    ("freiburg3", "sitting_xyz_validation"),
    ("freiburg3", "sitting_halfsphere_validation"),
    ("freiburg3", "sitting_rpy_validation"),
    ("freiburg3", "walking_static_validation"),
    ("freiburg3", "walking_xyz_validation"),
    ("freiburg3", "walking_halfsphere_validation"),
    ("freiburg3", "walking_rpy_validation"),
]


def download_and_extract(freiburg: str, name: str) -> None:
    seq_dir_name = f"rgbd_dataset_{freiburg}_{name}"
    dst = OUTPUT_DIR / seq_dir_name
    if dst.is_dir():
        print(f"[{seq_dir_name}] already present, skipping.")
        return

    url = f"{BASE_URL}/{freiburg}/{seq_dir_name}.tgz"
    tgz_path = OUTPUT_DIR / f"{seq_dir_name}.tgz"

    print(f"[{seq_dir_name}] downloading ...")
    subprocess.run(["wget", "-q", "-c", url, "-O", str(tgz_path)], check=True)

    print(f"[{seq_dir_name}] extracting ...")
    subprocess.run(["tar", "-xzf", str(tgz_path), "-C", str(OUTPUT_DIR)], check=True)

    tgz_path.unlink()
    print(f"[{seq_dir_name}] done.")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    failed = []
    for freiburg, name in SEQUENCES:
        try:
            download_and_extract(freiburg, name)
        except Exception as e:
            print(f"[rgbd_dataset_{freiburg}_{name}] FAILED: {e}")
            failed.append(f"{freiburg}/{name}")

    print(f"\nDone. {len(SEQUENCES) - len(failed)}/{len(SEQUENCES)} succeeded.")
    if failed:
        print("Failed:", failed)
