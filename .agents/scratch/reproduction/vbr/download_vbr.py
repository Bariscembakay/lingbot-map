"""Download + reorganize VBR for lingbot-map's benchmark/datasets/vbr.py.

Source: LoGeR's pre-processed HF release (Junyi42/vbr_processed) -- file
content already matches, just wrong directory layout (HF: {scene}/... ;
lingbot-map wants {scene}_processed_aligned/ + sibling processed_gt/).
Run on sof1 (fast public internet); register as a cluster dataset after so
msp3 uses `dataset pull` instead of re-running this there.
"""

import shutil
import subprocess
from pathlib import Path

from huggingface_hub import hf_hub_download

REPO_ID = "Junyi42/vbr_processed"
OUTPUT_DIR = Path("/group/compact-3dmem/datasets/vbr")

SCENES = [
    "campus_train0",
    "campus_train1",
    "ciampino_train1",
    "colosseo_train0",
    "diag_train0",
    "pincio_train0",
    "spagna_train0",
]


def process_scene(scene: str, tmp_dir: Path) -> None:
    print(f"[{scene}] downloading ...")
    tar_path = hf_hub_download(
        repo_id=REPO_ID,
        repo_type="dataset",
        filename=f"{scene}.tar.gz",
        local_dir=str(tmp_dir),
    )

    print(f"[{scene}] extracting ...")
    subprocess.run(["tar", "-xzf", tar_path, "-C", str(tmp_dir)], check=True)

    extracted = tmp_dir / scene
    aligned_dst = OUTPUT_DIR / f"{scene}_processed_aligned"
    gt_dst_dir = OUTPUT_DIR / "processed_gt"
    gt_dst_dir.mkdir(parents=True, exist_ok=True)

    gt_src = extracted / f"{scene}_gt.txt"
    shutil.move(str(gt_src), str(gt_dst_dir / f"{scene}_gt.txt"))

    # Not read by benchmark/datasets/vbr.py -- drop it, it's redundant with
    # camera_pose.txt and just adds thousands of tiny files.
    per_frame_pose_dir = extracted / "camera_pose"
    if per_frame_pose_dir.is_dir():
        shutil.rmtree(per_frame_pose_dir)

    shutil.move(str(extracted), str(aligned_dst))

    Path(tar_path).unlink()
    print(f"[{scene}] done -> {aligned_dst}")


if __name__ == "__main__":
    OUTPUT_DIR.mkdir(parents=True, exist_ok=True)
    tmp_dir = OUTPUT_DIR / "_tmp_extract"
    tmp_dir.mkdir(exist_ok=True)

    for scene in SCENES:
        process_scene(scene, tmp_dir)

    shutil.rmtree(tmp_dir, ignore_errors=True)
    print("All VBR scenes downloaded and reorganized.")
