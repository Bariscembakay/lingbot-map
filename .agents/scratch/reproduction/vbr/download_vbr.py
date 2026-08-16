"""Download + reorganize the VBR (Vision Benchmark in Rome) dataset for
lingbot-map's benchmark/datasets/vbr.py adapter.

Source: the LoGeR project's pre-processed/aligned release on HuggingFace
(Junyi42/vbr_processed) — see benchmark/README.md's "VBR" data-prep note
and https://github.com/Junyi42/LoGeR (eval/eval.md, "VBR" section, Option A).

The HF release already has the right FILE CONTENT (3x3 K intrinsics.txt,
TUM-format camera_pose.txt with integer frame-index timestamps — verified
by hand against benchmark/datasets/vbr.py's docstring), just the wrong
DIRECTORY LAYOUT:
    HF:          {scene}/{rgb/, camera_pose.txt, intrinsics.txt, camera_pose/, {scene}_gt.txt}
    lingbot-map: {scene}_processed_aligned/{rgb/, camera_pose.txt, intrinsics.txt}
                 processed_gt/{scene}_gt.txt   (sibling dir, not nested)

So this script downloads each scene's tarball, extracts it, and just
renames/moves things into place — no real preprocessing.

Zone note: same as download_oxford_spires.py — this belongs on sof1 (fast
public internet). Once done, register it as a cluster dataset
(`dataset create vbr /group/compact-3dmem/datasets/vbr`) so msp3 gets it
via `dataset pull vbr`, not by re-running this script there.
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
