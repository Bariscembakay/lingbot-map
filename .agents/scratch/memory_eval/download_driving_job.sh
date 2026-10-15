#!/usr/bin/env bash
#SBATCH --job-name=DOWNLOAD_kitti360_image00_poses_calib_9drives_and_vkitti2_rgb_textgt_cpu
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=8
#SBATCH --mem=16G
#SBATCH --time=12:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_download.log
# KITTI-360 (perspective camera 00 of the 9 drives with GT poses) and Virtual KITTI 2.
# lingbot trained on both, so they are fair training data for the driving pose refiner.
set -uo pipefail
D=/group/compact-3dmem/datasets
S3=https://s3.eu-central-1.amazonaws.com/avg-projects/KITTI-360
fetch() {  # url dest_dir: resumable download, unzip/untar, drop the archive
    local f="$2/$(basename "$1")"
    mkdir -p "$2"
    wget -q -c -O "$f" "$1" || { echo "[fail] $1"; return 1; }
    case "$f" in
        *.zip) unzip -q -o "$f" -d "$2" && rm "$f" ;;
        *.tar.gz) tar -xzf "$f" -C "$2" && rm "$f" ;;
        *.tar) tar -xf "$f" -C "$2" && rm "$f" ;;
    esac
    echo "[ok] $1"
}
fetch $S3/89a6bae3c8a6f789e12de4807fc1e8fdcf182cf4/data_poses.zip $D/kitti360
fetch $S3/384509ed5413ccc81328cf8c55cc6af078b8c444/calibration.zip $D/kitti360
fetch https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_textgt.tar.gz $D/vkitti2
fetch https://download.europe.naverlabs.com/virtual_kitti_2.0.3/vkitti_2.0.3_rgb.tar $D/vkitti2
for s in 0000 0002 0003 0004 0005 0006 0007 0009 0010; do
    fetch $S3/data_2d_raw/2013_05_28_drive_${s}_sync_image_00.zip $D/kitti360
done
du -sh $D/kitti360 $D/vkitti2
