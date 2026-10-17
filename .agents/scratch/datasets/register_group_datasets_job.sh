#!/usr/bin/env bash
#SBATCH --partition=login
#SBATCH --constraint=zone-sof1
#SBATCH --nodelist=sof1-h200-0
#SBATCH --cpus-per-task=8
#SBATCH --mem=32G
#SBATCH --time=12:00:00
#SBATCH --output=/group/compact-3dmem/datasets/_download_logs/%j_%x.log
# Move /group/compact-3dmem/datasets/<dir> into the dataset registry as /data/<name>, then delete the
# /group copy once /data/<name> holds the same number of files.
# lingbot-poserefine-v1 is staged through /scratch with symlinks dereferenced (kitti360_train links).
set -uo pipefail
G=/group/compact-3dmem/datasets
move() {  # <group dir> <dataset name> [--replace] [--deref]
    local src=$G/$1 name=$2 rep="" from=$G/$1; shift 2
    for a in "$@"; do case $a in --replace) rep=--replace;; --deref)
        from=/scratch/$USER/stage_$name; rm -rf "$from"; rsync -aL "$src/" "$from/" ;; esac; done
    echo "[reg] $(date) $from -> /data/$name $rep"
    dataset create "$name" "$from" $rep || { echo "[reg] create failed for $name"; return 1; }
    dataset pin "$name" || true
    local ns nd; ns=$(find -L "$src" -type f | wc -l); nd=$(find "/data/$name" -type f | wc -l)
    echo "[reg] $(date) $name: /data $nd files, group $ns files"
    [ "$ns" -eq "$nd" ] || { echo "[reg] count mismatch, keeping $src"; return 1; }
    rm -rf "$src"; [ "$from" = "$src" ] || rm -rf "$from"
    echo "[reg] deleted $src"
}
move kitti360 kitti360
move vkitti2 vkitti2
move lingbot-poserefine-v1 lingbot-poserefine-v1 --replace --deref
move chorus_scannetpp_v2_vox5cm chorus-scannetpp-v2-vox5cm --replace
echo "[reg] done"
