#!/usr/bin/env bash
#SBATCH --partition=login
#SBATCH --constraint=zone-msp3
#SBATCH --nodelist=msp3-login-3
#SBATCH --cpus-per-task=32
#SBATCH --mem=96G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/baris_bakay/%j_%x.log
# ScanNet v2 .sens -> frames (extract_scannet_sens.py) on msp3 scratch, registered as
# /data/scannet-v2-extracted; staging deleted once the registered file count matches.
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
V=/scratch/$USER/hfvenv
[ -x $V/bin/python ] || python3 -m venv $V
$V/bin/pip install -q numpy pillow
dataset pull scannet-v2 >/dev/null
S=/scratch/$USER/stage_scannet-v2-extracted
mkdir -p $S
$V/bin/python scripts/memory/pose_refine/extract_scannet_sens.py --root /data/scannet-v2/scans --out $S --workers 32
n_src=$(find $S -type f | wc -l)
echo "[sens] $(date) staged $n_src files, $(du -sh $S | cut -f1)"
dataset create scannet-v2-extracted $S
dataset pin scannet-v2-extracted || true
n_dst=$(find /data/scannet-v2-extracted -type f | wc -l)
echo "[sens] $(date) registered /data/scannet-v2-extracted: $n_dst files (staging $n_src)"
[ "$n_src" -eq "$n_dst" ] || { echo "[sens] count mismatch, staging kept"; exit 1; }
rm -rf $S
echo "[sens] done"
