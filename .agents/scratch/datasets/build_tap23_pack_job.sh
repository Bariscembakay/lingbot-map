#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
# Packs the tap-23 specials + poses of every lingbot-cache1048 dataset for the bidirectional refiner.
set -euo pipefail
cd "$HOME/lingbot-map"
O=/group/compact-3dmem/datasets/_staging/lingbot-tap23pack-1048
for d in ${DATASETS:-matrixcity tartanair-v1 tartanair-v2 tartanground scannetpp scannet-v2}; do
    dataset pull lingbot-cache1048-$d >/dev/null
    python3 scripts/memory/pose_refine/build_tap23_pack.py --dataset $d --out $O
done
