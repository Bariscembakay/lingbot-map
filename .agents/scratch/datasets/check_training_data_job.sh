#!/usr/bin/env bash
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=03:00:00
# Raw-sequence readiness check (check_training_data.py). TartanAir sets live in zone-msp3, the
# rest in zone-sof1, so it runs once per zone.
# Usage (msp3): sbatch --partition=login --constraint=zone-msp3 --output=/scratch/baris_bakay/%j_%x.log check_training_data_job.sh tartanair-v1,tartanair-v2,tartanground
#       (sof1): sbatch --partition=rendering --qos=rendering --gpus=a6000:1 --constraint=zone-sof1 --output=... check_training_data_job.sh scannetpp,kitti360,matrixcity
set -euo pipefail
SETS=$1
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
for d in ${SETS//,/ }; do case $d in scannetpp) dataset pull ScanNetpp >/dev/null;; matrixcity) dataset pull matrix_city >/dev/null;; *) dataset pull $d >/dev/null;; esac; done
if [ -x "${MAMBA_ROOT_PREFIX:-/nonexistent}/envs/lingbot_map/bin/python" ]; then P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
else V=/scratch/$USER/hfvenv; [ -x $V/bin/python ] || python3 -m venv $V; $V/bin/pip install -q numpy; P=$V/bin/python; fi
OUT=${OUT:-/scratch/$USER/check_training_data_${SETS//,/_}.json}
"$P" scripts/memory/pose_refine/check_training_data.py --sets "$SETS" --out "$OUT" --workers 16
echo "[check] wrote $OUT"
