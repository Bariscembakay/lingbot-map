#!/usr/bin/env bash
# Generic runner for the benchmark pipeline (prepare/run/evaluate) on msp3,
# following the msp3 protocol: dataset pull -> compute on /scratch -> ship
# only metrics/logs back to sof1. Not tied to any one dataset.
#
# Prerequisites (per dataset, done once, not by this script):
#   - the dataset registered on sof1 (`dataset create <name> ...`)
#   - a msp3-local copy of the benchmark config with `raw_data_root` set to
#     /data/<name> and `workspace` set to a /scratch path -- these
#     genuinely differ per zone, so they're a local (uncommitted) edit on
#     msp3's checkout, not something this script rewrites for you.
#
# Usage (as the payload of an srun/sbatch job on msp3):
#   srun --partition=batch --constraint=zone-msp3 --gpus=h200:1 \
#        --cpus-per-task=16 --mem=96G --time=01:00:00 \
#        bash .agents/scratch/insait_cluster_files/submit_msp3.sh \
#        <dataset_name> configs/oxford.yaml \
#        /group/compact-3dmem/campaigns/lingbot_map/oxford_spires/sparse_s12
set -euo pipefail

DATASET_NAME="$1"
CONFIG="$2"
SOF1_CAMPAIGN="$3"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

echo "=== dataset pull $DATASET_NAME ==="
dataset pull "$DATASET_NAME"

source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
source .agents/scratch/insait_cluster_files/setup_bench_env.sh

cd benchmark
for stage in prepare run evaluate; do
    echo "=== $stage.py ($CONFIG) ==="
    micromamba run -n bench python "$stage.py" --config "$CONFIG"
done

WORKSPACE="$(micromamba run -n bench python -c "import yaml; print(yaml.safe_load(open('$CONFIG'))['workspace'])")"

echo "=== shipping metrics back to sof1 (eval/ + logs/ only) ==="
ssh sof1 "mkdir -p $SOF1_CAMPAIGN"
rsync -a \
    --include='*/' \
    --include='eval/***' \
    --include='logs/***' \
    --exclude='*' \
    "$WORKSPACE"/ "sof1:$SOF1_CAMPAIGN/"

echo "=== DONE ==="
