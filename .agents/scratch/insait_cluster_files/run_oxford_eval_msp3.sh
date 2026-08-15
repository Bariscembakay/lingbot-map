#!/usr/bin/env bash
# Runs the full Oxford Spires (sparse, Table 2) benchmark pipeline on msp3:
# dataset pull -> prepare.py -> run.py -> evaluate.py -> ship metrics back
# to sof1's real campaign path. Per the msp3 protocol: compute happens
# here, only small metrics/logs cross zones (the fast msp3->sof1
# direction), the workspace with large BSS frame data stays on /scratch.
set -euo pipefail
cd /home/baris_bakay/lingbot-map

echo "=== dataset pull oxford_spires_processed ==="
dataset pull oxford_spires_processed

echo "=== env bootstrap ==="
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
source .agents/scratch/insait_cluster_files/setup_bench_env.sh

cd benchmark

echo "=== prepare.py ==="
micromamba run -n bench python prepare.py --config configs/oxford.yaml

echo "=== run.py ==="
micromamba run -n bench python run.py --config configs/oxford.yaml

echo "=== evaluate.py ==="
micromamba run -n bench python evaluate.py --config configs/oxford.yaml

echo "=== shipping metrics back to sof1 (eval/ + logs/ only, not raw BSS frame data) ==="
SOF1_CAMPAIGN=/group/compact-3dmem/campaigns/lingbot_map/oxford_spires/sparse_s12
ssh sof1 "mkdir -p $SOF1_CAMPAIGN"
rsync -a \
    --include='*/' \
    --include='eval/***' \
    --include='logs/***' \
    --exclude='*' \
    /scratch/baris_bakay/oxford_eval_workspace/ sof1:$SOF1_CAMPAIGN/

echo "=== DONE ==="
