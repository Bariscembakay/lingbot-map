#!/usr/bin/env bash
set -euo pipefail
cd /home/baris_bakay/lingbot-map
export PYTHONPATH=/home/baris_bakay/lingbot-map
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python" \
    .agents/scratch/memory_eval/lora_gpu_test.py
