#!/usr/bin/env bash
# Runs ONE (dataset, method) combination via run_worker.py.
#
# run.py loops over the config's datasets x methods sequentially, which serialises
# an ablation's arms behind each other. This entry point lets each arm be its own
# job so they fan out across GPUs instead.
#
# An optional 4th arg pins a single scene, so a long-sequence arm can be split one
# job per scene instead of one job grinding through all ten sequentially.
#
# Usage: bash run_arm.sh <config> <dataset-key> <method-key> [scene]
set -euo pipefail

CONFIG="$1"; DATASET="$2"; METHOD="$3"; SCENE="${4:-}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR/benchmark"

micromamba run -n lingbot_map python run_worker.py \
    --config "$CONFIG" --dataset "$DATASET" --method "$METHOD" \
    ${SCENE:+--scene "$SCENE"}
