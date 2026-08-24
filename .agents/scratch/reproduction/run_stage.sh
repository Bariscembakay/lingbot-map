#!/usr/bin/env bash
# Runs one pipeline stage (prepare/evaluate/report) for a config, in the bench env.
# One stage at a time so prepare and evaluate can be separate jobs
# with a Slurm dependency between them and the per-arm run jobs.
#
# Extra flags for the stage script go in STAGE_EXTRA_ARGS, not as positional args:
# everything after <stage> is treated as a config path, so a bare --force would be
# read as a second config and fail.
#
# Usage: bash run_stage.sh <stage> <config> [more configs...]
#        STAGE_EXTRA_ARGS=--force bash run_stage.sh evaluate configs/foo.yaml
set -euo pipefail

STAGE="$1"; shift
: "${MAMBA_ROOT_PREFIX:=/scratch/$USER/micromamba}"
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR/benchmark"

# CPU-only allocation: no visible GPU, so torch cannot autodetect an arch if
# anything in the eval path JIT-compiles a CUDA extension.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"

for cfg in "$@"; do
    echo "=== $STAGE.py ($cfg) ==="
    "$MAMBA_ROOT_PREFIX/envs/bench/bin/python" "$STAGE.py" --config "$cfg" ${STAGE_EXTRA_ARGS:-}
done
