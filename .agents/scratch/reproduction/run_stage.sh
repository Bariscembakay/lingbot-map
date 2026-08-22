#!/usr/bin/env bash
# Runs one pipeline stage (prepare/evaluate/report) for a config, in the bench env.
# Split out from run_benchmark.sh so prepare and evaluate can be separate jobs
# with a Slurm dependency between them and the per-arm run jobs.
#
# Usage: bash run_stage.sh <stage> <config> [more configs...]
set -euo pipefail

STAGE="$1"; shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR/benchmark"

# CPU-only allocation: no visible GPU, so torch cannot autodetect an arch if
# anything in the eval path JIT-compiles a CUDA extension.
export TORCH_CUDA_ARCH_LIST="${TORCH_CUDA_ARCH_LIST:-8.0;8.6;9.0}"

for cfg in "$@"; do
    echo "=== $STAGE.py ($cfg) ==="
    micromamba run -n bench python "$STAGE.py" --config "$cfg"
done
