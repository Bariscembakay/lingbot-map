#!/usr/bin/env bash
# Smoke both vendored baselines on one GPU node: env build + tiny inference.
set -uo pipefail
echo "=== node $(hostname), gpu: $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) ==="
dataset pull NRGBD >/dev/null 2>&1 || true

echo "=== 1. TTT3R (cut3r env) ==="
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
set +e +u
PY_C="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
(cd "$HOME/lingbot-map/TTT3R" && "$PY_C" "$HOME/lingbot-map/.agents/scratch/baselines/smoke_ttt3r.py")
TTT_RC=$?
echo "=== TTT3R exit: $TTT_RC ==="

echo "=== 2. ZipMap (zipmap env) ==="
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_env.sh"
set +e +u
PY_Z="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/zipmap/bin/python"
(cd "$HOME/lingbot-map/ZipMap" && "$PY_Z" "$HOME/lingbot-map/.agents/scratch/baselines/smoke_zipmap.py")
ZIP_RC=$?
echo "=== ZipMap exit: $ZIP_RC ==="
exit $(( TTT_RC || ZIP_RC ))
