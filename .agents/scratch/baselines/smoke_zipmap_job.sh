#!/usr/bin/env bash
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/zipmap/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
cd "$HOME/lingbot-map/ZipMap"
exec "$P" "$HOME/lingbot-map/.agents/scratch/baselines/smoke_zipmap.py"
