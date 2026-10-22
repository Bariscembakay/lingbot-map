#!/usr/bin/env bash
#SBATCH --cpus-per-task=4
#SBATCH --mem=32G
#SBATCH --time=00:30:00
set -euo pipefail
cd "$HOME/lingbot-map"
git pull -q --ff-only || true
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
"$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python" scripts/memory/pose_refine/flex_kernel_bench.py "$@"
