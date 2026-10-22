#!/usr/bin/env bash
#SBATCH --cpus-per-task=8
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# eval_refiner_bench.py in the sof1 zone (benchmark trees live on sof1's /group; gcp-eu1 shares it).
# Usage: sbatch ... eval_refiner_bench_job.sh TASK BENCHES [SCENES]   (refiner run dirs in $RUNS, comma-separated)
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 & KA=$!
trap 'kill $KA 2>/dev/null' EXIT
"$P" scripts/memory/pose_refine/eval_refiner_bench.py "$1" --benches "$2" ${3:+--scenes $3} --runs "${RUNS:-}"
