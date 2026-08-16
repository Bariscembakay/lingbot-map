#!/usr/bin/env bash
# Generic runner for the benchmark pipeline (prepare/run/evaluate) on sof1
# zone. Not tied to any one dataset -- pass whichever benchmark config you
# need. Assumes the config's `workspace`/`raw_data_root` already point at
# real sof1 paths (no zone override needed here, unlike msp3).
#
# Usage (as the payload of an srun/sbatch job -- pick GPU/time/mem/zone
# yourself, those vary per run and aren't baked in here):
#   srun --partition=batch --constraint=zone-sof1 --gpus=a6000:1 \
#        --cpus-per-task=16 --mem=96G --time=01:00:00 \
#        bash .agents/scratch/insait_cluster_files/submit_sof1.sh \
#        configs/oxford.yaml [prepare run evaluate]
set -euo pipefail

CONFIG="$1"
shift
STAGES=("${@:-prepare run evaluate}")
[ "${#STAGES[@]}" -eq 1 ] && STAGES=(${STAGES[0]})

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR"

source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
source .agents/scratch/insait_cluster_files/setup_bench_env.sh

cd benchmark
for stage in "${STAGES[@]}"; do
    echo "=== $stage.py ($CONFIG) ==="
    micromamba run -n bench python "$stage.py" --config "$CONFIG"
done
echo "=== DONE ==="
