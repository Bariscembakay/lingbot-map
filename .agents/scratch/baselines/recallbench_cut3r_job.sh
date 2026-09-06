#!/usr/bin/env bash
# recall bench, CUT3R family: $1 = update rule (cut3r|ttt3r), $2.. = scenes
set -uo pipefail
RULE="$1"; shift
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
trap 'kill $! 2>/dev/null' EXIT
cd "$HOME/lingbot-map"
RC=0
for SCENE in "$@"; do
  "$P" scripts/memory/recall_bench/run_cut3r_family.py --update-rule "$RULE" \
    --scene "$SCENE" --out /group/compact-3dmem/campaigns/spatial_memory/recallbench || RC=1
done
exit $RC
