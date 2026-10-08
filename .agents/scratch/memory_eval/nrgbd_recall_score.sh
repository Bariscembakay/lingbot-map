#!/usr/bin/env bash
# NRGBD recall benchmark, CPU phase: score one scene's n100/n300/n500 dumps
# (posed, no Sim(3)). Skips cells already scored; clouds (~1.5 GB/cell) only
# for PLY_SCENE at n500.
# Usage: sbatch --array=0-6 nrgbd_recall_score.sh <method> [ply_scene]
set -uo pipefail
METHOD=$1; PLY_SCENE=${2:-whiteroom}
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
rc=0
for T in 100 300 500; do
    od="$CAMP/recallbench/$METHOD/${S}_n$T"
    want_ply=0; [ "$S" = "$PLY_SCENE" ] && [ "$T" = 500 ] && want_ply=1
    if [ -f "$od/metrics.json" ] && { [ $want_ply = 0 ] || [ -f "$od/recalled_cloud_err.ply" ]; }; then
        echo "[skip] $S n$T scored"; continue
    fi
    dump="$CAMP/recallbench_dumps/$METHOD/${S}_n$T.npz"
    [ -f "$dump" ] || { echo "[missing dump] $dump"; rc=1; continue; }
    ply=--no-ply; [ $want_ply = 1 ] && ply=""
    "$P" scripts/memory/recall_bench/run_recall_score.py --method "$METHOD" --posed $ply \
        --scene "$S" --tier "$T" --dump-dir "$CAMP/recallbench_dumps" --out "$CAMP/recallbench" || rc=1
done
exit $rc
