#!/usr/bin/env bash
# ZipMap paper Table 3 reproduction: mv_recon on 7-Scenes + NRGBD (sparse+dense)
# for ZipMap / TTT3R / CUT3R via the vendored ZipMap_eval harness.
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_eval_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/zipmap_eval/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
cd "$HOME/lingbot-map/ZipMap_eval"
# Same reaper profile as the TTT3R eval: GPU bursts + long CPU chamfer phases.
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KEEP_ALIVE_PID=$!
trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
"$P" mv_recon/eval.py "$@"
RC=$?
echo "=== eval exit $RC ==="
find outputs/mv_recon -name "_seq_metrics.csv" | head -20
exit $RC
