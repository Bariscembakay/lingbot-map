#!/usr/bin/env bash
# Run one ZipMap_eval suite: $1 = entry script (relative), rest = hydra overrides.
set -uo pipefail
ENTRY="$1"; shift
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_eval_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/zipmap_eval/bin/python"
cd "$HOME/lingbot-map/ZipMap_eval"
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KEEP_ALIVE_PID=$!
trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
"$P" "$ENTRY" "$@"
RC=$?
echo "=== $ENTRY exit $RC ==="
exit $RC
