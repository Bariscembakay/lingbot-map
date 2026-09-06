#!/usr/bin/env bash
# videodepth reproduction: infer then eval (two-stage per upstream README).
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_zipmap_eval_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/zipmap_eval/bin/python"
cd "$HOME/lingbot-map/ZipMap_eval"
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KEEP_ALIVE_PID=$!
trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
OV=(evaluation=videodepth 'eval_models=[ZipMap,ttt3r,cut3r]' 'eval_datasets=[kitti,bonn]')
"$P" videodepth/infer.py "${OV[@]}" || exit 1
"$P" videodepth/eval.py "${OV[@]}"
RC=$?
echo "=== videodepth exit $RC ==="
exit $RC
