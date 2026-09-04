#!/usr/bin/env bash
# TTT3R repo's own mv_recon eval (7-Scenes, 13 seqs, kf_every=2, max_frames=200)
# in the mode given as $1 (ttt3r | cut3r). Secondary reproduction check against
# the paper's Figure 9; primary gate is ZipMap_eval Table 3.
set -uo pipefail
MODE="$1"
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
cd "$HOME/lingbot-map/TTT3R"
OUT="eval_results/video_recon/7scenes_200/$MODE"
mkdir -p "$OUT"
# The eval alternates GPU inference with long CPU chamfer phases -- exactly the
# idle-GPU reaper's kill profile (820955 was CANCELLED by root at 3/18 seqs).
# Keep-alive holds a small VRAM slice and issues calibrated matmul bursts.
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KEEP_ALIVE_PID=$!
trap 'kill $KEEP_ALIVE_PID 2>/dev/null' EXIT
NCCL_TIMEOUT=360000 "$P" -m accelerate.commands.launch --num_processes 1 --main_process_port $((29500 + RANDOM % 500)) \
  eval/mv_recon/launch.py \
  --weights src/cut3r_512_dpt_4_64.pth \
  --output_dir "$OUT" \
  --model_name "$MODE" \
  --model_update_type "$MODE" \
  --max_frames 200
RC=$?
echo "=== eval exit $RC ==="
tail -20 "$OUT"/logs_all.txt 2>/dev/null || ls -la "$OUT"
exit $RC
