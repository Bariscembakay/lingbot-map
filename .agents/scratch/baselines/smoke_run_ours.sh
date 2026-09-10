#!/usr/bin/env bash
# Smoke: exercise run_ours.py end-to-end on a tiny tier before the real array
# (873600) commits 7 tasks to code that has never executed. Writes to /scratch,
# never to the campaign record.
set -uo pipefail
CAMP=/group/compact-3dmem/campaigns/spatial_memory
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
"$P" scripts/memory/recall_bench/run_ours.py \
  --scene whiteroom \
  --ckpt "$CAMP/scenes96_96f_b4_write4_read2_lingbothead_unfrozenhead/best.pt" \
  --cache "$CAMP/eval_nrgbd_cache/whiteroom_500f_s2" \
  --tiers 8 --out "/scratch/$USER/ours_smoke"
echo "[smoke] rc=$?"
ls -l "/scratch/$USER/ours_smoke/ours/whiteroom_n8/" 2>/dev/null
