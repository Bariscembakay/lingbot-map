#!/usr/bin/env bash
# val24 recall comparison, GPU dumps: BIGRUN, FASTRECIPE xyz / +semantics, CUT3R, TTT3R
# (scripts/memory/recall_bench/val24_dump.py). Score afterwards on CPU: val24_score.py.
set -uo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
for d in lingbot-tapcache-v4-40 lingbot-tapcache-v5-tap23 ScanNetpp; do dataset pull $d >/dev/null 2>&1; done
# CUT3R/TTT3R decode each clip's frames on the CPU between GPU passes.
"$P" "$HOME/ASVGGT/scratch/lib/gpu_keep_alive.py" 0.05 &
KA=$!; trap 'kill $KA 2>/dev/null' EXIT
OUT=/group/compact-3dmem/campaigns/spatial_memory/val24_compare
rc=0
"$P" scripts/memory/recall_bench/val24_dump.py --method ours --label ours_BIGRUN944_s5200 --ckpt /group/compact-3dmem/campaigns/spatial_memory/scenes944_96f_b12x4_write8oneway_read8_tap23_lingbothead_LORAr16_h200/best_step5200.pt --dump $OUT || rc=1
"$P" scripts/memory/recall_bench/val24_dump.py --method ours --label ours_FASTRECIPE_xyz96_s5500 --ckpt /group/compact-3dmem/campaigns/spatial_memory/scenes96_N64-160_stops16k2_b14_write8oneway_read8_tap23_FASTRECIPE_xyzworld_lr3.5e-4_scratch/best_step5500.pt --dump $OUT || rc=1
"$P" scripts/memory/recall_bench/val24_dump.py --method ours --label ours_FASTRECIPE_xyz96_sem_s3000 --ckpt /group/compact-3dmem/campaigns/spatial_memory/scenes96_N64-160_stops16k2_b14_write8oneway_read8_tap23_FASTRECIPE_xyzworld_semupcast_f4p4_lr3.5e-4_scratch/best_step3000.pt --dump $OUT || rc=1
"$P" scripts/memory/recall_bench/val24_dump.py --method cut3r --label cut3r_512 --dump $OUT || rc=1
"$P" scripts/memory/recall_bench/val24_dump.py --method ttt3r --label ttt3r_512 --dump $OUT || rc=1
exit $rc
