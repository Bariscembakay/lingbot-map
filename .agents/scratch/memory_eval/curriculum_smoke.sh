#!/usr/bin/env bash
# Tiny 3-stage chain: cut3r sampler, LoRA stage-to-stage init, variable
# length + revisits + grad-last, pinned val length.
set -euo pipefail
cd /home/baris_bakay/lingbot-map
export PYTHONPATH=/home/baris_bakay/lingbot-map
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
dataset pull lingbot-tapcache-v4-40 >/dev/null
TR=(/data/lingbot-tapcache-v4-40/train/*)
R=/scratch/$USER/curr_smoke_$SLURM_JOB_ID
common=(--clips "${TR[0]}" "${TR[1]}" "${TR[2]}" --val-clips /data/lingbot-tapcache-v4-40/val_top/210f741378_c0
    --head smallread_lingbot --dec-depth 4 --read-depth 2 --lora-head --sampler cut3r
    --max-frames 40 --val-frames 24 --tbptt 0 --n-past 4 --probe-every 1
    --val-every 2 --save-every 2 --viz-every 1000 --log-every 1 --wandb off --lr 1e-3 --warmup 1)
f='head-output|lora-head|init\]|model\]|^\[ +[0-9]+\]|^\[val\]|Error|error|Traceback'
echo "=== A: 4 frames"
"$PY" scripts/memory/train_state.py "${common[@]}" --frames 4 --batch 3 --updates 4 \
    --init-from /group/compact-3dmem/campaigns/spatial_memory/scenes96_96f_b4_write4_read2_lingbothead/last.pt --out $R/A 2>&1 | grep -E "$f" | sed 's/valx.*//'
echo "=== B: 16 frames, init from A (LoRA ckpt)"
"$PY" scripts/memory/train_state.py "${common[@]}" --frames 16 --batch 2 --updates 4 --init-from $R/A/best.pt --out $R/B 2>&1 | grep -E "$f" | sed 's/valx.*//'
echo "=== C: 4..32 frames, revisits, grad-last 8"
"$PY" scripts/memory/train_state.py "${common[@]}" --frames 32 --frames-min 4 --allow-repeat --grad-last 8 --batch 2 --updates 6 --init-from $R/B/best.pt --out $R/C 2>&1 | grep -E "$f" | sed 's/valx.*//'
"$PY" -c "import json;h=json.load(open('$R/C/history.json'));v=[r for r in h if 'valm_self' in r];print('valm==valx at pinned 24 frames:',[ (r['valm_self'],r['valx_self']) for r in v])"
rm -rf $R
