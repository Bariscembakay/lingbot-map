#!/usr/bin/env bash
# Unit checks, then 4 real updates per arm and a resume, on 2 tiny clips.
set -euo pipefail
cd /home/baris_bakay/lingbot-map
export PYTHONPATH=/home/baris_bakay/lingbot-map
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
"$PY" .agents/scratch/memory_eval/pts3d_smoke.py
dataset pull lingbot-tapcache-v4-40 >/dev/null
TR=(/data/lingbot-tapcache-v4-40/train/*)
ROOT=/scratch/$USER/pts3d_smoke_$SLURM_JOB_ID
for arm in "--pts-frame cam" "--pts-frame world" "--fresh-last-conv"; do
    out="$ROOT/$(echo $arm | tr -d ' -')"
    common=(--clips "${TR[0]}" "${TR[1]}" --val-clips /data/lingbot-tapcache-v4-40/val_top/210f741378_c0
        --head smallread_lingbot --dec-depth 4 --read-depth 2 --lora-head $arm
        --init-from /group/compact-3dmem/campaigns/spatial_memory/scenes96_96f_b4_write4_read2_lingbothead/last.pt
        --frames 8 --max-frames 16 --batch 1 --n-past 4 --probe-every 1 --tbptt 8
        --val-every 2 --save-every 2 --viz-every 1000 --log-every 1 --wandb off --resume auto --out "$out")
    echo "=== $arm"
    "$PY" scripts/memory/train_state.py "${common[@]}" --updates 4 2>&1 | grep -E "head-output|lora-head|model\]|^\[ +[0-9]+\]|valm|Error|error" || true
    "$PY" scripts/memory/train_state.py "${common[@]}" --updates 6 2>&1 | grep -E "resume|^\[ +[0-9]+\]|Error|error" || true
    "$PY" -c "import json;h=json.load(open('$out/history.json'));print([ (r['step'],round(r['valm_self'],4),round(r.get('valm_offray_deg',-1),3)) for r in h if 'valm_self' in r])"
done
rm -rf "$ROOT"
