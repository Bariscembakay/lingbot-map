#!/usr/bin/env bash
# 2-rank DDP smoke through the real job script: coalesced all_reduce runs, and
# the keep-alive dies once training starts.
set -euo pipefail
cd /home/baris_bakay/lingbot-map
OUT=/scratch/$USER/ddp_smoke_$SLURM_JOB_ID
dataset pull lingbot-tapcache-v4-40 >/dev/null
TR=$(ls -d /data/lingbot-tapcache-v4-40/train/* | head -4 | tr '\n' ' ')
( for i in $(seq 1 40); do sleep 30; echo "[watch $((i*30))s] keep_alive procs: $(pgrep -u $USER -fc gpu_keep_alive.py || true)"; done ) &
W=$!
KEEPALIVE=1 bash scripts/memory/train_state_ddp_job.sh "$OUT" "$TR" \
    --val-clips /data/lingbot-tapcache-v4-40/val_top/210f741378_c0 \
    --head smallread_lingbot --dec-depth 8 --read-depth 8 --write-oneway --lora-head --clip-store cpu \
    --init-from /group/compact-3dmem/campaigns/spatial_memory/scenes944_96f_b12x2_write8oneway_read8_tap23_lingbothead/best.pt \
    --frames 8 --max-frames 16 --batch 2 --updates 6 --val-every 3 --save-every 3 --log-every 1 \
    --n-past 4 --probe-every 1 --tbptt 8 --wandb off 2>&1 | grep -E "^\[(ddp|data|init|lora-head|val|resume)\]|^\[ +[0-9]+\]|Error|error|Traceback" || true
kill $W 2>/dev/null || true
echo "after job: keep_alive procs: $(pgrep -u $USER -fc gpu_keep_alive.py || true)"
"${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python" -c "import json;h=json.load(open('$OUT/history.json'));print('steps',[r['step'] for r in h]);print('val',[(r['step'],round(r['valm_self'],4)) for r in h if 'valm_self' in r])"
rm -rf "$OUT" /scratch/$USER/train_state/ddp_smoke_$SLURM_JOB_ID
