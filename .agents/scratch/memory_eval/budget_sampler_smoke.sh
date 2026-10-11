#!/usr/bin/env bash
# Smoke + speed check: constant-budget sampler (+ bf16 / half-res head, full
# BPTT) vs the FRESHDPT_xyzworld recipe, from scratch, same 8 clips, same GPU.
# Usage: sbatch budget_sampler_smoke.sh
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh >/dev/null
dataset pull lingbot-tapcache-v4-40 >/dev/null
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
TR=$(ls -d /data/lingbot-tapcache-v4-40/train/* | head -8)
VA=$(ls -d /data/lingbot-tapcache-v4-40/val_top/* | head -2)
COMMON="--head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head --pts-frame world
        --val-every 100000 --patience 0 --batch 4 --lr 1e-4 --wd 0.05 --warmup 300 --max-frames 160
        --clip-store cpu --save-every 100000 --viz-every 100000 --wandb off --log-every 1"
run() {
    local tag=$1; shift
    local out=/scratch/$USER/smoke_budget_$tag; rm -rf "$out"
    # shellcheck disable=SC2086
    "$PY" scripts/memory/train_state.py --clips $TR --val-clips $VA --out "$out" $COMMON "$@" \
        2>&1 | grep -E "^\[ *[0-9]+\]|^\[head-train|^\[fresh|Error|Traceback|error" | sed "s/^/[$tag] /"
}
run new --updates 12 --frames-min 64 --frames 160 --frames-step 16 --probe-stops 16 --tbptt 0 \
    --head-bf16 --head-down 2 --val-frames 96
run old --updates 4 --frames 96 --n-past 4 --probe-every 1 --tbptt 8
