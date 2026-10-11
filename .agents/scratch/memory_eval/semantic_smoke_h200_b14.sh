#!/usr/bin/env bash
# H200 smoke for the fast recipe + Chorus semantic queries: batch 14 at the worst-case length (N=160),
# compile off vs on, --loss-keep 0.5, and the async validator end to end.
# Usage: sbatch budget_sampler_smoke_h200.sh
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh >/dev/null
dataset pull lingbot-tapcache-v4-40 >/dev/null
PY="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"
TR=$(ls -d /data/lingbot-tapcache-v4-40/train/* | head -16)
VA=$(ls -d /data/lingbot-tapcache-v4-40/val_top/* | head -2)
ARGS="--head smallread_lingbot --dec-depth 8 --write-oneway --read-depth 8 --fresh-head --pts-frame world
      --patience 0 --batch 14 --lr 1e-4 --wd 0.05 --warmup 300 --max-frames 160 --clip-store cpu
      --frames-min 160 --frames 160 --frames-step 16 --probe-stops 16 --tbptt 0 --head-bf16 --head-down 2
      --loss-keep 0.5 --sem-feat upcast --sem-pixels 64 --sem-frames 4 --sem-val-pixels 64 --async-val --val-every 2 --val-frames 96 --save-every 100000 --viz-every 100000
      --wandb off --log-every 1 --updates 7"
export PYTORCH_CUDA_ALLOC_CONF=expandable_segments:True
for tag in sem_b14; do
    out=/scratch/$USER/smoke_fast_$tag; rm -rf "$out"; mkdir -p "$out"
    extra=""; [ $tag = compile ] && extra="--compile"
    # shellcheck disable=SC2086
    "$PY" scripts/memory/train_state.py --clips $TR --val-clips $VA --out "$out" $ARGS $extra --validator \
        > "$out/val.log" 2>&1 &
    vpid=$!
    # shellcheck disable=SC2086
    "$PY" scripts/memory/train_state.py --clips $TR --val-clips $VA --out "$out" $ARGS $extra 2>&1 \
        | grep -E "^\[ *[0-9]+\]|^\[head-train|^\[sem|^\[data|Error|Traceback" | sed "s/^/[$tag] /"
    touch "$out/val_queue/DONE"; wait $vpid || true
    grep -E "^\[val\]|^\[validator|^\[sem|Traceback|Error" "$out/val.log" | sed "s/^/[$tag val] /" | cut -c1-400
    ls "$out" "$out/val_queue" | tr '\n' ' ' | sed "s/^/[$tag files] /"; echo
done
