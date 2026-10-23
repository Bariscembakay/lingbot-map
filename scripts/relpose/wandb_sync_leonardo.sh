#!/usr/bin/env bash
# Upload Leonardo trainings to wandb from INSAIT: Leonardo's compute nodes are offline and the wandb key stays on
# sof1. Each round pulls every arm's wandb/ dir and its wandb_id, and syncs that arm's offline runs (one per
# --resume job, all named by the id) to the one wandb run; a re-sync uploads only the steps added since the last.
# A lapsed Leonardo certificate only skips rounds until it is renewed.
# Usage: wandb_sync_leonardo.sh [remote runs dir] [seconds between rounds]
set -uo pipefail
R=${1:-/leonardo_scratch/large/userexternal/bbakay00/runs/lingbot_rel/backbone}
EVERY=${2:-900}
L=/scratch/$USER/wandb_leonardo/$(basename "$R")
W=${WANDB_BIN:-/scratch/$USER/micromamba/envs/wandb_sync/bin/wandb}
mkdir -p "$L"
while :; do
    if rsync -am --include='*/' --include='wandb_id' --include='wandb/***' --exclude='*' \
            -e 'ssh -o BatchMode=yes -o ConnectTimeout=30' "leonardo:$R/" "$L/" 2>/dev/null; then
        for a in "$L"/*/; do
            [ -f "$a/wandb_id" ] || continue
            id=$(cat "$a/wandb_id")
            for d in "$a"/wandb/offline-run-*-"$id"; do
                [ -d "$d" ] || continue
                f=$(ls "$d"/*.wandb 2>/dev/null | head -1)
                [ -n "$f" ] && [ -f "$f.synced" ] && [ ! "$f" -nt "$f.synced" ] && continue
                "$W" sync --include-synced --id "$id" "$d" 2>&1 | grep -iE "error|warn" | head -3
            done
        done
        echo "[wandb_sync_leonardo] $(date -u +%H:%M) synced"
    else
        echo "[wandb_sync_leonardo] $(date -u +%H:%M) leonardo unreachable (certificate?)"
    fi
    sleep "$EVERY"
done
