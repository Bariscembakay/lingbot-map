#!/usr/bin/env bash
#SBATCH --partition=login
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/%j_%x.log
# Download files of a Hugging Face dataset repo (HF_TOKEN from ~/.secrets for gated repos) to node-local
# /scratch, extract every .tar / .zip where it lies (repo paths kept), register /data/<name>.
# Usage: sbatch -J <long name> download_hf_tar_job.sh <dataset name> <hf repo> <allow pattern>...
set -euo pipefail
source /home/baris_bakay/lingbot-map/.agents/scratch/datasets/_dl_lib.sh
[ -f "$HOME/.secrets" ] && . "$HOME/.secrets"   # HF_TOKEN for gated repos
NAME=$1 REPO=$2; shift 2
S=/scratch/$USER/dl_$NAME
mkdir -p "$S/data"
ensure_dlenv
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1
log "$NAME on $(hostname): $REPO patterns $*"
"$DLPY" - "$REPO" "$S/data" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
for attempt in range(5):   # finished files are skipped on retry
    try:
        snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=out, max_workers=16)
        break
    except Exception as e:
        print(f"[dl] attempt {attempt} failed: {e}", flush=True)
else:
    sys.exit(1)
PY
rm -rf "$S/data/.cache"
log "downloaded $(du -sh "$S/data" | cut -f1)"
# split archives (x.tar00, x.tar01, ...) are one tar stream
find "$S/data" -name '*.tar00' | while read -r f; do b=${f%00}; cat "$b"[0-9][0-9] | tar -x -C "$(dirname "$f")" && rm "$b"[0-9][0-9]; done
find "$S/data" -name '*.tar' -print0 | xargs -0 -r -P 8 -I{} sh -c 'tar -xf "$1" -C "$(dirname "$1")" && rm "$1"' _ {}
unzip_all "$S/data"
find "$S/data" -maxdepth 4 -type d | sort | head -20 || true
register_dir "$NAME" "$S/data"
