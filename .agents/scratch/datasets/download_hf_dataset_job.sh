#!/usr/bin/env bash
#SBATCH --partition=login
#SBATCH --constraint=zone-msp3
#SBATCH --nodelist=msp3-login-3
#SBATCH --cpus-per-task=16
#SBATCH --mem=48G
#SBATCH --time=36:00:00
#SBATCH --output=/scratch/baris_bakay/%j_%x.log
# Download selected files of a HuggingFace dataset repo to node-local /scratch, unzip, register it
# as /data/<name> with `dataset create`, check file counts, then delete the staging copy.
# Runs as a CPU-only job on an msp3 login-partition node (needs ~3x the repo size free on /scratch for
# depth stored as .npy); the dataset is created in zone-msp3, so sof1 jobs `dataset pull` it.
# Usage: [UNZIP_INTO_PARENT=1] sbatch -J <name> download_hf_dataset_job.sh <dataset name> <hf repo> <allow pattern>...
set -euo pipefail
NAME=$1 REPO=$2; shift 2
S=/scratch/$USER/dl_$NAME
V=/scratch/$USER/hfvenv
mkdir -p "$S/zips" "$S/data"
[ -x "$V/bin/python" ] || python3 -m venv "$V"
"$V/bin/pip" install -q -U huggingface_hub hf_transfer
export HF_HUB_ENABLE_HF_TRANSFER=1 HF_HUB_DISABLE_PROGRESS_BARS=1
# a rerun after a failed unzip/register must not re-download: unzipped zips are already deleted
if [ -e "$S/.downloaded" ] || [ -n "$(ls -A "$S/data")" ]; then
    echo "[dl] download already done, resuming"
else
    echo "[dl] $(date) $REPO -> $S patterns: $*"
    "$V/bin/python" - "$REPO" "$S/zips" "$@" <<'PY'
import sys
from huggingface_hub import snapshot_download
repo, out, pats = sys.argv[1], sys.argv[2], sys.argv[3:]
for attempt in range(5):   # resumable: finished files are skipped on retry
    try:
        snapshot_download(repo, repo_type="dataset", allow_patterns=pats, local_dir=out, max_workers=16)
        break
    except Exception as e:
        print(f"[dl] attempt {attempt} failed: {e}", flush=True)
else:
    sys.exit(1)
PY
    touch "$S/.downloaded"
fi
echo "[dl] $(date) downloaded $(find "$S/zips" -name '*.zip' | wc -l) zips, $(du -sh "$S/zips" | cut -f1)"
first=$(find "$S/zips" -name '*.zip' | sort | head -1 || true)
[ -z "$first" ] || { echo "[dl] sample zip $first:"; unzip -l "$first" | sed -n '4,8p;/pose/p' | head -12 || true; }
# TartanAir V1/V2 zips carry their env/difficulty/trajectory prefix; TartanGround zips do not (every
# zip holds a bare image_lcam_front/, pose_*.txt), so with UNZIP_INTO_PARENT=1 each zip unpacks into
# the directory it sits in. Without it, unzip -n would silently skip every later trajectory's files.
unz='d="$2"; [ "${UNZIP_INTO_PARENT:-0}" = 1 ] && d="$2/$(dirname "${1#$3/}")"; mkdir -p "$d" && unzip -q -n "$1" -d "$d" && rm "$1"'
find "$S/zips" -name '*.zip' -print0 | xargs -0 -P 16 -I{} sh -c "$unz" _ {} "$S/data" "$S/zips" || true
# parallel unzips race on creating the same env/difficulty dirs ("checkdir error: File exists"); redo those serially
find "$S/zips" -name '*.zip' -print0 | xargs -0 -r -I{} sh -c "$unz" _ {} "$S/data" "$S/zips" || true
left=$(find "$S/zips" -name '*.zip' | wc -l)
[ "$left" -eq 0 ] || { echo "[dl] $left zips failed to unzip"; exit 1; }
echo "[dl] $(date) unzipped: $(du -sh "$S/data" | cut -f1), $(find "$S/data" -type f | wc -l) files"
find "$S/data" -maxdepth 4 -type d | sort | head -15 || true  # head closes the pipe early; pipefail would abort
find "$S/data" -name '*pose*' | head -3 || true
dataset create "$NAME" "$S/data"
dataset pin "$NAME" || true
n_src=$(find "$S/data" -type f | wc -l)
n_dst=$(find "/data/$NAME" -type f | wc -l)
echo "[dl] $(date) registered /data/$NAME: $n_dst files (staging $n_src)"
[ "$n_src" -eq "$n_dst" ] || { echo "[dl] file count mismatch, staging kept at $S"; exit 1; }
rm -rf "$S"
echo "[dl] done"
