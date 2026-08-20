#!/usr/bin/env bash
# Warm lingbot-map's viewer cache: GT-aligned point clouds + RGB thumbnails.
#
# NOT ON A LOGIN NODE. Thumbnail generation calls loader.load_rgb_list(), which
# reads EVERY frame of a scene into memory before resizing -- oxford_spires dense
# is 3,840 frames at 518x378x3, about 2.2 GB for one scene, and the login nodes
# have a per-user memory cgroup that OOM-kills work of that size. 16 workers each
# holding one scene's frames is what --mem is for.
#
# CPU-only: np.load, backprojection, PIL resize. No model, no GPU.
#
# Usage: submit_warm_cache.sh [WORKERS] [TIME] [DATASET]
set -euo pipefail

WORKERS="${1:-16}"
TIME_LIMIT="${2:-6:00:00}"
DATASET="${3:-}"

REPO="/home/baris_bakay/lingbot-map"
WORKSPACE="/group/compact-3dmem/campaigns/reproduction/viewer_workspace"
LOGDIR="/group/compact-3dmem/campaigns/_joblogs"
mkdir -p "$LOGDIR"

DS_ARG=""
[ -n "$DATASET" ] && DS_ARG="--dataset $DATASET"

SCRIPT=$(mktemp /tmp/warmcache_XXXXXX.sh)
cat > "$SCRIPT" <<EOS
#!/bin/bash
#SBATCH --job-name=warm_viewer_cache
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=$((WORKERS * 2))
#SBATCH --mem=180G
#SBATCH --time=${TIME_LIMIT}
#SBATCH --output=${LOGDIR}/warmcache_%j.log
#SBATCH --error=${LOGDIR}/warmcache_%j.err
set -euo pipefail
export PATH="\$HOME/bin:\$PATH"      # the conda shim is not on PATH in job shells
cd ${REPO}/benchmark
micromamba run -n lingbot_map python \
  ../.agents/scratch/reproduction/precompute_viewer_cache.py \
  ${WORKSPACE} --workers ${WORKERS} ${DS_ARG}
EOS

JOB=$(sbatch --parsable "$SCRIPT")
rm -f "$SCRIPT"
echo "submitted ${JOB}"
echo "  log: ${LOGDIR}/warmcache_${JOB}.log"
