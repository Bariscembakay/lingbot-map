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

# 4, not 16. Job 726896 ran 16 and reached 181 GB total RSS against a 180 GB
# request, with every worker stuck 2h17m on its first oxford_spires entry and no
# completions. oxford dense is 3,840 frames per scene and _generate_point_clouds
# holds them all: 18-30 GB per worker, so concurrency is bounded by memory, not
# cores. The light datasets (108 of 310 entries) finished in the first 7 minutes
# at 16 workers -- it is only the big ones that need the smaller pool, and a
# restart skips everything already cached, so there is no cost to rerunning all.
WORKERS="${1:-4}"
TIME_LIMIT="${2:-12:00:00}"
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
# /scratch is per-node local disk and wiped weekly, so the env may not exist on
# whichever node this lands on -- job 726886 died with "the given prefix does not
# exist" for exactly that reason. This bootstrap is idempotent (a fast no-op when
# the env is already built) and takes a flock, so two jobs packed onto one node
# wait rather than racing on the install. It also puts ~/bin on PATH for the
# conda->micromamba shim, which job shells do not get by default.
source ${REPO}/.agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
cd ${REPO}/benchmark
micromamba run -n lingbot_map python \
  ../.agents/scratch/reproduction/precompute_viewer_cache.py \
  ${WORKSPACE} --workers ${WORKERS} ${DS_ARG}
EOS

JOB=$(sbatch --parsable "$SCRIPT")
rm -f "$SCRIPT"
echo "submitted ${JOB}"
echo "  log: ${LOGDIR}/warmcache_${JOB}.log"
