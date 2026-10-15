#!/usr/bin/env bash
#SBATCH --job-name=poserefine_BUILD_inputs_specials_taps4-11-17-23_scenes96_val24_nrgbd7_cpu
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=8
#SBATCH --mem=48G
#SBATCH --time=02:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_build.log
# Pose-refiner inputs: specials of taps 4/11/17/23 + lingbot pose_enc + GT, per clip.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
for d in lingbot-tapcache-v4-40 lingbot-tapcache-v4-ext64 lingbot-tapcache-v5-val16-4tap; do dataset pull $d >/dev/null; done
OUT=/group/compact-3dmem/datasets/lingbot-poserefine-v1
L=$(mktemp -d)
ls -d /data/lingbot-tapcache-v4-40/train/* /data/lingbot-tapcache-v4-ext64/train/* > $L/train.txt
sed 's#/data/lingbot-tapcache-v5-tap23/val/#/data/lingbot-tapcache-v5-val16-4tap/val/#' \
    .agents/scratch/memory_eval/val24.txt | tr ' ' '\n' | grep . > $L/val24.txt
ls -d /group/compact-3dmem/campaigns/spatial_memory/eval_nrgbd_cache/*_500f_s2 > $L/nrgbd.txt
wc -l $L/*.txt
for s in train val24 nrgbd; do
    "$P" scripts/memory/pose_refine/build_inputs.py --clips $L/$s.txt --out $OUT/$s
done
