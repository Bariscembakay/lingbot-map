#!/usr/bin/env bash
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=8
#SBATCH --mem=160G
#SBATCH --time=01:30:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_poserefine_logs/%j_%x.log
# KV-cache refiner smoke: cache reconstruction check, then a few steps of each arm and a VBR eval.
set -euo pipefail
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh
dataset pull lingbot-poserefine-v1 >/dev/null
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
IN=/data/lingbot-poserefine-v1
S=/scratch/$USER/poserefine_kv_smoke
# "$P" scripts/memory/pose_refine/verify_kv_reconstruction.py --scene /group/compact-3dmem/campaigns/lingbot_map/default/kitti/04
C="--inputs $IN --memory kvcache --anchor-frames 8 --scale-free-loss --norm-trans --rel-w 1 --train-store cpu
   --train-dirs train:0.45,kitti360_train:0.35,vkitti2:0.2 --t-min 32 --t-max 2048 --t-step 32 --t-max-start 1024
   --frames-per-batch 4096 --max-frames-rope 20000 --steps 30 --val-every 30 --warmup 5 --select val24,kitti360_val"
"$P" scripts/memory/pose_refine/train_pose_refine.py $C --variant tokres_frozencausal --no-head-rope \
    --pose-emb --kv-adapter --reg-w 0.1 --eval-sets val24:320,kitti360_val:full,kitti:full --out $S/A
"$P" scripts/memory/pose_refine/train_pose_refine.py $C --variant tokres_bidirhead \
    --eval-sets val24:320,kitti360_val:full,nrgbd:500,vbr:full --out $S/B
