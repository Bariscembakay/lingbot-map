#!/usr/bin/env bash
#SBATCH --account=AIFAC_S07_110
#SBATCH --partition=boost_usr_prod
#SBATCH --gres=gpu:1
#SBATCH --cpus-per-task=8
#SBATCH --mem=120G
#SBATCH --output=/leonardo_scratch/large/userexternal/bbakay00/logs/%x_%j.out
# One train_rel_head.py arm on CINECA Leonardo (A100 64 GB). Everything lives in $CINECA_SCRATCH:
# code (git), env (setup_lingbot_env_leonardo.sh), data copied from INSAIT, run dirs; finished runs
# are copied back to INSAIT /group by hand (sof1 pulls them). Compute nodes have no internet, so
# wandb runs offline (sync from a login node).
# Usage: sbatch -J <name> --time=.. relhead_leonardo_job.sh <arm> [train_rel_head.py args]
set -euo pipefail
S=$CINECA_SCRATCH
cd $S/lingbot-map-rel
P=$S/micromamba/envs/lingbot_map/bin/python
"$P" -c "import torch, sys; sys.exit(0 if torch.cuda.is_available() else 1)" || { echo "torch sees no GPU"; nvidia-smi; exit 4; }
export WANDB_MODE=offline RELHEAD_TOKENS=$S/relhead_cam23
echo "[relhead_leonardo_job] $(hostname) $(nvidia-smi --query-gpu=name --format=csv,noheader | head -1) git $(git rev-parse --short HEAD)"
ARM=$1; shift
# data still arriving from INSAIT: rsync renames a file into place only when it is complete
for ds in tartanair-v1 tartanair-v2 tartanground matrixcity scannetpp scannet-v2; do
    until [ -f $S/data/lingbot-tap23pack-1048/$ds/spec23.npy ] && [ -f $S/data/lingbot-tap23pack-1048/$ds/index.npz ]; do
        echo "[relhead_leonardo_job] waiting for $ds"; sleep 60; done
done
"$P" scripts/relpose/train_rel_head.py --out $S/runs/lingbot_rel/relhead/$ARM \
    --pack-root $S/data/lingbot-tap23pack-1048 --inputs $S/data/lingbot-poserefine-v1 "$@"
