#!/usr/bin/env bash
#SBATCH --partition=login
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=16
#SBATCH --mem=32G
#SBATCH --time=36:00:00
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_data_prep/%A_%a_%x.log
# DL3DV-ALL-960P, one 1K block per array task (~300 GB of zips each), registered as /data/dl3dv-960p-<k>k.
# Usage: sbatch --array=1-11%4 -J <long name> download_dl3dv_960p_array_job.sh
K=${SLURM_ARRAY_TASK_ID:?}
exec bash /home/baris_bakay/lingbot-map/.agents/scratch/datasets/download_hf_tar_job.sh dl3dv-960p-${K}k DL3DV/DL3DV-ALL-960P "${K}K/*"
