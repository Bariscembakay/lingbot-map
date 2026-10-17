#!/usr/bin/env bash
#SBATCH --job-name=recallbench_nrgbd_ICPDIAG_sim3icp_ours96_bigrun_ttt3r_cut3r_n100_n500_cpu
#SBATCH --partition=batch
#SBATCH --constraint=zone-sof1
#SBATCH --cpus-per-task=16
#SBATCH --mem=96G
#SBATCH --time=01:30:00
#SBATCH --array=0-6
#SBATCH --output=/group/compact-3dmem/campaigns/spatial_memory/_icp_diag/_joblogs/%A_%a.log
set -uo pipefail
SCENES=(whiteroom kitchen grey_white_room green_room complete_kitchen breakfast_room staircase)
S=${SCENES[${SLURM_ARRAY_TASK_ID:?}]}
cd "$HOME/lingbot-map"
source .agents/scratch/insait_cluster_files/setup_cut3r_env.sh
set +u
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
M=ours_unfrozenhead96_s3400_rerun,ours_bigrun944_s5200,ttt3r,cut3r
for T in 100 500; do
    "$P" scripts/memory/recall_bench/icp_diag.py --scene "$S" --tier $T --methods $M \
        --out /group/compact-3dmem/campaigns/spatial_memory/_icp_diag/${S}_n$T.json
done
