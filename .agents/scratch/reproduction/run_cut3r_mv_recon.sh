#!/usr/bin/env bash
# Reproduce CUT3R's Table 4 / Table 5 on NRGBD.
#
# Usage: run_cut3r_mv_recon.sh <out_root> [extra args to launch.py]
#   e.g. ... online          -> Table 4
#   e.g. ... --revisit 2 --freeze  -> Table 5 "Ours Revisit"
set -euo pipefail

OUT="$1"; shift
REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
CUT3R_DIR="$REPO_DIR/CUT3R"

source "$REPO_DIR/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"

# Never `micromamba run` in a job: it locks under ~/.cache/mamba/proc on CephFS
# and a long job holds that lock for its whole lifetime.
PY_ENV="${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/python"

mkdir -p "$OUT"
cd "$CUT3R_DIR"

# Recreate the two links the eval needs. Done here rather than committed,
# because a symlink resolves only in the zone it was made in: /group is per zone
# and /data is an autofs registry mount that needs `dataset pull` first.
dataset pull NRGBD >/dev/null 2>&1 || true
mkdir -p "$CUT3R_DIR/data"
ln -sfn /data/NRGBD "$CUT3R_DIR/data/neural_rgbd"
ln -sfn /group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth \
        "$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth"

# add_ckpt_path.py puts dirname(weights) on sys.path so `dust3r` resolves, so the
# weights argument must be the in-repo symlink, not the /group path it points to.
"$PY_ENV" eval/mv_recon/launch.py \
    --weights "$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth" \
    --output_dir "$OUT" \
    --model_name ours \
    --size 512 \
    "$@"
