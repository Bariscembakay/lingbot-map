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
# NRGBD ships raw `depth/`, `depth_filtered/` and `depth_with_noise/`. The loader
# reads `depth/`, but reconstruction papers conventionally score against the
# filtered variant, which masks ~8% of unreliable pixels. Which one upstream used
# is not documented, so it is selectable: NRGBD_DEPTH=depth_filtered builds a
# shadow tree whose `depth` points at the filtered maps.
NRGBD_DEPTH="${NRGBD_DEPTH:-depth}"
if [ "$NRGBD_DEPTH" = "depth" ]; then
    ln -sfn /data/NRGBD "$CUT3R_DIR/data/neural_rgbd"
else
    SHADOW="$CUT3R_DIR/data/neural_rgbd_$NRGBD_DEPTH"
    for sc in /data/NRGBD/*/; do
        name="$(basename "$sc")"
        mkdir -p "$SHADOW/$name"
        ln -sfn "$sc/images"          "$SHADOW/$name/images"
        ln -sfn "$sc/$NRGBD_DEPTH"    "$SHADOW/$name/depth"
        ln -sfn "$sc/poses.txt"       "$SHADOW/$name/poses.txt"
        ln -sfn "$sc/focal.txt"       "$SHADOW/$name/focal.txt"
    done
    ln -sfn "$SHADOW" "$CUT3R_DIR/data/neural_rgbd"
fi
ln -sfn /group/compact-3dmem/checkpoints/CUT3R/cut3r_512_dpt_4_64.pth \
        "$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth"

# launch.py always CONSTRUCTS both datasets before --datasets can filter them,
# and SevenScenes.load_all_scenes os.listdir()s its root unconditionally. An
# empty directory lists to zero scenes, so the entry costs nothing and is then
# dropped by the filter. Replace this with a real link if 7-Scenes is ever
# wanted -- it needs `.depth.proj.png`, which we do not have (see UPSTREAM.md).
mkdir -p "$CUT3R_DIR/data/7scenes"

# add_ckpt_path.py puts dirname(weights) on sys.path so `dust3r` resolves, so the
# weights argument must be the in-repo symlink, not the /group path it points to.
# NPROC>1 uses `accelerate launch`, which is how upstream's run.sh invokes this
# (--num_processes 8). launch.py shards with accelerator.split_between_processes,
# whose default apply_padding=False means each scene is still evaluated exactly
# once -- but that is reasoning, and this switch lets it be measured.
NPROC="${NPROC:-1}"
if [ "$NPROC" -gt 1 ]; then
    "${MAMBA_ROOT_PREFIX:-/scratch/$USER/micromamba}/envs/cut3r/bin/accelerate" launch \
        --num_processes "$NPROC" --main_process_port 29501 eval/mv_recon/launch.py \
        --weights "$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth" \
        --output_dir "$OUT" --model_name ours --size 512 "$@"
else
    "$PY_ENV" eval/mv_recon/launch.py \
        --weights "$CUT3R_DIR/src/cut3r_512_dpt_4_64.pth" \
        --output_dir "$OUT" \
        --model_name ours \
        --size 512 \
        "$@"
fi

# /scratch is per-node local disk, so anything left there is invisible from the
# submitting node -- the first successful run stranded its logs_all.txt on a gcp
# node. Ship the whole workspace to sof1's /group, which is the system of record.
SHIP="${SHIP_TO:-/group/compact-3dmem/campaigns/spatial_memory/cut3r_repro}"
mkdir -p "$SHIP"
rsync -a "$OUT/" "$SHIP/$(basename "$OUT")/" || true
echo "[run_cut3r_mv_recon] shipped $OUT -> $SHIP/$(basename "$OUT")"
