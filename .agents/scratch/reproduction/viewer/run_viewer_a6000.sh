#!/usr/bin/env bash
# Requests an interactive a6000 on hala (rendering partition/qos, zone-sof1
# pinned) and launches the reproduction viser viewer there in the foreground.
# Ctrl+C tears down both the keep-alive and the srun allocation.
#
# --time=04:00:00: the cluster's Lua submit plugin rejects any interactive
# srun without an explicit time cap under 4h ("Use of srun for interactive
# jobs is forbidden for time longer than 4h") -- rendering's own MaxTime is
# unlimited, but that plugin check is separate. Re-run this script for
# another 4h session if you need longer.
#
# The viewer itself never touches the GPU, so without a keep-alive the idle-
# GPU reaper will kill this allocation while you're just browsing -- see
# gpu_keep_alive.py.
#
# --mem=220G: this cluster's documented "auto cpus-per-gpu/mem-per-gpu" plugin
# does NOT actually fire (disproven 2026-08-17, see CLAUDE.md) -- srun/sbatch
# jobs get Slurm's bare default (cpu=1, mem=2G) unless requested explicitly.
# The viewer builds a scene's whole point cloud in memory on first load
# (viewer.py's load_point_cloud_grid loads every frame's depth+RGB before any
# subsampling), and VBR's largest scenes (up to ~18,846 frames) need on the
# order of 100+ GB for that -- matches the precompute fix in
# precompute_viewer_cache.py's submitter.
#
# Usage: bash run_viewer_a6000.sh [workspace_dir] [port]
set -euo pipefail

WORKSPACE="${1:-/group/compact-3dmem/campaigns/reproduction/viewer_workspace}"
PORT="${2:-20540}"

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../../.." && pwd)"

srun --partition=rendering --qos=rendering --gres=gpu:a6000:1 \
    --cpus-per-task=4 --mem=220G \
    --constraint=zone-sof1 --time=04:00:00 --pty bash -c "
set -euo pipefail
cd '$REPO_DIR/benchmark'
export MAMBA_ROOT_PREFIX=/scratch/\$USER/micromamba

micromamba run -n lingbot_map python \
    ../.agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.05 &
KEEP_ALIVE_PID=\$!
trap 'kill \$KEEP_ALIVE_PID 2>/dev/null' EXIT

echo
echo \"Viewer node: \$(hostname)\"
echo \"From your machine:  ssh -L $PORT:localhost:$PORT \\\$USER@\$(hostname)\"
echo \"Then open:          http://localhost:$PORT\"
echo

micromamba run -n lingbot_map python \
    ../.agents/scratch/reproduction/viewer/launch_viewer.py '$WORKSPACE' $PORT
"
