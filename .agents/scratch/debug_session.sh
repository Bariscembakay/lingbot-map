#!/usr/bin/env bash
# Run demo.py under debugpy on a GPU node and wait for VS Code to attach.
#
#   bash .agents/scratch/debug_session.sh [extra demo.py args...]
#
# Then run the "⚡ Attach to debugpy on GPU node" config in VS Code and give it
# the host this script prints.
#
# Env overrides:
#   DEBUG_GPU=h200:1 DEBUG_PARTITION=batch   # qos auto-drops to normal; no preemption
#   DEBUG_PORT=5679                          # if 5678 is taken
#   DEBUG_TIME=4:00:00                       # 4 h is the `debug` partition cap
#   DEBUG_QOS=rendering                      # override the partition->qos mapping
#   DEBUG_BACKEND=flashinfer                 # default sdpa (dense, inspectable KV)
set -euo pipefail

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"

PORT="${DEBUG_PORT:-5678}"
GPU="${DEBUG_GPU:-a6000:1}"
# `debug` is hala-only, 4 h cap.  A GPU *type* is mandatory here -- bare
# --gpus=1 is rejected.
PARTITION="${DEBUG_PARTITION:-debug}"
TIME="${DEBUG_TIME:-2:00:00}"
BACKEND="${DEBUG_BACKEND:-sdpa}"

# `debug` and `rendering` each carry AllowQos=<own name>, so a job that omits
# the matching --qos is REJECTED outright, not just deprioritised.  `batch`
# sets no AllowQos and defaults to `normal`, so it must be left unset there.
# Both of those QoSes have priority 1e6 and preempt normal + every conference
# QoS (REQUEUE, 300 s grace) -- that is the mechanism that gets you a card on
# a full node.
if [ -n "${DEBUG_QOS:-}" ]; then
    QOS="$DEBUG_QOS"
else
    case "$PARTITION" in
        debug)     QOS=debug ;;
        rendering) QOS=rendering ;;
        *)         QOS="" ;;
    esac
fi

QOS_ARG=()
[ -n "$QOS" ] && QOS_ARG=(--qos="$QOS")

SDPA_FLAG=""
[ "$BACKEND" = "sdpa" ] && SDPA_FLAG="--use_sdpa"

# Small run that still crosses every branch: phase 1 is 4 frames, and
# keyframe_interval=2 sends every other phase-2 frame down the
# append -> attend -> rollback path.
DEMO_ARGS=(
    --model_path "$REPO_DIR/ckpt/lingbot-map.pt"
    --image_folder "$REPO_DIR/example/university"
    --first_k 24
    --num_scale_frames 4
    --keyframe_interval 2
)
[ -n "$SDPA_FLAG" ] && DEMO_ARGS+=("$SDPA_FLAG")
DEMO_ARGS+=("$@")

echo "[debug_session] partition=$PARTITION qos=${QOS:-<default>} gpu=$GPU port=$PORT backend=$BACKEND"
[ -n "$QOS" ] && echo "[debug_session] qos=$QOS preempts running normal/conference jobs on this node (requeued, 5 min grace)"
echo "[debug_session] waiting for an allocation ..."

srun \
    --partition="$PARTITION" \
    "${QOS_ARG[@]}" \
    --constraint=zone-sof1 \
    --gpus="$GPU" \
    --cpus-per-task=8 \
    --mem=64G \
    --time="$TIME" \
    --chdir="$REPO_DIR" \
    --pty bash -l -c '
set -euo pipefail
source .agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh

# /scratch is wiped weekly, so treat this as part of setup rather than a
# one-time install.
micromamba run -n lingbot_map python -c "import debugpy" 2>/dev/null \
    || micromamba run -n lingbot_map pip install -q debugpy

# Sitting at a breakpoint reads as an idle GPU to the cluster reaper, which
# will deallocate the job mid-session.  0.1 keeps the reservation well clear
# of the model'"'"'s own footprint.
micromamba run -n lingbot_map python \
    .agents/scratch/insait_cluster_files/gpu_keep_alive.py 0.1 &
KEEP_ALIVE_PID=$!
trap "kill $KEEP_ALIVE_PID 2>/dev/null" EXIT

echo
echo "==================================================================="
echo "  Attach VS Code to:  $(hostname)"
echo "  Port:               '"$PORT"'"
echo "==================================================================="
echo

LINGBOT_DEBUG_KV=1 micromamba run -n lingbot_map python -m debugpy \
    --listen 0.0.0.0:'"$PORT"' --wait-for-client \
    demo.py '"${DEMO_ARGS[*]}"'
'
