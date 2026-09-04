#!/usr/bin/env bash
set -uo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_cut3r_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/cut3r/bin/python"
dataset pull NRGBD >/dev/null 2>&1 || true
cd "$HOME/lingbot-map/TTT3R"
exec "$P" "$HOME/lingbot-map/.agents/scratch/baselines/smoke_ttt3r.py"
