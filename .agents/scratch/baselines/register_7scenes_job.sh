#!/usr/bin/env bash
set -euo pipefail
source "$HOME/lingbot-map/.agents/scratch/insait_cluster_files/setup_lingbot_map_env.sh"
P="$MAMBA_ROOT_PREFIX/envs/lingbot_map/bin/python"
"$P" -c "import skimage, joblib" 2>/dev/null || "$P" -m pip install scikit-image joblib
exec "$P" "$HOME/lingbot-map/.agents/scratch/baselines/register_7scenes_depth.py"
