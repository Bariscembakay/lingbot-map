#!/usr/bin/env bash
# Runs prepare/run/evaluate for any benchmark/configs/<name>.yaml. Not
# dataset-specific. Run as the command passed to a submit_sof1.sh/
# submit_msp3.sh cluster submitter (assumes envs already set up).
#
# Usage: bash reproduction/run_benchmark.sh <config> [stages...]
#   stages defaults to: prepare run evaluate
set -euo pipefail

CONFIG="$1"
shift
STAGES=("${@:-prepare run evaluate}")
[ "${#STAGES[@]}" -eq 1 ] && STAGES=(${STAGES[0]})

REPO_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")/../../.." && pwd)"
cd "$REPO_DIR/benchmark"

for stage in "${STAGES[@]}"; do
    echo "=== $stage.py ($CONFIG) ==="
    micromamba run -n bench python "$stage.py" --config "$CONFIG"
done
