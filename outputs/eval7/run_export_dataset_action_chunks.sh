#!/usr/bin/env bash

set -euo pipefail

REPO_ROOT="$(cd "$(dirname "${BASH_SOURCE[0]}")/../.." && pwd)"
cd "$REPO_ROOT"

if [ "$#" -lt 1 ]; then
  echo "Usage: bash outputs/eval7/run_export_dataset_action_chunks.sh /path/to/pretrained_model [extra_args...]"
  exit 1
fi

POLICY_PATH="$1"
shift

ENV_NAME="lerobot_py312"
EXPORT_SCRIPT="$REPO_ROOT/src/lerobot/scripts/lerobot_export_dataset_action_chunks.py"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export HF_DATASETS_CACHE="${HF_DATASETS_CACHE:-/tmp/hf-datasets}"
export HUGGINGFACE_HUB_CACHE="${HUGGINGFACE_HUB_CACHE:-/tmp/hf-hub}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config}"
export PYTHONPATH="${PYTHONPATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:$REPO_ROOT/src}"

mkdir -p "$MPLCONFIGDIR" "$HF_DATASETS_CACHE" "$HUGGINGFACE_HUB_CACHE"

COMMON_ARGS=(
  --batch_size=2
  --seed=1000
  --max_episodes=200
  --policy.device=cuda
  --policy.use_amp=true
  --policy.n_action_steps=15
  --policy.num_inference_steps=20
  --policy.future_condition_delta=4
)

conda run --no-capture-output -n "$ENV_NAME" \
  python "$EXPORT_SCRIPT" \
  --policy.path="$POLICY_PATH" \
  "${COMMON_ARGS[@]}" \
  "$@"
