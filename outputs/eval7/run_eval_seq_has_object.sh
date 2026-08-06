#!/usr/bin/env bash

set -euo pipefail

ROOT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
WORKSPACE_ROOT="$(cd "${ROOT_DIR}/../.." && pwd)"
cd "$ROOT_DIR"
SCRIPT_ARGS=("$@")

ENV_NAME="lerobot_py312"
EVAL_SCRIPT="${ROOT_DIR}/lerobot_eval_realtime_dyn_mini_has_object.py"

export NUMBA_DISABLE_JIT=1
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config}"
export PYTHONPATH="${PYTHONPATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:${WORKSPACE_ROOT}/src}"

mkdir -p "$MPLCONFIGDIR" outputs/eval7_has_object

COMMON_ARGS=(
  --env.type=libero
  --env.task=libero_dyn_mini
  --eval.batch_size=2
  --env.episode_length=260
  --policy.device=cuda
  --policy.use_amp=true
  --policy.n_action_steps=15
  --policy.num_inference_steps=20
  --policy.future_condition_delta=4
  --env.ball_grasp_eval_mode=strict
  --env.ball_grasp_strict_require_pad_contact=true
  --env.ball_grasp_strict_lift_multiplier=1.2
  --env.ball_grasp_strict_grip_center_max_dist=0.045
)

# Optional: point this at a pre-generated fixed episode-start cache so every run replays
# the exact same initial simulator states instead of relying on reset-time randomization.
FIXED_EPISODE_STARTS_PATH="${FIXED_EPISODE_STARTS_PATH:-}"
if [ -n "$FIXED_EPISODE_STARTS_PATH" ]; then
  COMMON_ARGS+=(--env.episode_start_states_path="$FIXED_EPISODE_STARTS_PATH")
fi

run_one() {
  local tag="$1"
  local policy_path="$2"
  shift 2

  local ts output_dir log_path
  ts="$(date +%Y%m%d_%H%M%S)"
  output_dir="outputs/eval7_has_object/${tag}_${ts}"
  log_path="${output_dir}.log"

  echo "============================================================"
  echo "[$(date '+%F %T')] start: ${tag}"
  echo "policy_path=${policy_path}"
  echo "output_dir=${output_dir}"
  echo "log_path=${log_path}"
  if [ "$#" -gt 0 ]; then
    echo "extra_args=$*"
  fi

  conda run --no-capture-output -n "$ENV_NAME" \
    python "$EVAL_SCRIPT" \
    --policy.path="$policy_path" \
    --output_dir="$output_dir" \
    "${COMMON_ARGS[@]}" \
    "${SCRIPT_ARGS[@]}" \
    "$@" 2>&1 | tee "$log_path"

  echo "[$(date '+%F %T')] done: ${tag}"
}

# 顺序执行。这里默认挂上目前已训练好的 use_labels / has_object 权重。
run_one \
  "base_use_labels" \
  "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base_use_labels/checkpoints/020000/pretrained_model"

run_one \
  "scene_only_random_delay_use_labels" \
  "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/scene_only_random_delay_use_labels/checkpoints/020000/pretrained_model"

run_one \
  "scene_only_nogate_random_delay_use_labels" \
  "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/scene_only_nogate_random_delay_use_labels/checkpoints/020000/pretrained_model"

run_one \
  "scene_only_kalman_future_random_delay_use_labels" \
  "/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/scene_only_kalman_future_random_delay_use_labels/checkpoints/020000/pretrained_model"

# 这个任务单独多带一个参数的写法
# run_one \
#   "robot_scene_future_delta_2" \
#   "/path/to/your/checkpoint/pretrained_model" \
#   --policy.future_condition_delta=2

# 还可以继续加别的单独参数
# run_one \
#   "robot_scene_future_delta_4_seed123" \
#   "/path/to/your/checkpoint/pretrained_model" \
#   --policy.future_condition_delta=4 \
#   --seed=123
