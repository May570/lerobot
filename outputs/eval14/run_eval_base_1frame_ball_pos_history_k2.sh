#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_py312}"
POLICY_PATH="${POLICY_PATH:-/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu/base_1frame_ball_pos_history_k2/checkpoints/020000/pretrained_model}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${SCRIPT_DIR}/lerobot_eval_dyn_mini_sync_ballpos.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"
RUN_TAG="${RUN_TAG:-base_1frame_ball_pos_history_k2}"
RUNS_PER_ACTION_STEP="${RUNS_PER_ACTION_STEP:-1}"
ACTION_STEPS_LIST="${ACTION_STEPS_LIST:-8}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-0}"

N_EPISODES="${N_EPISODES:-100}"
EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-260}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-true}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-20}"

EPISODE_START_STATES_PATH="${EPISODE_START_STATES_PATH:-/home/admin123/桌面/wjl/lerobot/outputs/eval7/fixed_starts/libero_dyn_mini_task0_seed1000_b2_ep100.npz}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${MPLCONFIGDIR:-/tmp/mpl}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Policy checkpoint looks invalid, missing config.json under: ${POLICY_PATH}" >&2
  exit 1
fi

if [[ ! -f "${EPISODE_START_STATES_PATH}" ]]; then
  echo "Episode start states file not found: ${EPISODE_START_STATES_PATH}" >&2
  exit 1
fi

if ! [[ "${RUNS_PER_ACTION_STEP}" =~ ^[0-9]+$ ]] || (( RUNS_PER_ACTION_STEP <= 0 )); then
  echo "RUNS_PER_ACTION_STEP must be a positive integer, got: ${RUNS_PER_ACTION_STEP}" >&2
  exit 1
fi

read -r -a ACTION_STEPS_ARRAY <<< "${ACTION_STEPS_LIST}"
if (( ${#ACTION_STEPS_ARRAY[@]} == 0 )); then
  echo "ACTION_STEPS_LIST resolved to an empty list." >&2
  exit 1
fi

TOTAL_RUNS=$(( ${#ACTION_STEPS_ARRAY[@]} * RUNS_PER_ACTION_STEP ))
GLOBAL_RUN_IX=0

echo "============================================================"
echo "[$(date '+%F %T')] starting eval sweep"
echo "policy_path=${POLICY_PATH}"
echo "eval_script=${EVAL_SCRIPT}"
echo "output_root=${OUTPUT_ROOT}"
echo "run_tag=${RUN_TAG}"
echo "action_steps_list=${ACTION_STEPS_LIST}"
echo "runs_per_action_step=${RUNS_PER_ACTION_STEP}"
echo "total_runs=${TOTAL_RUNS}"
echo "n_episodes=${N_EPISODES}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "episode_length=${EPISODE_LENGTH}"
echo "episode_start_states_path=${EPISODE_START_STATES_PATH}"

for action_steps in "${ACTION_STEPS_ARRAY[@]}"; do
  if ! [[ "${action_steps}" =~ ^[0-9]+$ ]] || (( action_steps <= 0 )); then
    echo "Invalid action_steps value: ${action_steps}" >&2
    exit 1
  fi

  for ((run_ix = 1; run_ix <= RUNS_PER_ACTION_STEP; run_ix++)); do
    GLOBAL_RUN_IX=$(( GLOBAL_RUN_IX + 1 ))
    ts="$(date '+%Y%m%d_%H%M%S')"
    run_name="${RUN_TAG}_actionsteps${action_steps}_${ts}_run${run_ix}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    log_path="${run_dir}/run.log"

    mkdir -p "${run_dir}"

    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] run ${GLOBAL_RUN_IX}/${TOTAL_RUNS} starting"
    echo "action_steps=${action_steps}"
    echo "repeat_index=${run_ix}/${RUNS_PER_ACTION_STEP}"
    echo "run_dir=${run_dir}"

    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:/home/admin123/桌面/wjl/lerobot/src \
    LIBERO_CONFIG_PATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    python "${EVAL_SCRIPT}" \
      --policy.path="${POLICY_PATH}" \
      --env.type=libero \
      --env.task=libero_dyn_mini \
      --eval.n_episodes="${N_EPISODES}" \
      --eval.batch_size="${EVAL_BATCH_SIZE}" \
      --policy.device="${POLICY_DEVICE}" \
      --policy.use_amp="${POLICY_USE_AMP}" \
      --policy.n_action_steps="${action_steps}" \
      --policy.num_inference_steps="${POLICY_NUM_INFERENCE_STEPS}" \
      --env.episode_length="${EPISODE_LENGTH}" \
      --env.ball_grasp_eval_mode=strict \
      --env.ball_grasp_strict_require_pad_contact=true \
      --env.ball_grasp_strict_lift_multiplier=1.2 \
      --env.ball_grasp_strict_grip_center_max_dist=0.045 \
      --env.episode_start_states_path="${EPISODE_START_STATES_PATH}" \
      --output_dir="${run_dir}" \
      2>&1 | tee "${log_path}"

    echo "[$(date '+%F %T')] run ${GLOBAL_RUN_IX}/${TOTAL_RUNS} finished"

    if (( SLEEP_BETWEEN_RUNS > 0 && GLOBAL_RUN_IX < TOTAL_RUNS )); then
      echo "[$(date '+%F %T')] sleeping ${SLEEP_BETWEEN_RUNS}s before next run"
      sleep "${SLEEP_BETWEEN_RUNS}"
    fi
  done
done

echo "============================================================"
echo "[$(date '+%F %T')] all ${TOTAL_RUNS} runs completed"
