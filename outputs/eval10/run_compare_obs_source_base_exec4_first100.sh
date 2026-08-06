#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_py312}"
POLICY_PATH="${POLICY_PATH:-/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base/checkpoints/020000/pretrained_model}"
EVAL_SCRIPT="${EVAL_SCRIPT:-${PROJECT_ROOT}/src/lerobot/scripts/lerobot_eval_dyn_mini_obs_source_compare.py}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"
OBSERVATION_SOURCES="${OBSERVATION_SOURCES:-online dataset}"
EXECUTE_STEPS_LIST="${EXECUTE_STEPS_LIST:-4}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-0}"

EPISODE_START_STATES_PATH="${EPISODE_START_STATES_PATH:-/home/admin123/桌面/wjl/lerobot/outputs/eval9/fixed_starts/libero_dyn_mini_dataset_first200_seed1000_b2_ep200.npz}"
DATASET_EPISODES="${DATASET_EPISODES:-0:100}"
N_EPISODES="${N_EPISODES:-100}"
BATCH_SIZE="${BATCH_SIZE:-2}"
EPISODE_LENGTH="${EPISODE_LENGTH:-300}"
POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-}"
ENABLE_IMAGE_NOISE="${ENABLE_IMAGE_NOISE:-false}"
IMAGE_NOISE_STD="${IMAGE_NOISE_STD:-0.0}"

mkdir -p "${OUTPUT_ROOT}"

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

read -r -a OBS_SOURCE_ARRAY <<< "${OBSERVATION_SOURCES}"
read -r -a EXEC_STEPS_ARRAY <<< "${EXECUTE_STEPS_LIST}"

if (( ${#OBS_SOURCE_ARRAY[@]} == 0 )); then
  echo "OBSERVATION_SOURCES resolved to an empty list." >&2
  exit 1
fi

if (( ${#EXEC_STEPS_ARRAY[@]} == 0 )); then
  echo "EXECUTE_STEPS_LIST resolved to an empty list." >&2
  exit 1
fi

TOTAL_RUNS=$(( ${#OBS_SOURCE_ARRAY[@]} * ${#EXEC_STEPS_ARRAY[@]} ))
GLOBAL_RUN_IX=0

echo "============================================================"
echo "[$(date '+%F %T')] starting unified observation-source comparison"
echo "policy_path=${POLICY_PATH}"
echo "eval_script=${EVAL_SCRIPT}"
echo "output_root=${OUTPUT_ROOT}"
echo "observation_sources=${OBSERVATION_SOURCES}"
echo "execute_steps_list=${EXECUTE_STEPS_LIST}"
echo "dataset_episodes=${DATASET_EPISODES}"
echo "n_episodes=${N_EPISODES}"
echo "batch_size=${BATCH_SIZE}"
echo "episode_length=${EPISODE_LENGTH}"
echo "episode_start_states_path=${EPISODE_START_STATES_PATH}"
echo "enable_image_noise=${ENABLE_IMAGE_NOISE}"
echo "image_noise_std=${IMAGE_NOISE_STD}"

for observation_source in "${OBS_SOURCE_ARRAY[@]}"; do
  for exec_steps in "${EXEC_STEPS_ARRAY[@]}"; do
    GLOBAL_RUN_IX=$(( GLOBAL_RUN_IX + 1 ))
    ts="$(date '+%Y%m%d_%H%M%S')"
    run_name="base_${observation_source}_obs_exec${exec_steps}_first100_${ts}"
    run_dir="${OUTPUT_ROOT}/${run_name}"
    log_path="${run_dir}/run.log"

    mkdir -p "${run_dir}"

    echo "------------------------------------------------------------"
    echo "[$(date '+%F %T')] run ${GLOBAL_RUN_IX}/${TOTAL_RUNS} starting"
    echo "observation_source=${observation_source}"
    echo "execute_n_action_steps=${exec_steps}"
    echo "run_dir=${run_dir}"

    cmd=(
      python "${EVAL_SCRIPT}"
      "--policy.path=${POLICY_PATH}"
      "--policy.device=cuda"
      "--policy.use_amp=true"
      "--env.type=libero"
      "--env.task=libero_dyn_mini"
      "--env.task_ids=0"
      "--env.episode_length=${EPISODE_LENGTH}"
      "--env.ball_grasp_eval_mode=strict"
      "--env.ball_grasp_strict_require_pad_contact=true"
      "--env.ball_grasp_strict_lift_multiplier=1.2"
      "--env.ball_grasp_strict_grip_center_max_dist=0.045"
      "--env.episode_start_states_path=${EPISODE_START_STATES_PATH}"
      "--rollout.observation_source=${observation_source}"
      "--rollout.execute_n_action_steps=${exec_steps}"
      "--dataset.episodes=${DATASET_EPISODES}"
      "--eval.n_episodes=${N_EPISODES}"
      "--eval.batch_size=${BATCH_SIZE}"
      "--dataset.image_noise.enable=${ENABLE_IMAGE_NOISE}"
      "--dataset.image_noise.std=${IMAGE_NOISE_STD}"
      "--output_dir=${run_dir}"
    )

    if [[ -n "${POLICY_N_ACTION_STEPS}" ]]; then
      cmd+=("--policy.n_action_steps=${POLICY_N_ACTION_STEPS}")
    fi

    if [[ -n "${POLICY_NUM_INFERENCE_STEPS}" ]]; then
      cmd+=("--policy.num_inference_steps=${POLICY_NUM_INFERENCE_STEPS}")
    fi

    MPLCONFIGDIR=/tmp/mpl \
    PYTHONPATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:/home/admin123/桌面/wjl/lerobot/src \
    LIBERO_CONFIG_PATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config \
    conda run --no-capture-output -n "${CONDA_ENV}" \
    "${cmd[@]}" 2>&1 | tee "${log_path}"

    echo "[$(date '+%F %T')] run ${GLOBAL_RUN_IX}/${TOTAL_RUNS} finished"

    if (( SLEEP_BETWEEN_RUNS > 0 && GLOBAL_RUN_IX < TOTAL_RUNS )); then
      echo "[$(date '+%F %T')] sleeping ${SLEEP_BETWEEN_RUNS}s before next run"
      sleep "${SLEEP_BETWEEN_RUNS}"
    fi
  done
done

echo "============================================================"
echo "[$(date '+%F %T')] all ${TOTAL_RUNS} runs completed"
