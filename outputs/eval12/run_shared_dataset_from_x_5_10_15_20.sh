#!/usr/bin/env bash
set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
ROOT_DIR="$(cd "${SCRIPT_DIR}/../.." && pwd)"
PROJECT_ROOT="${ROOT_DIR}"
EVAL_SCRIPT="${SCRIPT_DIR}/paired_obs_source_shared_dataset_from_x.py"
CONDA_ENV="${CONDA_ENV:-lerobot_py312}"
POLICY_PATH="${POLICY_PATH:-/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base/checkpoints/020000/pretrained_model}"
EPISODE_START_STATES_PATH="${EPISODE_START_STATES_PATH:-/home/admin123/桌面/wjl/lerobot/outputs/eval9/fixed_starts/libero_dyn_mini_dataset_first200_seed1000_b2_ep200.npz}"
DATASET_EPISODES="${DATASET_EPISODES:-0:20}"
N_EPISODES="${N_EPISODES:-20}"
BATCH_SIZE="${BATCH_SIZE:-2}"

cd "${ROOT_DIR}"

for X in 5 10 15 20; do
  OUT_DIR="${SCRIPT_DIR}/shared_dataset_from_inf${X}"
  mkdir -p "${OUT_DIR}"

  MPLCONFIGDIR=/tmp/mpl \
  PYTHONPATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:/home/admin123/桌面/wjl/lerobot/src \
  LIBERO_CONFIG_PATH=/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config \
  conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${EVAL_SCRIPT}" \
    --policy.path="${POLICY_PATH}" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --env.type=libero \
    --env.task=libero_dyn_mini \
    --env.task_ids=0 \
    --env.episode_length=300 \
    --env.ball_grasp_eval_mode=strict \
    --env.ball_grasp_strict_require_pad_contact=true \
    --env.ball_grasp_strict_lift_multiplier=1.2 \
    --env.ball_grasp_strict_grip_center_max_dist=0.045 \
    --env.episode_start_states_path="${EPISODE_START_STATES_PATH}" \
    --dataset.episodes="${DATASET_EPISODES}" \
    --eval.n_episodes="${N_EPISODES}" \
    --eval.batch_size="${BATCH_SIZE}" \
    --rollout.execute_n_action_steps=4 \
    --rollout.shared_input_start_inference="${X}" \
    --output_dir="${OUT_DIR}" \
    2>&1 | tee "${OUT_DIR}/run.log"
done
