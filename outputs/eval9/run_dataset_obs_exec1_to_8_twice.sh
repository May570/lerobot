#!/usr/bin/env bash
set -euo pipefail

LEROBOT_ROOT="/home/admin123/桌面/wjl/lerobot"
EVAL_SCRIPT="$LEROBOT_ROOT/src/lerobot/scripts/lerobot_eval_dyn_mini_dataset_obs.py"
POLICY_PATH="/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu_ep200_1gpu_bs32_nw8_20000/base/checkpoints/020000/pretrained_model"
EPISODE_START_STATES_PATH="/home/admin123/桌面/wjl/lerobot/outputs/eval9/fixed_starts/libero_dyn_mini_dataset_first200_seed1000_b2_ep200.npz"
DATASET_IMAGE_NOISE_ENABLE="true"
DATASET_IMAGE_NOISE_STD="0.05"

LOG_ROOT="$LEROBOT_ROOT/outputs/eval10/batch_logs"
mkdir -p "$LOG_ROOT"

cd "$LEROBOT_ROOT"

run_one() {
  local round_id="$1"
  local exec_steps="$2"
  local log_ts
  log_ts="$(date +%Y%m%d_%H%M%S)"
  local log_path="$LOG_ROOT/round${round_id}_exec${exec_steps}_${log_ts}.log"
  local save_noisy_images="false"
  if [[ "$exec_steps" == "8" ]]; then
    save_noisy_images="true"
  fi

  echo "[$(date '+%F %T')] round=${round_id} exec_steps=${exec_steps} log=${log_path}"

  PYTHONPATH=src \
  python "$EVAL_SCRIPT" \
    --policy.path="$POLICY_PATH" \
    --policy.device=cuda \
    --policy.use_amp=true \
    --env.type=libero \
    --env.task=libero_dyn_mini \
    --env.task_ids=0 \
    --env.episode_start_states_path="$EPISODE_START_STATES_PATH" \
    --env.episode_length=300 \
    --eval.n_episodes=50 \
    --eval.batch_size=2 \
    --rollout.execute_n_action_steps="$exec_steps" \
    --dataset.image_noise.enable="$DATASET_IMAGE_NOISE_ENABLE" \
    --dataset.image_noise.std="$DATASET_IMAGE_NOISE_STD" \
    --dataset.image_noise.save_images.enable="$save_noisy_images" \
    2>&1 | tee "$log_path"
}

for round_id in 1 2; do
  for exec_steps in 8 7 6 5 4 3 2 1; do
    run_one "$round_id" "$exec_steps"
  done
done

echo "[$(date '+%F %T')] all runs finished"
