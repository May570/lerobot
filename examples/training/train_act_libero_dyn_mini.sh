#!/usr/bin/env bash
set -euo pipefail

MODEL="${MODEL:-scene_only}"
DATASET_ROOT="${DATASET_ROOT:-/share/project/wujiling/datasets/libero_dyn_mini}"
DATASET_REPO_ID="${DATASET_REPO_ID:-libero_dyn_mini}"
STEPS="${STEPS:-20000}"
BATCH_SIZE="${BATCH_SIZE:-32}"
NUM_WORKERS="${NUM_WORKERS:-8}"
CHUNK_SIZE="${CHUNK_SIZE:-16}"
N_ACTION_STEPS="${N_ACTION_STEPS:-8}"
FUTURE_DELTA="${FUTURE_DELTA:-4}"
DELAY_RANDOM="${DELAY_RANDOM:-false}"
OUTPUT_DIR="${OUTPUT_DIR:-/share/project/wujiling/lerobot/outputs/train/act_libero_dyn_mini_${MODEL}_$(date +%Y%m%d_%H%M%S)}"

cd /share/project/wujiling/lerobot

python -m lerobot.scripts.lerobot_train \
  --dataset.repo_id="${DATASET_REPO_ID}" \
  --dataset.root="${DATASET_ROOT}" \
  --policy.type=act \
  --policy.device=cuda \
  --policy.use_amp=true \
  --policy.n_obs_steps=1 \
  --policy.chunk_size="${CHUNK_SIZE}" \
  --policy.n_action_steps="${N_ACTION_STEPS}" \
  --policy.model="${MODEL}" \
  --policy.future_condition_delta="${FUTURE_DELTA}" \
  --policy.delay_random="${DELAY_RANDOM}" \
  --policy.future_ball_pos_key=observation.ball_pos \
  --policy.future_ball_pos_mlp_dim=8 \
  --batch_size="${BATCH_SIZE}" \
  --num_workers="${NUM_WORKERS}" \
  --steps="${STEPS}" \
  --save_freq=5000 \
  --eval_freq=0 \
  --output_dir="${OUTPUT_DIR}"
