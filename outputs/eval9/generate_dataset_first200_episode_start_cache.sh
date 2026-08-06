#!/usr/bin/env bash
set -euo pipefail

LEROBOT_ROOT="/home/admin123/桌面/wjl/lerobot"
LIBERO_ROOT="/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini"

PLAN_SRC="$LIBERO_ROOT/init_files/libero_dyn_mini/rolling_ball_to_bowl.eval_from_dataset_balanced500_scripted_v2_first200.jsonl"
PLAN_COPY="$LEROBOT_ROOT/outputs/eval9/rolling_ball_to_bowl.eval_from_dataset_balanced500_scripted_v2_first200.jsonl"
CACHE_DIR="$LEROBOT_ROOT/outputs/eval9/fixed_starts"
CACHE_PATH="$CACHE_DIR/libero_dyn_mini_dataset_first200_seed1000_b2_ep200.npz"

mkdir -p "$CACHE_DIR"
cp -f "$PLAN_SRC" "$PLAN_COPY"

cd "$LEROBOT_ROOT"

echo "[$(date '+%F %T')] copied dataset-aligned plan to:"
echo "  $PLAN_COPY"

PYTHONPATH=src \
python src/lerobot/scripts/generate_libero_episode_start_cache.py \
  --env.task=libero_dyn_mini \
  --env.task_ids=0 \
  --env.control_mode=relative \
  --env.init_states \
  --env.init_plan_path="$PLAN_COPY" \
  --env.no_init_plan_loop \
  --env.ball_grasp_eval_mode=strict \
  --env.ball_grasp_strict_lift_multiplier=1.2 \
  --env.ball_grasp_strict_grip_center_max_dist=0.045 \
  --env.ball_grasp_strict_require_pad_contact \
  --env.observation_height=360 \
  --env.observation_width=360 \
  --eval.batch_size=2 \
  --eval.n_episodes=200 \
  --seed=1000 \
  --output="$CACHE_PATH"

echo "[$(date '+%F %T')] episode-start cache generated:"
echo "  $CACHE_PATH"
