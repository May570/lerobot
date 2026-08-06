#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/../.." && pwd)"

CONDA_ENV="${CONDA_ENV:-lerobot_py312}"
EVAL_SCRIPT="${EVAL_SCRIPT:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/scripts/lerobot_eval_realtime_dyn_mini.py}"
POLICY_PATH="${POLICY_PATH:-/media/admin123/f2745ce8-5417-4e98-a1df-b274a4ca83e8/home/apj/Desktop/1gpu/dun_mini_pi05_0state/checkpoints/020000/pretrained_model}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${SCRIPT_DIR}}"
RUN_TAG="${RUN_TAG:-pi05_realtime}"

EVAL_BATCH_SIZE="${EVAL_BATCH_SIZE:-1}"
N_EPISODES="${N_EPISODES:-100}"
EPISODE_LENGTH="${EPISODE_LENGTH:-260}"
POLICY_DEVICE="${POLICY_DEVICE:-cuda}"
POLICY_USE_AMP="${POLICY_USE_AMP:-false}"
# POLICY_N_ACTION_STEPS="${POLICY_N_ACTION_STEPS:-20}"
POLICY_NUM_INFERENCE_STEPS="${POLICY_NUM_INFERENCE_STEPS:-10}"
TOKENIZER_PATH="${TOKENIZER_PATH:-/home/admin123/桌面/offline_orange_test/google/paligemma-3b-pt-224}"

ENV_TASK="${ENV_TASK:-libero_dyn_mini}"
BALL_GRASP_EVAL_MODE="${BALL_GRASP_EVAL_MODE:-strict}"
BALL_GRASP_REQUIRE_PAD_CONTACT="${BALL_GRASP_REQUIRE_PAD_CONTACT:-true}"
BALL_GRASP_LIFT_MULTIPLIER="${BALL_GRASP_LIFT_MULTIPLIER:-1.2}"
BALL_GRASP_GRIP_CENTER_MAX_DIST="${BALL_GRASP_GRIP_CENTER_MAX_DIST:-0.045}"
EPISODE_START_STATES_PATH="${EPISODE_START_STATES_PATH:-}"

mkdir -p "${OUTPUT_ROOT}"
mkdir -p "${MPLCONFIGDIR:-/tmp/mpl}"

export NUMBA_DISABLE_JIT="${NUMBA_DISABLE_JIT:-1}"
export MPLCONFIGDIR="${MPLCONFIGDIR:-/tmp/mpl}"
export LIBERO_CONFIG_PATH="${LIBERO_CONFIG_PATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/config}"
export PYTHONPATH="${PYTHONPATH:-/home/admin123/桌面/wjl/LIBERO/libero_dyn_mini/py:${PROJECT_ROOT}/src}"

if [[ ! -f "${EVAL_SCRIPT}" ]]; then
  echo "Eval script not found: ${EVAL_SCRIPT}" >&2
  exit 1
fi

if [[ ! -f "${POLICY_PATH}/config.json" ]]; then
  echo "Policy checkpoint looks invalid, missing config.json under: ${POLICY_PATH}" >&2
  exit 1
fi

ts="$(date '+%Y%m%d_%H%M%S')"
run_name="${RUN_TAG}_${ts}"
run_dir="${OUTPUT_ROOT}/${run_name}"
log_path="${run_dir}/run.log"
mkdir -p "${run_dir}"

EFFECTIVE_POLICY_PATH="${POLICY_PATH}"

if [[ ! -d "${TOKENIZER_PATH}" ]]; then
  echo "Tokenizer directory not found: ${TOKENIZER_PATH}" >&2
  exit 1
fi

if [[ -f "${POLICY_PATH}/policy_preprocessor.json" ]] && \
   grep -q '"/share/project/wujiling/checkpoints/tokenizers/paligemma-3b-pt-224"' "${POLICY_PATH}/policy_preprocessor.json"; then
  EFFECTIVE_POLICY_PATH="${run_dir}/policy_override"
  mkdir -p "${EFFECTIVE_POLICY_PATH}"

  SOURCE_PREPROCESSOR_JSON="${POLICY_PATH}/policy_preprocessor.json" \
  TARGET_PREPROCESSOR_JSON="${EFFECTIVE_POLICY_PATH}/policy_preprocessor.json" \
  TOKENIZER_DIR="${TOKENIZER_PATH}" \
  python - <<'PY'
import json
import os

source = os.environ["SOURCE_PREPROCESSOR_JSON"]
target = os.environ["TARGET_PREPROCESSOR_JSON"]
tokenizer_dir = os.environ["TOKENIZER_DIR"]

with open(source, "r", encoding="utf-8") as f:
    config = json.load(f)

patched = False
for step in config.get("steps", []):
    if step.get("registry_name") == "tokenizer_processor":
        step.setdefault("config", {})["tokenizer_name"] = tokenizer_dir
        patched = True

if not patched:
    raise SystemExit("tokenizer_processor step not found in policy_preprocessor.json")

with open(target, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

  SOURCE_CONFIG_JSON="${POLICY_PATH}/config.json" \
  TARGET_CONFIG_JSON="${EFFECTIVE_POLICY_PATH}/config.json" \
  python - <<'PY'
import json
import os

source = os.environ["SOURCE_CONFIG_JSON"]
target = os.environ["TARGET_CONFIG_JSON"]

with open(source, "r", encoding="utf-8") as f:
    config = json.load(f)

# The realtime eval path stacks image tensors before PI05's own missing-camera handling runs.
# Disable synthetic empty cameras here so rollout only expects real env image keys.
config["empty_cameras"] = 0
if "input_features" in config:
    input_features = dict(config["input_features"])
    input_features.pop("observation.images.empty_camera_0", None)
    config["input_features"] = input_features

with open(target, "w", encoding="utf-8") as f:
    json.dump(config, f, ensure_ascii=False, indent=2)
    f.write("\n")
PY

  entry_names=(
    model.safetensors
    policy_postprocessor.json
    policy_postprocessor_step_0_unnormalizer_processor.safetensors
    policy_preprocessor_step_2_normalizer_processor.safetensors
    train_config.json
  )

  for entry_name in "${entry_names[@]}"; do
    if [[ -e "${POLICY_PATH}/${entry_name}" ]]; then
      ln -s "${POLICY_PATH}/${entry_name}" "${EFFECTIVE_POLICY_PATH}/${entry_name}"
    fi
  done
fi

ARGS=(
  --policy.path="${EFFECTIVE_POLICY_PATH}"
  --env.type=libero
  --env.task="${ENV_TASK}"
  --eval.n_episodes="${N_EPISODES}"
  --eval.batch_size="${EVAL_BATCH_SIZE}"
  --env.episode_length="${EPISODE_LENGTH}"
  --policy.device="${POLICY_DEVICE}"
  --policy.use_amp="${POLICY_USE_AMP}"
  # --policy.n_action_steps="${POLICY_N_ACTION_STEPS}"
  --policy.num_inference_steps="${POLICY_NUM_INFERENCE_STEPS}"
  --env.ball_grasp_eval_mode="${BALL_GRASP_EVAL_MODE}"
  --env.ball_grasp_strict_require_pad_contact="${BALL_GRASP_REQUIRE_PAD_CONTACT}"
  --env.ball_grasp_strict_lift_multiplier="${BALL_GRASP_LIFT_MULTIPLIER}"
  --env.ball_grasp_strict_grip_center_max_dist="${BALL_GRASP_GRIP_CENTER_MAX_DIST}"
  --output_dir="${run_dir}"
)

if [[ -n "${EPISODE_START_STATES_PATH}" ]]; then
  if [[ ! -f "${EPISODE_START_STATES_PATH}" ]]; then
    echo "Episode start states file not found: ${EPISODE_START_STATES_PATH}" >&2
    exit 1
  fi
  ARGS+=(--env.episode_start_states_path="${EPISODE_START_STATES_PATH}")
fi

echo "============================================================"
echo "[$(date '+%F %T')] start realtime pi eval"
echo "policy_path=${POLICY_PATH}"
if [[ "${EFFECTIVE_POLICY_PATH}" != "${POLICY_PATH}" ]]; then
  echo "effective_policy_path=${EFFECTIVE_POLICY_PATH}"
fi
echo "tokenizer_path=${TOKENIZER_PATH}"
echo "eval_script=${EVAL_SCRIPT}"
echo "output_dir=${run_dir}"
echo "log_path=${log_path}"
echo "n_episodes=${N_EPISODES}"
echo "eval_batch_size=${EVAL_BATCH_SIZE}"
echo "policy_device=${POLICY_DEVICE}"
echo "policy_use_amp=${POLICY_USE_AMP}"
# echo "policy_n_action_steps=${POLICY_N_ACTION_STEPS}"
echo "policy_num_inference_steps=${POLICY_NUM_INFERENCE_STEPS}"
if [[ -n "${EPISODE_START_STATES_PATH}" ]]; then
  echo "episode_start_states_path=${EPISODE_START_STATES_PATH}"
fi

conda run --no-capture-output -n "${CONDA_ENV}" \
  python "${EVAL_SCRIPT}" \
  "${ARGS[@]}" 2>&1 | tee "${log_path}"

echo "[$(date '+%F %T')] done realtime pi eval"
