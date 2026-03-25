#!/usr/bin/env bash
set -euo pipefail

# Reusable LIBERO rollout evaluation helper for LeRobot diffusion checkpoints.
# This script is designed to avoid disturbing ongoing GPU training by default
# (policy runs on CPU unless explicitly overridden).

ROOT_DIR="/share/project/wujiling"
LEROBOT_DIR="${ROOT_DIR}/lerobot"
LEROBOT_PY="${ROOT_DIR}/envs/lerobot/bin/lerobot-eval"

# Default checkpoint (can be overridden by first positional argument).
DEFAULT_POLICY_PATH="${LEROBOT_DIR}/outputs/train/diffusion_libero_flow_gpu/checkpoints/015000/pretrained_model"
POLICY_PATH="${1:-${DEFAULT_POLICY_PATH}}"

# Optional backend selector:
#   osmesa    -> pure headless software rendering
#   glx_xvfb  -> virtual X server + GLX rendering
BACKEND="${2:-osmesa}"

# Optional evaluation size (small default for smoke checks).
N_EPISODES="${3:-1}"
BATCH_SIZE="${4:-1}"

# Optional overrides via environment variables:
#   TASK_IDS="[0]"            -> evaluate a subset of task ids
#   EPISODE_LENGTH=80         -> shorten rollout horizon for quick video checks
#   POLICY_DEVICE="cpu|cuda"  -> inference device (default cpu to avoid training interference)
TASK_IDS="${TASK_IDS:-}"
EPISODE_LENGTH="${EPISODE_LENGTH:-}"
POLICY_DEVICE="${POLICY_DEVICE:-cpu}"

if [[ ! -d "${POLICY_PATH}" ]]; then
  echo "[ERROR] policy path not found: ${POLICY_PATH}" >&2
  exit 1
fi

TIMESTAMP="$(date -u +%Y-%m-%d_%H-%M-%S)"
OUT_DIR="${LEROBOT_DIR}/outputs/eval/libero_rollout_${BACKEND}_${TIMESTAMP}"

# Keep runtime paths explicit so LIBERO and HF caches don't fall back to /root.
export LIBERO_CONFIG_PATH="${ROOT_DIR}/.libero"
export HOME="${ROOT_DIR}"
export HF_HOME="${ROOT_DIR}/.cache/huggingface"
export HF_DATASETS_CACHE="${ROOT_DIR}/.cache/huggingface/datasets"
export LD_LIBRARY_PATH="/lib/x86_64-linux-gnu:${ROOT_DIR}/envs/lerobot/lib:${LD_LIBRARY_PATH:-}"
export NUMBA_DISABLE_JIT=1

BASE_CMD=(
  "${LEROBOT_PY}"
  "--policy.path=${POLICY_PATH}"
  "--env.type=libero"
  "--env.task=libero_10"
  "--eval.n_episodes=${N_EPISODES}"
  "--eval.batch_size=${BATCH_SIZE}"
  "--policy.device=${POLICY_DEVICE}"
  "--policy.use_amp=false"
  "--libero_legacy_obs_compat=true"
  "--output_dir=${OUT_DIR}"
)

if [[ -n "${TASK_IDS}" ]]; then
  BASE_CMD+=("--env.task_ids=${TASK_IDS}")
fi

if [[ -n "${EPISODE_LENGTH}" ]]; then
  BASE_CMD+=("--env.episode_length=${EPISODE_LENGTH}")
fi

echo "[INFO] backend       : ${BACKEND}"
echo "[INFO] policy_path   : ${POLICY_PATH}"
echo "[INFO] n_episodes    : ${N_EPISODES}"
echo "[INFO] batch_size    : ${BATCH_SIZE}"
echo "[INFO] policy_device : ${POLICY_DEVICE}"
echo "[INFO] task_ids      : ${TASK_IDS:-<all>}"
echo "[INFO] episode_len   : ${EPISODE_LENGTH:-<default>}"
echo "[INFO] output_dir    : ${OUT_DIR}"

if [[ "${BACKEND}" == "osmesa" ]]; then
  export MUJOCO_GL="osmesa"
  "${BASE_CMD[@]}"
elif [[ "${BACKEND}" == "glx_xvfb" ]]; then
  export MUJOCO_GL="glx"
  xvfb-run -s "-screen 0 1280x1024x24" "${BASE_CMD[@]}"
else
  echo "[ERROR] unsupported backend: ${BACKEND}" >&2
  echo "        supported: osmesa | glx_xvfb" >&2
  exit 2
fi
