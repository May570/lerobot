#!/usr/bin/env bash

set -euo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
TARGET_SCRIPT="${SCRIPT_DIR}/run_eval_seq_future_mask.sh"
RUNS="${RUNS:-3}"
SLEEP_BETWEEN_RUNS="${SLEEP_BETWEEN_RUNS:-0}"
CONTINUE_ON_ERROR="${CONTINUE_ON_ERROR:-0}"

if [[ ! -f "${TARGET_SCRIPT}" ]]; then
  echo "Target script not found: ${TARGET_SCRIPT}" >&2
  exit 1
fi

if ! [[ "${RUNS}" =~ ^[0-9]+$ ]] || (( RUNS <= 0 )); then
  echo "RUNS must be a positive integer, got: ${RUNS}" >&2
  exit 1
fi

for ((run_ix = 1; run_ix <= RUNS; run_ix++)); do
  echo "============================================================"
  echo "[$(date '+%F %T')] repeat run ${run_ix}/${RUNS} starting"
  echo "target_script=${TARGET_SCRIPT}"
  if [[ -n "${FIXED_EPISODE_STARTS_PATH:-}" ]]; then
    echo "FIXED_EPISODE_STARTS_PATH=${FIXED_EPISODE_STARTS_PATH}"
  fi

  if bash "${TARGET_SCRIPT}" "$@"; then
    echo "[$(date '+%F %T')] repeat run ${run_ix}/${RUNS} finished successfully"
  else
    status=$?
    echo "[$(date '+%F %T')] repeat run ${run_ix}/${RUNS} failed with exit code ${status}" >&2
    if [[ "${CONTINUE_ON_ERROR}" != "1" ]]; then
      exit "${status}"
    fi
  fi

  if (( run_ix < RUNS && SLEEP_BETWEEN_RUNS > 0 )); then
    echo "[$(date '+%F %T')] sleeping ${SLEEP_BETWEEN_RUNS}s before next run"
    sleep "${SLEEP_BETWEEN_RUNS}"
  fi
done

echo "[$(date '+%F %T')] all ${RUNS} runs completed"
