#!/usr/bin/env bash

# Linux MAPPO launcher with a persistent console log.
# Usage:
#   bash scripts/train_mappo_4v4_linux.sh
#   PYTHON_BIN=/opt/conda/bin/python bash scripts/train_mappo_4v4_linux.sh --updates 10

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PYTHON_BIN="${PYTHON_BIN:-python3}"
OUTPUT_ROOT="${OUTPUT_ROOT:-${PROJECT_ROOT}/results}"
RUNS_ROOT="${OUTPUT_ROOT}/4v4/mappo"
STAMP="$(date -u +%Y%m%dT%H%M%SZ)"
TEMP_LOG="${OUTPUT_ROOT}/.train_console_${STAMP}.log"
PID_FILE="${OUTPUT_ROOT}/.mappo_train.pid"

mkdir -p "${OUTPUT_ROOT}"
cd "${PROJECT_ROOT}"

if [[ -f "${PID_FILE}" ]]; then
    old_wrapper="$(sed -n 's/^wrapper_pid=//p' "${PID_FILE}" | tail -n 1)"
    if [[ -n "${old_wrapper}" ]] && kill -0 "${old_wrapper}" 2>/dev/null; then
        echo "已有训练进程运行中，PID=${old_wrapper}。请先运行 stop_train_mappo_4v4_linux.sh。" >&2
        exit 1
    fi
    rm -f "${PID_FILE}"
fi

cleanup() {
    status=$?
    run_dir=""

    # CompetitionRunner prints the exact newly-created run directory.
    if [[ -f "${TEMP_LOG}" ]]; then
        run_dir="$(sed -n 's/^训练输出目录：//p' "${TEMP_LOG}" | tail -n 1)"
    fi

    # Fallback for an early failure before the runner prints its directory.
    if [[ -z "${run_dir}" && -d "${RUNS_ROOT}" ]]; then
        run_dir="$(find "${RUNS_ROOT}" -mindepth 1 -maxdepth 1 -type d -name 'run-*' -printf '%T@ %p\n' 2>/dev/null \
            | sort -nr | head -n 1 | cut -d' ' -f2-)"
    fi

    if [[ -n "${run_dir}" ]]; then
        mkdir -p "${run_dir}/log"
        cp "${TEMP_LOG}" "${run_dir}/log/train_console.log"
        echo "控制台日志：${run_dir}/log/train_console.log"
    else
        echo "未找到本轮 run 目录，临时控制台日志保留在：${TEMP_LOG}" >&2
    fi

    rm -f "${PID_FILE}"

    exit "${status}"
}

trap cleanup EXIT

echo "Linux MAPPO 训练开始：$(date -Is)"
echo "Python：${PYTHON_BIN}"
echo "项目：${PROJECT_ROOT}"

echo "wrapper_pid=$$" > "${PID_FILE}"
echo "temp_log=${TEMP_LOG}" >> "${PID_FILE}"

( PYTHONUNBUFFERED=1 "${PYTHON_BIN}" -u scripts/train_mappo_4v4.py \
      --output "${OUTPUT_ROOT}" \
      "$@" 2>&1 | tee "${TEMP_LOG}" ) &
PIPELINE_PID=$!
echo "pipeline_pid=${PIPELINE_PID}" >> "${PID_FILE}"

wait "${PIPELINE_PID}"
exit "$?"
