#!/usr/bin/env bash

# Stop one training launcher and every child it started, including ZK processes.
# Usage:
#   bash scripts/stop_train_mappo_4v4_linux.sh
#   bash scripts/stop_train_mappo_4v4_linux.sh /path/to/.mappo_train.pid

set -uo pipefail

SCRIPT_DIR="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
PROJECT_ROOT="$(cd "${SCRIPT_DIR}/.." && pwd)"
PID_FILE="${1:-${OUTPUT_ROOT:-${PROJECT_ROOT}/results}/.mappo_train.pid}"

if [[ ! -f "${PID_FILE}" ]]; then
    echo "未找到训练 PID 文件：${PID_FILE}"
    echo "训练可能已经结束，或启动脚本不是 train_mappo_4v4_linux.sh。"
    exit 0
fi

wrapper_pid="$(sed -n 's/^wrapper_pid=//p' "${PID_FILE}" | tail -n 1)"
temp_log="$(sed -n 's/^temp_log=//p' "${PID_FILE}" | tail -n 1)"

if [[ ! "${wrapper_pid}" =~ ^[0-9]+$ ]]; then
    echo "PID 文件格式无效：${PID_FILE}" >&2
    exit 1
fi

descendants() {
    local parent="$1"
    local child
    for child in $(ps -eo pid=,ppid= | awk -v parent="${parent}" '$2 == parent {print $1}'); do
        descendants "${child}"
    done
    echo "${parent}"
}

if ! kill -0 "${wrapper_pid}" 2>/dev/null; then
    echo "训练主进程 ${wrapper_pid} 已经不存在，清理 PID 文件。"
    rm -f "${PID_FILE}"
    exit 0
fi

mapfile -t pids < <(descendants "${wrapper_pid}" | sort -rn -u)
echo "正在终止训练进程树：${pids[*]}"

for pid in "${pids[@]}"; do
    kill -TERM "${pid}" 2>/dev/null || true
done

for _ in 1 2 3 4 5 6 7 8 9 10; do
    alive=0
    for pid in "${pids[@]}"; do
        if kill -0 "${pid}" 2>/dev/null; then
            alive=1
            break
        fi
    done
    [[ "${alive}" -eq 0 ]] && break
    sleep 1
done

for pid in "${pids[@]}"; do
    kill -KILL "${pid}" 2>/dev/null || true
done

# If SIGKILL prevented the launcher's EXIT trap, preserve the console log too.
if [[ -n "${temp_log}" && -f "${temp_log}" ]]; then
    run_dir="$(sed -n 's/^训练输出目录：//p' "${temp_log}" | tail -n 1)"
    if [[ -n "${run_dir}" ]]; then
        mkdir -p "${run_dir}/log"
        cp "${temp_log}" "${run_dir}/log/train_console.log"
        echo "控制台日志：${run_dir}/log/train_console.log"
    fi
fi

rm -f "${PID_FILE}"
echo "训练进程及其子进程已终止。"
