#!/bin/bash

# Kill InstructDiffusion training runs only if owned by intern4 user
set -euo pipefail

BASE_DIR="$(cd "$(dirname "$0")" && pwd)"
MAIN_PATH="${BASE_DIR}/main.py"
timeout_seconds=20

echo "[INFO] Looking for running main.py processes (torchrun/python) under ${BASE_DIR}"

# Collect candidate PIDs: torchrun launcher + worker processes referencing this main.py
PIDS=$( (
    pgrep -f "torchrun.*${MAIN_PATH}" || true
    pgrep -f "torchrun.*main.py" || true
    pgrep -f "python.*${MAIN_PATH}" || true
    pgrep -f "python.*main.py" || true
) | sort -u )

if [ -z "${PIDS}" ]; then
    echo "[INFO] No main.py processes found."
    exit 0
fi

echo "[INFO] Found main.py-related PIDs: ${PIDS}"

for pid in ${PIDS}; do
    # Check if process owner is intern4
    OWNER=$(ps -o user= -p "${pid}" 2>/dev/null | tr -d ' ')
    
    if [ "${OWNER}" != "intern4" ]; then
        echo "[INFO] Skipping PID ${pid} - owned by ${OWNER}, not intern4"
        continue
    fi
    
    # Use process group to ensure all worker ranks get the signal.
    pgid=$(ps -o pgid= -p "${pid}" | tr -d ' ')
    target="-${pgid}"

    echo "[INFO] Killing PID ${pid} (owned by ${OWNER}); PGID ${pgid}"
    kill -15 "${target}" || true

    i=0
    while kill -0 "${pid}" 2>/dev/null; do
        if [ "${i}" -ge "${timeout_seconds}" ]; then
            echo "[WARN] PGID ${pgid} still alive after ${timeout_seconds}s. Sending SIGKILL."
            kill -9 "${target}" || true
            break
        fi
        sleep 1
        i=$((i + 1))
    done

    if ! kill -0 "${pid}" 2>/dev/null; then
        echo "[INFO] PID ${pid}/PGID ${pgid} has exited."
    fi
done

echo "[INFO] All main.py process groups have been terminated."
