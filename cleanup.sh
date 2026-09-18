#!/usr/bin/env bash
set -euo pipefail

export PATH="$HOME/.local/bin:$HOME/.cargo/bin:/usr/local/bin:/usr/bin:/bin:$PATH"

SCRIPT_DIR="$(cd "$(dirname "$0")" && pwd)"
YESTERDAY=$(date -d "yesterday" +%Y-%m-%d)
TODAY=$(date +%Y-%m-%d)

SRC_DIR="/mnt/disk3/containers/motioneye/var/Camera1/${YESTERDAY}"
# OpenVINO IR directory exported from yolo26n.pt (see README: clearcam-convert)
MODEL="yolo26n_openvino_model"
CONF="0.3"
SAMPLE_INTERVAL="1.0"

LOG_FILE="${SCRIPT_DIR}/${TODAY}.log"
LOCK_FILE="${SCRIPT_DIR}/cleanup.pid"

if [ -f "$LOCK_FILE" ]; then
    OLD_PID=$(cat "$LOCK_FILE")
    if kill -0 "$OLD_PID" 2>/dev/null; then
        echo "[$(date)] Script already running with PID $OLD_PID, exiting" >> "$LOG_FILE"
        exit 1
    else
        echo "[$(date)] Removing stale lock file (PID $OLD_PID)" >> "$LOG_FILE"
        rm -f "$LOCK_FILE"
    fi
fi

echo $$ > "$LOCK_FILE"
trap 'rm -f "$LOCK_FILE"' EXIT

if [ ! -d "$SRC_DIR" ]; then
    echo "[$(date)] No folder found: $SRC_DIR" >> "$LOG_FILE"
    exit 0
fi

echo "[$(date)] Processing: $SRC_DIR" >> "$LOG_FILE"
cd "$SCRIPT_DIR"
if [ ! -d ".venv" ]; then
    echo "[$(date)] Virtualenv not found, running uv sync..." >> "$LOG_FILE"
    uv sync >> "$LOG_FILE" 2>&1 || { echo "[$(date)] uv sync failed" >> "$LOG_FILE"; exit 1; }
fi
# Annotate detected videos in place (motionEye keeps listing them) and delete
# videos with no detected objects to free space. Nothing is moved.
uv run clearcam "$SRC_DIR" \
    --annotate \
    --delete-no-object \
    --model "$MODEL" \
    --sample-interval "$SAMPLE_INTERVAL" \
    --conf "$CONF" \
    >> "$LOG_FILE" 2>&1
echo "[$(date)] Done." >> "$LOG_FILE"

find "$SCRIPT_DIR" -maxdepth 1 -name "*.log" -type f -mtime +30 -delete
