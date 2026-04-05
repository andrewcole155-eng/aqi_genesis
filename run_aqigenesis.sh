#!/bin/bash

# --- Configuration ---
SCRIPT_DIR="/home/andrew/.ssh/Trading/AQI_Genesis"
LOG_DIR="/home/andrew/.ssh/Trading/AQI_Genesis/Logs"
PYTHON_EXEC="/home/andrew/miniconda3/envs/tf_noavx/bin/python"

# --- Ensure log directory exists ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- 1. Run AQI Genesis Data Engine (The Backend) ---
echo "Starting AQI Genesis Pulse script at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
# This fetches data, runs the math, updates the SQLite DB, and sends emails
"$PYTHON_EXEC" "$SCRIPT_DIR/0_AQI_Genesis.py" >> "$LOG_DIR/aqigenesis_briefing_${TIMESTAMP}.log" 2>&1

# --- 2. Sync with Streamlit Cloud via Git ---
echo "Pushing database and updates to GitHub at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
cd "$SCRIPT_DIR" || exit

# Git commands (This pushes the fresh aqi_saas_backend.db to the cloud)
git add .
git commit -m "Automated terminal & DB sync: $TIMESTAMP"
git push origin main >> "$LOG_DIR/git_push_${TIMESTAMP}.log" 2>&1

echo "Pipeline finished and pushed to Git at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
exit 0