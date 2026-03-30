#!/bin/bash

# --- Configuration ---
SCRIPT_DIR="/home/andrew/.ssh/Trading/AQI_Genesis"
LOG_DIR="/home/andrew/.ssh/Trading/AQI_Genesis/Logs"
PYTHON_EXEC="/home/andrew/miniconda3/envs/tf_noavx/bin/python"

# --- Ensure log directory exists ---
mkdir -p "$LOG_DIR"
TIMESTAMP=$(date +"%Y%m%d_%H%M%S")

# --- 1. Run AQI Genesis Data Engine (11:00 AM) ---
echo "Starting AQI Genesis Pulse script at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
# Removed the '&' so the shell waits for the massive data fetch to actually finish
"$PYTHON_EXEC" "$SCRIPT_DIR/0_AQI_Genesis.py" >> "$LOG_DIR/aqigenesis_briefing_${TIMESTAMP}.log" 2>&1

# --- 2. Staggered Delay (Wait for 11:15 AM) ---
# Assuming cron triggers this script at exactly 11:00 AM, a 900-second sleep guarantees an 11:15 AM start.
echo "Sleeping for 15 minutes before dashboard launch..." >> "$LOG_DIR/runner_${TIMESTAMP}.log"
sleep 900

# --- 3. Run Dashboard Script (11:15 AM) ---
echo "Starting Dashboard script at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
"$PYTHON_EXEC" "$SCRIPT_DIR/0_AQI_Genesis_Dashboard.py" >> "$LOG_DIR/aqigenesisdashboard_briefing_${TIMESTAMP}.log" 2>&1

# --- 4. Sync with Streamlit Cloud via Git ---
echo "Pushing database and updates to GitHub at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
cd "$SCRIPT_DIR" || exit

# Git commands (Your .gitignore will naturally protect your config.json and passwords here)
git add .
git commit -m "Automated terminal & DB sync: $TIMESTAMP"
git push origin main >> "$LOG_DIR/git_push_${TIMESTAMP}.log" 2>&1

echo "All scripts finished and pushed to Git at $(date)" >> "$LOG_DIR/runner_${TIMESTAMP}.log"
exit 0