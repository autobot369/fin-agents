#!/bin/bash
# Monthly SIP execution script — called by cron on Oracle VM
# Cron entry: 0 9 1 * * /home/ubuntu/fin-agents/sip_execution_mas/deploy/run_sip.sh

set -e

PROJECT_DIR="$HOME/fin-agents"
LOG_FILE="$PROJECT_DIR/sip_execution_mas/outputs/scheduler.log"
VENV="$PROJECT_DIR/venv/bin/python"

# Ensure log directory exists
mkdir -p "$PROJECT_DIR/sip_execution_mas/outputs"

echo "" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"
echo "[$(date '+%Y-%m-%d %H:%M:%S')] Monthly SIP triggered" >> "$LOG_FILE"
echo "============================================================" >> "$LOG_FILE"

cd "$PROJECT_DIR"

PYTHONUTF8=1 PYTHONIOENCODING=utf-8 \
    "$VENV" -m sip_execution_mas.simulator.scheduler --now --sip 500 \
    >> "$LOG_FILE" 2>&1

echo "[$(date '+%Y-%m-%d %H:%M:%S')] Done" >> "$LOG_FILE"
