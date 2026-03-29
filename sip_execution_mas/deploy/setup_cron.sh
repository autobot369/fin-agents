#!/bin/bash
# Run on the Oracle VM after upload_to_oracle.sh + oracle_setup.sh
# Installs the monthly cron job

SCRIPT_PATH="$HOME/fin-agents/sip_execution_mas/deploy/run_sip.sh"

# Make run script executable
chmod +x "$SCRIPT_PATH"

# Add cron job: 1st of every month at 09:00 AM UTC
CRON_ENTRY="0 9 1 * * $SCRIPT_PATH"

# Check if already exists
if crontab -l 2>/dev/null | grep -qF "$SCRIPT_PATH"; then
    echo "Cron job already exists — no changes made."
else
    # Append to existing crontab
    (crontab -l 2>/dev/null; echo "$CRON_ENTRY") | crontab -
    echo "Cron job installed:"
    echo "  $CRON_ENTRY"
fi

echo ""
echo "Current crontab:"
crontab -l

echo ""
echo "To test immediately (dry-run):"
echo "  cd ~/fin-agents && PYTHONUTF8=1 venv/bin/python -m sip_execution_mas.simulator.scheduler --now --sip 500"
