#!/bin/bash
# Run this from your LOCAL Windows machine (Git Bash)
# Uploads the fin-agents project to your Oracle VM
#
# Usage:
#   bash upload_to_oracle.sh <VM_PUBLIC_IP> <PATH_TO_SSH_KEY>
#
# Example:
#   bash upload_to_oracle.sh 140.238.x.x ~/.ssh/oracle_sip.key

VM_IP="${1:?Usage: bash upload_to_oracle.sh <VM_IP> <SSH_KEY>}"
SSH_KEY="${2:?Usage: bash upload_to_oracle.sh <VM_IP> <SSH_KEY>}"
REMOTE_USER="ubuntu"
REMOTE_DIR="~/fin-agents"

echo "Uploading fin-agents to $REMOTE_USER@$VM_IP ..."

# Sync project (exclude cache, venv, .git, local outputs)
rsync -avz --progress \
    --exclude '__pycache__' \
    --exclude '*.pyc' \
    --exclude '.git' \
    --exclude 'venv' \
    --exclude '*.egg-info' \
    --exclude 'outputs/*.csv' \
    --exclude 'outputs/*.json' \
    --exclude 'simulator/outputs/*.json' \
    -e "ssh -i $SSH_KEY -o StrictHostKeyChecking=no" \
    "$(dirname "$0")/../../../" \
    "$REMOTE_USER@$VM_IP:$REMOTE_DIR/"

echo ""
echo "Upload complete."
echo ""
echo "Next steps on the VM:"
echo "  ssh -i $SSH_KEY $REMOTE_USER@$VM_IP"
echo "  bash ~/fin-agents/sip_execution_mas/deploy/oracle_setup.sh"
echo "  nano ~/fin-agents/.env   # paste your API keys"
echo "  bash ~/fin-agents/sip_execution_mas/deploy/setup_cron.sh"
