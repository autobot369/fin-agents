#!/bin/bash
# Oracle Cloud VM — One-time setup script
# Run as: bash oracle_setup.sh
# Ubuntu 22.04 / ARM (A1.Flex)

set -e

echo "============================================================"
echo "  SIP Execution MAS — Oracle Cloud Setup"
echo "============================================================"

# 1. System updates
echo "[1/6] Updating system packages..."
sudo apt-get update -qq && sudo apt-get upgrade -y -qq

# 2. Python 3.11
echo "[2/6] Installing Python 3.11..."
sudo apt-get install -y -qq python3.11 python3.11-venv python3.11-dev python3-pip git

# 3. Project directory
echo "[3/6] Creating project directory..."
mkdir -p ~/fin-agents
cd ~/fin-agents

# 4. Virtual environment
echo "[4/6] Creating virtual environment..."
python3.11 -m venv venv
source venv/bin/activate

# 5. Install dependencies
echo "[5/6] Installing Python dependencies..."
pip install --upgrade pip -q
pip install -q \
    yfinance>=0.2.40 \
    pandas>=2.0.0 \
    python-dotenv>=1.0.0 \
    langgraph>=0.2.0 \
    langchain-google-genai \
    google-generativeai>=0.8.0 \
    vaderSentiment>=3.3.2 \
    ddgs \
    certifi>=2024.0.0 \
    apscheduler>=3.10.0

# 6. Output directories
echo "[6/6] Creating output directories..."
mkdir -p ~/fin-agents/sip_execution_mas/outputs
mkdir -p ~/fin-agents/sip_execution_mas/simulator/outputs

echo ""
echo "============================================================"
echo "  Setup complete!"
echo "  Next: upload your project files and configure .env"
echo "  See: deploy/README_ORACLE.md"
echo "============================================================"
