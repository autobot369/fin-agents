"""
SIP Execution MAS — Main Entry Point
======================================
6-Node Gemini-powered execution loop.

Usage
-----
  # Dry-run (paper mode, default)
  python -m sip_execution_mas

  # Dry-run with custom SIP
  python -m sip_execution_mas --sip 750 --top 10 --core 5

  # Live Alpaca paper trading (still needs ALPACA_API_KEY + ALPACA_SECRET_KEY)
  python -m sip_execution_mas --live

  # Override risk limits
  python -m sip_execution_mas --max-position 0.20 --max-region 0.60

  # Custom TER ceiling (0.50% means only very cheap ETFs pass)
  python -m sip_execution_mas --ter 0.50

  # Force even if already invested this month (in dry-run it's always bypassed)
  python -m sip_execution_mas --force

Env vars
--------
  GEMINI_API_KEY      Required for Node 2 (scorer) and Node 3/4 (Gemini reasoning).
                      Falls back to VADER if not set.
  ALPACA_API_KEY      Optional. Enable Alpaca execution in --live mode.
  ALPACA_SECRET_KEY   Optional. Alpaca secret.
  ALPACA_PAPER        "true" (default) = Alpaca paper account.
"""
from __future__ import annotations

import argparse
import os
import sys
from pathlib import Path

# Ensure package root on sys.path
_ROOT = Path(__file__).resolve().parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))

# ── Import must come after path setup ─────────────────────────────────────────
import sip_execution_mas  # noqa: F401  triggers SSL patch

# Load .env from repo root (if present) before any API key reads
from dotenv import load_dotenv as _load_dotenv
_load_dotenv(_ROOT / ".env", override=False)

from sip_execution_mas.graph.workflow import run_sip_execution


def main() -> None:
    p = argparse.ArgumentParser(
        description="SIP Execution MAS — 6-Node Gemini Loop",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sip",          type=float, default=500.0,
                   help="Monthly SIP amount in USD (default: 500)")
    p.add_argument("--top",          type=int,   default=5,
                   help="Total ETFs to invest in (default: 5, locked v4 universe)")
    p.add_argument("--core",         type=int,   default=2,
                   help="Core ETF count (default: 2, locked v4 universe)")
    p.add_argument("--core-pct",     type=float, default=0.70,
                   help="Fraction of SIP for core bucket (default: 0.70)")
    p.add_argument("--ter",          type=float, default=0.70,
                   help="TER ceiling %% for ETF filter (default: 0.70)")
    p.add_argument("--max-position", type=float, default=0.50,
                   help="Max fraction of SIP per ETF (default: 0.50 = 50%%, raised for 5-ETF v4 universe)")
    p.add_argument("--max-region",   type=float, default=0.80,
                   help="Max fraction of SIP per region (default: 0.80 = 80%%, raised for LSE-heavy v4 universe)")
    p.add_argument("--live",         action="store_true",
                   help="Disable dry-run; execute via broker API (default: off)")
    p.add_argument("--force",        action="store_true",
                   help="Bypass Rule 4: allow re-investment even if already done this month")
    p.add_argument("--ledger",       type=str, default="",
                   help="Path to portfolio_ledger.json (default: simulator/outputs/ inside sip_execution_mas/)")
    p.add_argument("--run-id",       type=str, default="",
                   help="Custom run ID (default: auto-generated)")
    args = p.parse_args()

    dry_run = not args.live

    # Warn if going live without Alpaca keys
    if not dry_run and not os.environ.get("ALPACA_API_KEY"):
        print("⚠  --live set but ALPACA_API_KEY not found — falling back to paper mode")
        dry_run = True

    # Warn if GEMINI_API_KEY missing
    if not os.environ.get("GEMINI_API_KEY"):
        print("⚠  GEMINI_API_KEY not set — Signal Scorer will use VADER fallback")

    print(f"""
╔══════════════════════════════════════════════════════════════╗
║     SIP EXECUTION MAS  —  Gemini 6-Node Loop  (v4)          ║
╠══════════════════════════════════════════════════════════════╣
║  SIP          : ${args.sip:.2f}
║  Universe     : v4 — 5 ETFs (2 core / 3 satellite)
║  TER ceiling  : {args.ter:.2f}%
║  Max position : {args.max_position*100:.0f}%
║  Max region   : {args.max_region*100:.0f}%
║  Mode         : {'DRY RUN (paper)' if dry_run else 'LIVE'}
╚══════════════════════════════════════════════════════════════╝
""")

    final_state = run_sip_execution(
        sip_amount       = args.sip,
        top_n            = args.top,
        core_count       = args.core,
        core_pct         = args.core_pct,
        ter_threshold    = args.ter / 100.0,
        max_position_pct = args.max_position,
        max_region_pct   = args.max_region,
        dry_run          = dry_run,
        force            = args.force,
        ledger_path      = args.ledger,
        run_id           = args.run_id,
    )

    status = final_state.get("execution_status", "unknown")
    if status in ("success", "dry_run", "partial"):
        sys.exit(0)
    else:
        sys.exit(1)


if __name__ == "__main__":
    main()
