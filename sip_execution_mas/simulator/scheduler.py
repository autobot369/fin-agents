"""
Monthly SIP Scheduler
======================
Fires on the 1st of every month. Delegates execution to the
6-node Gemini MAS (sip_execution_mas) for scoring, optimization,
risk audit, and broker routing.

Usage
-----
  # Execute an investment immediately (dry-run / paper mode)
  python -m simulator.scheduler --now

  # Start the persistent monthly scheduler (Ctrl+C to stop)
  python -m simulator.scheduler

  # Show ledger status
  python -m simulator.scheduler --status

  # Custom SIP settings
  python -m simulator.scheduler --now --sip 750 --top 10 --core 5

  # Live broker execution (requires ALPACA_API_KEY)
  python -m simulator.scheduler --now --live

  # Force re-investment even if already done this month
  python -m simulator.scheduler --now --force

Scheduler fires at 09:00 on the 1st of every month (local time).
Requires GEMINI_API_KEY for Node 2 (scorer). Falls back to VADER
sentiment if not set.
"""

from __future__ import annotations

import argparse
import os
import sys
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, Optional

_ROOT     = Path(__file__).resolve().parent.parent   # sip_execution_mas/
_FIN_ROOT = _ROOT.parent                              # fin-agents/
for _p in (str(_ROOT), str(_FIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from dotenv import load_dotenv
load_dotenv(_FIN_ROOT / ".env", override=False)

from simulator.ledger import (
    already_invested_this_month,
    ledger_summary,
    load_ledger,
)

_LEDGER_FILE = Path(__file__).resolve().parent / "outputs" / "portfolio_ledger.json"


# ── Core investment function ──────────────────────────────────────────────────

def execute_monthly_investment(
    sip_amount:    float = 500.0,
    top_n:         int   = 10,
    core_count:    int   = 5,
    core_pct:      float = 0.70,
    ter_threshold: float = 0.007,
    ledger_path:   str   = str(_LEDGER_FILE),
    force:         bool  = False,
    live:          bool  = False,
) -> Optional[Dict[str, Any]]:
    """
    Execute one month's SIP investment via sip_execution_mas.

    Delegates entirely to the 6-node Gemini MAS:
      Node 1: Regional Researcher (ETF universe + news)
      Node 2: Signal Scorer      (Gemini/VADER sentiment)
      Node 3: Portfolio Optimizer (allocation)
      Node 4: Risk Auditor       (hard rules + Gemini explanation)
      Node 5: Broker Connector   (paper / Alpaca / Dhan)
      Node 6: Execution Logger   (portfolio_ledger.json + execution_log.csv)

    Returns the final LangGraph state dict, or None if aborted.
    """
    now = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    print(f"\n{'='*60}")
    print(f"  SIP SCHEDULER  —  Monthly Investment Run (via MAS)")
    print(f"  {now}")
    print(f"{'='*60}")

    if not os.environ.get("GEMINI_API_KEY"):
        print("  ⚠  GEMINI_API_KEY not set — Signal Scorer will use VADER fallback")

    from sip_execution_mas.graph.workflow import run_sip_execution

    final_state = run_sip_execution(
        sip_amount    = sip_amount,
        top_n         = top_n,
        core_count    = core_count,
        core_pct      = core_pct,
        ter_threshold = ter_threshold,
        max_position_pct = 0.50,   # v4: 5-ETF universe, core ETFs each hold ~35%
        max_region_pct   = 0.80,   # v4: LSE-heavy universe (3 of 5 ETFs on LSE)
        dry_run       = not live,
        force         = force,
        ledger_path   = ledger_path,
    )

    status = final_state.get("execution_status", "unknown")
    if status == "aborted":
        print(f"\n  [scheduler] MAS run aborted — no investment recorded.")
        return None

    print(f"\n  [scheduler] MAS run complete — status: {status}")
    return final_state


# ── Status printer ────────────────────────────────────────────────────────────

def print_status(ledger_path: str = str(_LEDGER_FILE)) -> None:
    ledger  = load_ledger(ledger_path)
    summary = ledger_summary(ledger)

    print(f"\n{'='*60}")
    print(f"  LEDGER STATUS")
    print(f"{'='*60}")

    if summary["months"] == 0:
        print("  No investments recorded yet.")
        print("  Run with --now to record the first investment.\n")
        return

    print(f"  Months invested    : {summary['months']}")
    print(f"  First investment   : {summary['first_investment']}")
    print(f"  Last investment    : {summary['last_investment']}")
    print(f"  Total invested     : ${summary['total_invested_usd']:.2f}")
    print(f"  Tickers held       : {', '.join(sorted(summary['tickers']))}")

    this_month = already_invested_this_month(ledger)
    if this_month:
        print(f"  This month         : ✓ invested on {this_month}")
    else:
        print(f"  This month         : ✗ not yet invested")

    print(f"\n  Recent entries:")
    for entry in ledger["entries"][-3:]:
        print(f"    {entry['date']}  ${entry['total_invested_usd']:.2f}  "
              f"({len(entry['positions'])} positions)")
    print()


# ── Scheduler ─────────────────────────────────────────────────────────────────

def start_scheduler(
    sip_amount:    float,
    top_n:         int,
    core_count:    int,
    core_pct:      float,
    ter_threshold: float,
    ledger_path:   str,
    live:          bool,
    hour:          int,
    minute:        int,
) -> None:
    try:
        from apscheduler.schedulers.blocking import BlockingScheduler
        from apscheduler.triggers.cron import CronTrigger
    except ImportError:
        print("ERROR: apscheduler not installed. Run: pip install apscheduler")
        sys.exit(1)

    scheduler = BlockingScheduler(timezone="local")

    def _job():
        execute_monthly_investment(
            sip_amount    = sip_amount,
            top_n         = top_n,
            core_count    = core_count,
            core_pct      = core_pct,
            ter_threshold = ter_threshold,
            ledger_path   = ledger_path,
            live          = live,
        )

    scheduler.add_job(
        _job,
        trigger = CronTrigger(day=1, hour=hour, minute=minute),
        id      = "monthly_sip",
        name    = "Monthly SIP Investment",
    )

    next_run = scheduler.get_jobs()[0].next_run_time
    mode_label = "LIVE" if live else "DRY RUN (paper)"
    print(f"\n{'='*60}")
    print(f"  SIP SCHEDULER STARTED")
    print(f"  Fires on : 1st of every month at {hour:02d}:{minute:02d} local time")
    print(f"  Next run : {next_run}")
    print(f"  SIP      : ${sip_amount:.2f}  |  top-{top_n}  |  core-{core_count}")
    print(f"  Mode     : {mode_label}")
    print(f"  Ledger   : {ledger_path}")
    print(f"  Press Ctrl+C to stop.")
    print(f"{'='*60}\n")

    try:
        scheduler.start()
    except (KeyboardInterrupt, SystemExit):
        print("\n[scheduler] Stopped.")


# ── CLI ───────────────────────────────────────────────────────────────────────

def main() -> None:
    p = argparse.ArgumentParser(
        description="Monthly SIP Scheduler — powered by sip_execution_mas",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--now",      action="store_true",
                   help="Execute investment immediately and exit")
    p.add_argument("--status",   action="store_true",
                   help="Show ledger status and exit")
    p.add_argument("--force",    action="store_true",
                   help="Force re-investment even if already done this month")
    p.add_argument("--live",     action="store_true",
                   help="Enable live broker execution (default: dry-run / paper)")
    p.add_argument("--sip",      type=float, default=500.0,
                   help="Monthly SIP amount in USD (default: 500)")
    p.add_argument("--top",      type=int,   default=10,
                   help="Total ETFs to invest in (default: 10)")
    p.add_argument("--core",     type=int,   default=5,
                   help="Core ETF count (default: 5)")
    p.add_argument("--core-pct", type=float, default=0.70,
                   help="Core bucket fraction (default: 0.70)")
    p.add_argument("--ter",      type=float, default=0.70,
                   help="TER ceiling %% for ETF filter (default: 0.70)")
    p.add_argument("--ledger",   type=str,   default=str(_LEDGER_FILE),
                   help="Path to portfolio_ledger.json")
    p.add_argument("--hour",     type=int,   default=9,
                   help="Hour of day to fire (default: 9 = 09:00)")
    p.add_argument("--minute",   type=int,   default=0,
                   help="Minute of hour to fire (default: 0)")
    args = p.parse_args()

    if args.status:
        print_status(args.ledger)
        return

    if args.now:
        execute_monthly_investment(
            sip_amount    = args.sip,
            top_n         = args.top,
            core_count    = args.core,
            core_pct      = args.core_pct,
            ter_threshold = args.ter / 100.0,
            ledger_path   = args.ledger,
            force         = args.force,
            live          = args.live,
        )
        return

    # Default: start the persistent monthly scheduler
    start_scheduler(
        sip_amount    = args.sip,
        top_n         = args.top,
        core_count    = args.core,
        core_pct      = args.core_pct,
        ter_threshold = args.ter / 100.0,
        ledger_path   = args.ledger,
        live          = args.live,
        hour          = args.hour,
        minute        = args.minute,
    )


if __name__ == "__main__":
    main()
