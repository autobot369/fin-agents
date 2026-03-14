"""
SIP Simulator — Sentiment-Weighted Dynamic SIP
================================================
Simulates a $500/month (configurable) buy-only SIP across ETFs ranked
by etf-selection-mas, using the Core/Satellite allocation strategy.

Usage
-----
  # Run simulator using the most recent rankings
  python -m simulator.main

  # Custom SIP, period, and portfolio size
  python -m simulator.main --sip 1000 --months 24 --top 10 --core 5

  # Regenerate fresh rankings first via etf-selection-mas (requires ANTHROPIC_API_KEY)
  python -m simulator.main --run-mas

  # Regenerate rankings via sip_execution_mas Gemini pipeline (requires GEMINI_API_KEY)
  python -m simulator.main --run-sip-mas

  # Point at a specific rankings file
  python -m simulator.main --rankings /path/to/rankings.json

Strategy
--------
  Core    (default 70%): top `--core` ETFs, score-weighted within bucket
  Satellite (remaining): ETFs core+1 … top_n, score-weighted within bucket
  Weight_i = ConsensusScore_i / Σ ConsensusScore_bucket

  No selling. Units accumulate every month regardless of price movement.
"""

from __future__ import annotations

import argparse
import json
import os
import subprocess
import sys
from datetime import datetime
from pathlib import Path

# Ensure both sip_execution_mas/ and fin-agents/ are on sys.path
_ROOT     = Path(__file__).resolve().parent.parent   # sip_execution_mas/
_FIN_ROOT = _ROOT.parent                              # fin-agents/
for _p in (str(_ROOT), str(_FIN_ROOT)):
    if _p not in sys.path:
        sys.path.insert(0, _p)

from simulator.allocator import compute_allocation
from simulator.backtest import get_usd_inr_rate, run_backtest
from simulator.report import generate_markdown_report, print_terminal_report, save_report

_DEFAULT_RANKINGS = _FIN_ROOT / "etf-selection-mas" / "outputs" / "rankings.json"
_OUTPUT_DIR       = Path(__file__).resolve().parent / "outputs"


# ── CLI ──────────────────────────────────────────────────────────────────────

def _parse_args() -> argparse.Namespace:
    p = argparse.ArgumentParser(
        description="Sentiment-Weighted Dynamic SIP Simulator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog=__doc__,
    )
    p.add_argument("--sip",      type=float, default=500.0,
                   help="Monthly SIP amount in USD (default: 500)")
    p.add_argument("--months",   type=int,   default=12,
                   help="Months of historical backtest (default: 12)")
    p.add_argument("--top",      type=int,   default=10,
                   help="Total ETFs to invest in, core + satellite (default: 10)")
    p.add_argument("--core",     type=int,   default=5,
                   help="Number of top ETFs in the Core bucket (default: 5)")
    p.add_argument("--core-pct", type=float, default=0.70,
                   help="Fraction of SIP for Core bucket (default: 0.70 = 70%%)")
    p.add_argument("--rankings", type=str,   default=str(_DEFAULT_RANKINGS),
                   help="Path to rankings.json from etf-selection-mas")
    p.add_argument("--run-mas",     action="store_true",
                   help="Run etf-selection-mas pipeline first for fresh rankings (requires ANTHROPIC_API_KEY)")
    p.add_argument("--run-sip-mas", action="store_true",
                   help="Run sip_execution_mas Gemini pipeline for fresh rankings (requires GEMINI_API_KEY)")
    p.add_argument("--ter",         type=float, default=0.70,
                   help="TER ceiling %% passed to MAS when using --run-mas / --run-sip-mas (default: 0.70)")
    return p.parse_args()


# ── MAS integration ───────────────────────────────────────────────────────────

def _run_mas(ter: float, top_n: int) -> None:
    mas_dir = _FIN_ROOT / "etf-selection-mas"
    print(f"[simulator] Running etf-selection-mas pipeline (TER <= {ter}%, top {top_n}) ...")
    result = subprocess.run(
        [sys.executable, "main.py", "--ter", str(ter), "--top", str(top_n)],
        cwd=str(mas_dir),
    )
    if result.returncode != 0:
        print("[simulator] ERROR: etf-selection-mas failed. Check ANTHROPIC_API_KEY.")
        sys.exit(1)
    print("[simulator] etf-selection-mas complete.\n")


def _run_sip_mas_rankings(ter: float, top_n: int, core_count: int) -> list:
    """
    Run sip_execution_mas in dry-run mode to get Gemini-scored ETF rankings.
    Returns a rankings list compatible with compute_allocation() / run_backtest().
    """
    import os
    if not os.environ.get("GEMINI_API_KEY"):
        print("[simulator] WARNING: GEMINI_API_KEY not set - Signal Scorer will use VADER fallback")

    print(f"[simulator] Running sip_execution_mas Gemini pipeline (TER <= {ter}%, top {top_n}) ...")
    from sip_execution_mas.graph.workflow import run_sip_execution

    final_state = run_sip_execution(
        sip_amount       = 500.0,   # arbitrary — we only need scores
        top_n            = top_n,
        core_count       = core_count,
        core_pct         = 0.70,
        ter_threshold    = ter / 100.0,
        max_position_pct = 0.15,
        max_region_pct   = 0.50,
        dry_run          = True,
        force            = True,    # bypass duplicate-month check
    )

    proposed_orders = final_state.get("proposed_orders", [])
    if not proposed_orders:
        print("[simulator] ERROR: sip_execution_mas returned no proposed orders.")
        sys.exit(1)

    # Convert ProposedOrder → rankings format expected by compute_allocation / run_backtest
    rankings = []
    for order in proposed_orders:
        rankings.append({
            "ticker":        order["ticker"],
            "name":          order.get("name", order["ticker"]),
            "region":        order.get("region", "INTL"),
            "market":        order.get("market", "NASDAQ"),
            "category":      order.get("bucket", "core"),
            "final_score":   order.get("consensus_score", 0.5),
            "expense_ratio": None,    # not directly available in ProposedOrder
            "sentiment_score": order.get("sentiment_score", 0.5),
            "expense_score":   order.get("expense_score", 0.5),
        })

    print(f"[simulator] sip_execution_mas returned {len(rankings)} scored ETFs.\n")
    return rankings


def _load_rankings(path: str) -> dict:
    if not os.path.exists(path):
        print(
            f"[simulator] ERROR: Rankings file not found: {path}\n"
            f"  Run etf-selection-mas first:  python etf-selection-mas/main.py\n"
            f"  Or use --run-mas to do it automatically."
        )
        sys.exit(1)
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


# ── Main ─────────────────────────────────────────────────────────────────────

def main() -> None:
    args = _parse_args()

    # Validate core < top
    if args.core >= args.top:
        print(f"[simulator] ERROR: --core ({args.core}) must be less than --top ({args.top})")
        sys.exit(1)

    if args.run_mas and args.run_sip_mas:
        print("[simulator] ERROR: --run-mas and --run-sip-mas are mutually exclusive.")
        sys.exit(1)

    # Option A: use sip_execution_mas Gemini pipeline for fresh rankings
    if args.run_sip_mas:
        rankings      = _run_sip_mas_rankings(ter=args.ter, top_n=args.top, core_count=args.core)
        boom_triggers = []
        generated_at  = datetime.today().strftime("%Y-%m-%d %H:%M:%S") + " (sip_execution_mas)"
        available     = len(rankings)
    else:
        # Option B: optionally regenerate via etf-selection-mas, then load from file
        if args.run_mas:
            _run_mas(ter=args.ter, top_n=20)

        print(f"[simulator] Loading rankings from: {args.rankings}")
        data          = _load_rankings(args.rankings)
        rankings      = data.get("rankings", [])
        boom_triggers = data.get("boom_triggers_fired", [])
        generated_at  = data.get("generated_at", "unknown")
        available     = len(rankings)

    if not rankings:
        print("[simulator] ERROR: rankings.json contains no ETFs.")
        sys.exit(1)

    # Cap --top to available rankings
    top_n = min(args.top, available)
    core  = min(args.core, top_n - 1)
    if top_n < args.top:
        print(f"[simulator] Note: only {available} ranked ETFs available; using top-{top_n}.")

    print(
        f"[simulator] Rankings: {available} ETFs  |  "
        f"Investing in top-{top_n}  |  "
        f"Core: {core}, Satellite: {top_n - core}  |  "
        f"Generated: {generated_at}"
    )

    # Step 1 — Allocation plan
    plan = compute_allocation(
        rankings   = rankings,
        sip_amount = args.sip,
        top_n      = top_n,
        core_count = core,
        core_pct   = args.core_pct,
    )

    # Step 2 — Exchange rate
    print(f"[simulator] Fetching USD/INR exchange rate ...")
    usd_inr = get_usd_inr_rate()
    print(f"[simulator] USD/INR: {usd_inr:.2f}")

    # Step 3 — Backtest (buy-only)
    print(f"[simulator] Running {args.months}-month buy-only backtest ...")
    result = run_backtest(plan, months=args.months, usd_inr_rate=usd_inr)

    # Step 4 — Terminal report
    print_terminal_report(plan, result, args.months, boom_triggers)

    # Step 5 — Markdown report
    md = generate_markdown_report(
        plan                 = plan,
        result               = result,
        months               = args.months,
        boom_triggers        = boom_triggers,
        rankings_generated_at= generated_at,
    )
    _OUTPUT_DIR.mkdir(exist_ok=True)
    date_str  = datetime.today().strftime("%Y%m%d")
    out_path  = _OUTPUT_DIR / f"SIP_SIMULATION_{date_str}.md"
    save_report(md, str(out_path))
    print(f"[simulator] Report saved -> {out_path}")


if __name__ == "__main__":
    main()
