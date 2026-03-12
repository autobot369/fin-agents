"""
Node 6 — Execution Logger (Memory)
=====================================
Persists the run outcome in two places:

  1. portfolio_ledger.json  (shared with simulator Streamlit dashboard)
     → Uses simulator.ledger.build_and_append_entry so the existing
       Streamlit app can read both scheduler and MAS runs.

  2. execution_log.csv  (audit trail per order)
     → Records broker, order_id, status, units, fill price for every order.

Also prints a formatted terminal summary.
"""
from __future__ import annotations

import csv
import os
import sys
import uuid
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List

from sip_execution_mas.graph.state import SIPExecutionState

_ROOT     = Path(__file__).resolve().parent.parent.parent   # fin-agents/
_SIP_ROOT = Path(__file__).resolve().parent.parent          # sip_execution_mas/
if str(_SIP_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIP_ROOT))

from simulator.ledger import (   # type: ignore
    aggregate_holdings,
    build_and_append_entry,
    load_ledger,
    ledger_summary,
    save_ledger,
)

_DEFAULT_LEDGER   = _SIP_ROOT / "simulator" / "outputs" / "portfolio_ledger.json"
_DEFAULT_EXEC_LOG = _SIP_ROOT / "outputs" / "execution_log.csv"

_CSV_HEADERS = [
    "run_id", "timestamp", "month", "ticker", "region", "bucket",
    "broker", "status", "requested_usd", "filled_usd", "units",
    "price_native", "currency", "order_id", "error",
]


# ── Ledger entry builder ──────────────────────────────────────────────────────

def _build_ledger_entry_from_results(
    state: SIPExecutionState,
) -> None:
    """
    Append a ledger entry using the existing simulator.ledger helpers.
    Uses allocation_plan (what was planned) as the source of truth;
    overlays executed prices from execution_results.
    """
    ledger_path = state.get("ledger_path", str(_DEFAULT_LEDGER))
    plan        = state.get("allocation_plan")
    results     = state.get("execution_results", [])
    usd_inr     = state.get("usd_inr_rate", 84.0)
    exec_status = state.get("execution_status", "unknown")

    if exec_status == "aborted" or not plan:
        return   # nothing to record

    # Build price map from execution results (use planned prices if not executed)
    price_map: Dict[str, float] = {}
    for r in results:
        if r["price_native"] and r["price_native"] > 0:
            price_map[r["ticker"]] = r["price_native"]

    # Fall back to yfinance prices from ETF data if broker didn't fill
    for alloc in plan.get("all", []):
        ticker = alloc["ticker"]
        if ticker not in price_map:
            etf_data = state.get("all_etf_data", {})
            rec = etf_data.get(ticker, {})
            price = rec.get("current_price")
            if price:
                price_map[ticker] = float(price)

    if not price_map:
        print("  [Node 6] ⚠  No prices available — skipping ledger write")
        return

    ledger = load_ledger(ledger_path)
    build_and_append_entry(
        ledger                = ledger,
        plan                  = plan,
        prices                = price_map,
        usd_inr_rate          = usd_inr,
        rankings_generated_at = datetime.now().strftime("%Y-%m-%d"),
    )
    save_ledger(ledger, ledger_path)
    print(f"  [Node 6] Ledger updated → {ledger_path}")


# ── CSV execution log ─────────────────────────────────────────────────────────

def _write_csv_log(
    run_id: str,
    state: SIPExecutionState,
    log_path: str,
) -> None:
    results     = state.get("execution_results", [])
    proposed    = state.get("proposed_orders", [])
    timestamp   = datetime.now().strftime("%Y-%m-%d %H:%M:%S")
    month       = datetime.now().strftime("%Y-%m")

    # Map ticker → bucket from proposed orders
    bucket_map  = {o["ticker"]: o.get("bucket", "?") for o in proposed}

    # Ensure directory
    Path(log_path).parent.mkdir(parents=True, exist_ok=True)

    file_exists = os.path.isfile(log_path)
    with open(log_path, "a", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=_CSV_HEADERS)
        if not file_exists:
            writer.writeheader()

        for r in results:
            writer.writerow({
                "run_id":        run_id,
                "timestamp":     timestamp,
                "month":         month,
                "ticker":        r["ticker"],
                "region":        next(
                    (o["region"] for o in proposed if o["ticker"] == r["ticker"]), "?"),
                "bucket":        bucket_map.get(r["ticker"], "?"),
                "broker":        r["broker"],
                "status":        r["status"],
                "requested_usd": r["requested_usd"],
                "filled_usd":    r["filled_usd"],
                "units":         r["units"],
                "price_native":  r["price_native"],
                "currency":      r["currency"],
                "order_id":      r.get("order_id", ""),
                "error":         r.get("error", ""),
            })
    print(f"  [Node 6] Execution log → {log_path}")


# ── Terminal summary ──────────────────────────────────────────────────────────

def _print_summary(run_id: str, state: SIPExecutionState) -> None:
    results     = state.get("execution_results", [])
    proposed    = state.get("proposed_orders", [])
    exec_status = state.get("execution_status", "unknown")
    sip         = state.get("sip_amount", 500.0)
    usd_inr     = state.get("usd_inr_rate", 84.0)
    violations  = state.get("risk_violations", [])
    audit_notes = state.get("risk_audit_notes", "")

    width = 64
    print(f"\n{'='*width}")
    print(f"  SIP EXECUTION MAS  —  Run {run_id}")
    print(f"  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Status : {exec_status.upper()}")
    print(f"{'='*width}")

    if exec_status == "aborted":
        print("\n  ✗ ABORTED — Risk audit failed after max retries")
        if violations:
            for v in violations:
                print(f"    • {v}")
        return

    print(f"\n  {'#':<3} {'Ticker':<22} {'Bkt':<4} {'Price':>10} {'Units':>10} {'USD':>8}")
    print(f"  {'─'*width}")
    total_filled = 0.0
    for i, r in enumerate(results, 1):
        order = next((o for o in proposed if o["ticker"] == r["ticker"]), None)
        bucket = order["bucket"][:3].upper() if order else "?"
        curr_label = r["currency"]
        price_str = f"{r['price_native']:.2f} {curr_label}" if r["price_native"] else "N/A"
        units_str = f"{r['units']:.4f}" if r["units"] else "-"
        usd_str = f"${r['filled_usd']:.2f}" if r["filled_usd"] else "$0"
        status_tag = "" if r["status"] in ("filled", "dry_run") else f" [{r['status']}]"
        print(f"  {i:<3} {r['ticker']:<22} {bucket:<4} {price_str:>14}  "
              f"{units_str:>10}  {usd_str:>8}{status_tag}")
        total_filled += r["filled_usd"]

    print(f"  {'─'*width}")
    print(f"  Total invested this month : ${total_filled:.2f}")
    print(f"  USD/INR rate              : {usd_inr:.2f}")

    try:
        ledger_path = state.get("ledger_path", str(_DEFAULT_LEDGER))
        ledger = load_ledger(ledger_path)
        summary = ledger_summary(ledger)
        print(f"  Ledger total (all months) : ${summary['total_invested_usd']:.2f} "
              f"over {summary['months']} month(s)")
    except Exception:
        pass

    print(f"\n  Macro: {state.get('macro_summary', '')[:120]}")
    triggers = state.get("boom_triggers", [])
    if triggers:
        print(f"  Boom triggers: {', '.join(triggers)}")
    if audit_notes and exec_status != "aborted":
        print(f"  Audit: {audit_notes[:100]}")
    print()


# ── Node function ─────────────────────────────────────────────────────────────

def execution_logger_node(state: SIPExecutionState) -> dict:
    """
    Node 6 — Execution Logger

    Writes the trade outcome to the ledger and CSV, prints terminal summary.
    Terminal node — always reached regardless of abort/success.
    """
    run_id     = state.get("run_id", str(uuid.uuid4())[:8].upper())
    log_path   = str(_DEFAULT_EXEC_LOG)

    print(f"\n[Node 6] Execution Logger — run_id={run_id}")

    # 1. Append to portfolio_ledger.json
    _build_ledger_entry_from_results(state)

    # 2. Write CSV execution log
    _write_csv_log(run_id, state, log_path)

    # 3. Print terminal summary
    _print_summary(run_id, state)

    return {
        "log_path":   log_path,
        "run_id_out": run_id,
    }
