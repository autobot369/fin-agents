"""
Portfolio Ledger
=================
Append-only record of every monthly SIP investment decision.

One entry per calendar month. Each entry captures:
  - The allocation plan used (Core / Satellite buckets)
  - Price paid per ETF at time of investment
  - Units accumulated
  - USD/INR rate at investment time

The ledger file is the single source of truth for the Streamlit dashboard.
No ANTHROPIC_API_KEY or live MAS pipeline required at investment time —
only rankings.json (pre-generated) and yfinance prices.
"""

from __future__ import annotations

import json
import os
from datetime import date, datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

from simulator.state import AllocationPlan

_LEDGER_FILE = Path(__file__).resolve().parent / "outputs" / "portfolio_ledger.json"


# ── Load / save ───────────────────────────────────────────────────────────────

def load_ledger(path: str = str(_LEDGER_FILE)) -> Dict[str, Any]:
    """Load ledger from disk. Returns an empty ledger if file does not exist."""
    if not os.path.exists(path):
        return {
            "version":    "1.0",
            "created_at": date.today().isoformat(),
            "entries":    [],
        }
    with open(path, "r", encoding="utf-8") as f:
        return json.load(f)


def save_ledger(ledger: Dict[str, Any], path: str = str(_LEDGER_FILE)) -> None:
    """Write ledger to disk."""
    os.makedirs(os.path.dirname(path), exist_ok=True)
    ledger["last_updated"] = datetime.now().isoformat()
    with open(path, "w", encoding="utf-8") as f:
        json.dump(ledger, f, indent=2, default=str)


# ── Guards ────────────────────────────────────────────────────────────────────

def already_invested_this_month(ledger: Dict[str, Any]) -> Optional[str]:
    """
    Return the entry date string if an investment was already recorded this
    calendar month, otherwise return None.
    """
    this_month = date.today().strftime("%Y-%m")
    for entry in ledger.get("entries", []):
        if entry.get("month") == this_month:
            return entry.get("date")
    return None


# ── Entry builder ─────────────────────────────────────────────────────────────

def build_and_append_entry(
    ledger: Dict[str, Any],
    plan: AllocationPlan,
    prices: Dict[str, float],     # ticker → current price in native currency
    usd_inr_rate: float,
    rankings_generated_at: str,
) -> Dict[str, Any]:
    """
    Record this month's SIP investment into the ledger.

    For each ETF in the allocation plan:
      - USD allocation is converted to INR for BSE tickers
      - Units bought = allocation / price
    Returns the appended entry dict.
    """
    today     = date.today()
    positions = []
    total_invested = 0.0

    for alloc in plan["all"]:
        ticker     = alloc["ticker"]
        price      = prices.get(ticker)
        if not price or price <= 0:
            print(f"  [ledger] ⚠ No price for {ticker} — skipped this month")
            continue

        monthly_usd = alloc["monthly_usd"]

        if alloc["region"] == "BSE":
            units = (monthly_usd * usd_inr_rate) / price
        else:
            units = monthly_usd / price

        positions.append({
            "ticker":          ticker,
            "name":            alloc["name"],
            "bucket":          alloc["bucket"],
            "region":          alloc["region"],
            "category":        alloc["category"],
            "trade_on":        alloc["trade_on"],
            "consensus_score": round(alloc["consensus_score"], 5),
            "weight":          round(alloc["weight"], 5),
            "monthly_usd":     round(monthly_usd, 2),
            "price_native":    round(price, 4),
            "units_bought":    round(units, 6),
            "currency":        alloc["currency"],
        })
        total_invested += monthly_usd

    entry = {
        "month":                 today.strftime("%Y-%m"),
        "date":                  today.isoformat(),
        "rankings_generated_at": rankings_generated_at,
        "usd_inr_rate":          round(usd_inr_rate, 4),
        "sip_amount":            plan["sip_amount"],
        "core_budget":           plan["core_budget"],
        "satellite_budget":      plan["satellite_budget"],
        "core_pct":              plan["core_pct"],
        "total_invested_usd":    round(total_invested, 2),
        "positions":             positions,
    }

    ledger["entries"].append(entry)
    return entry


# ── Portfolio aggregation (used by Streamlit) ─────────────────────────────────

def aggregate_holdings(ledger: Dict[str, Any]) -> Dict[str, Dict[str, Any]]:
    """
    Aggregate all ledger entries into a current holdings map.

    Returns:
        { ticker: { units, invested_usd, region, bucket, category,
                    trade_on, name, entries: [per-month records] } }
    """
    holdings: Dict[str, Dict[str, Any]] = {}

    for entry in ledger.get("entries", []):
        for pos in entry.get("positions", []):
            ticker = pos["ticker"]
            if ticker not in holdings:
                holdings[ticker] = {
                    "ticker":      ticker,
                    "name":        pos.get("name", ticker),
                    "region":      pos.get("region", "?"),
                    "bucket":      pos.get("bucket", "?"),
                    "category":    pos.get("category", "—"),
                    "trade_on":    pos.get("trade_on", "—"),
                    "currency":    pos.get("currency", "USD"),
                    "total_units": 0.0,
                    "invested_usd":0.0,
                    "monthly_records": [],
                }
            holdings[ticker]["total_units"]   += pos["units_bought"]
            holdings[ticker]["invested_usd"]  += pos["monthly_usd"]
            holdings[ticker]["monthly_records"].append({
                "month":       entry["month"],
                "units":       pos["units_bought"],
                "price":       pos["price_native"],
                "invested":    pos["monthly_usd"],
                "usd_inr":     entry["usd_inr_rate"],
            })

    # Round aggregates
    for h in holdings.values():
        h["total_units"]  = round(h["total_units"], 6)
        h["invested_usd"] = round(h["invested_usd"], 2)

    return holdings


def ledger_summary(ledger: Dict[str, Any]) -> Dict[str, Any]:
    """Return top-level ledger statistics."""
    entries = ledger.get("entries", [])
    if not entries:
        return {"months": 0, "total_invested_usd": 0.0, "tickers": []}

    total_invested = sum(e.get("total_invested_usd", 0) for e in entries)
    tickers = list({
        pos["ticker"]
        for e in entries
        for pos in e.get("positions", [])
    })

    return {
        "months":            len(entries),
        "first_investment":  entries[0]["date"],
        "last_investment":   entries[-1]["date"],
        "total_invested_usd": round(total_invested, 2),
        "tickers":           tickers,
    }
