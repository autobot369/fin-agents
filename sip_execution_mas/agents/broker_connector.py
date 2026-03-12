"""
Node 5 — Broker Connector (Executor)
======================================
Executes proposed orders at the broker. Three modes:

  PAPER (default)  : Simulates fills at live yfinance prices. No real money.
  ALPACA           : Real paper-trading account via Alpaca API (US ETFs only).
                     Requires env vars: ALPACA_API_KEY, ALPACA_SECRET_KEY
                     Set ALPACA_PAPER=true (default) for paper account.
  DHAN             : Indian broker stub — always paper for BSE ETFs.

dry_run=True  → always paper mode, no broker API calls
dry_run=False → uses Alpaca if keys present, else paper
"""
from __future__ import annotations

import os
import sys
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import yfinance as yf

from sip_execution_mas.graph.state import ExecutionResult, ProposedOrder, SIPExecutionState

_ROOT = Path(__file__).resolve().parent.parent.parent
if str(_ROOT) not in sys.path:
    sys.path.insert(0, str(_ROOT))


# ── USD/INR rate ──────────────────────────────────────────────────────────────

def _get_usd_inr() -> float:
    try:
        hist = yf.Ticker("USDINR=X").history(period="5d", auto_adjust=True)
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 84.0   # fallback


# ── Live price fetch ──────────────────────────────────────────────────────────

def _fetch_prices(tickers: List[str]) -> Dict[str, float]:
    """Fetch latest close prices for a list of tickers."""
    prices: Dict[str, float] = {}
    for ticker in tickers:
        try:
            hist = yf.Ticker(ticker).history(period="5d", auto_adjust=True)
            if not hist.empty:
                prices[ticker] = round(float(hist["Close"].iloc[-1]), 4)
        except Exception:
            pass
        time.sleep(0.05)
    return prices


# ── Paper execution ───────────────────────────────────────────────────────────

def _execute_paper(
    orders: List[ProposedOrder],
    prices: Dict[str, float],
    usd_inr: float,
) -> List[ExecutionResult]:
    """Simulate fills at current market prices (paper mode)."""
    results: List[ExecutionResult] = []
    for order in orders:
        ticker = order["ticker"]
        price = prices.get(ticker)
        currency = order.get("currency", "USD")

        if price is None:
            results.append({
                "ticker":        ticker,
                "status":        "skipped",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    0.0,
                "units":         0.0,
                "price_native":  0.0,
                "currency":      currency,
                "broker":        "paper",
                "order_id":      None,
                "error":         "no_price",
            })
            continue

        if currency == "INR":
            inr_amount = order["monthly_usd"] * usd_inr
            units = inr_amount / price
            filled_usd = order["monthly_usd"]
        else:
            units = order["monthly_usd"] / price
            filled_usd = order["monthly_usd"]

        results.append({
            "ticker":        ticker,
            "status":        "dry_run",
            "requested_usd": order["monthly_usd"],
            "filled_usd":    round(filled_usd, 2),
            "units":         round(units, 6),
            "price_native":  price,
            "currency":      currency,
            "broker":        "paper",
            "order_id":      f"PAPER-{ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "error":         None,
        })
    return results


# ── Alpaca execution ──────────────────────────────────────────────────────────

def _execute_alpaca(
    orders: List[ProposedOrder],
    prices: Dict[str, float],
    usd_inr: float,
) -> List[ExecutionResult]:
    """
    Execute US ETF orders via Alpaca (paper or live account).
    BSE/NSE tickers are automatically routed to paper mode (Alpaca doesn't support them).
    """
    alpaca_key    = os.environ.get("ALPACA_API_KEY", "")
    alpaca_secret = os.environ.get("ALPACA_SECRET_KEY", "")
    paper_mode    = os.environ.get("ALPACA_PAPER", "true").lower() != "false"

    if not alpaca_key or not alpaca_secret:
        print("  [Node 5] Alpaca keys not found — falling back to paper mode")
        return _execute_paper(orders, prices, usd_inr)

    try:
        from alpaca.trading.client import TradingClient
        from alpaca.trading.requests import MarketOrderRequest
        from alpaca.trading.enums import OrderSide, TimeInForce
    except ImportError:
        print("  [Node 5] alpaca-py not installed (pip install alpaca-py) — falling back to paper")
        return _execute_paper(orders, prices, usd_inr)

    base_url = "https://paper-api.alpaca.markets" if paper_mode else "https://api.alpaca.markets"
    client = TradingClient(alpaca_key, alpaca_secret, paper=paper_mode)

    results: List[ExecutionResult] = []
    for order in orders:
        ticker   = order["ticker"]
        currency = order.get("currency", "USD")

        # Alpaca only handles US-listed tickers
        if currency == "INR" or ticker.endswith(".NS") or ticker.endswith(".BO"):
            price = prices.get(ticker)
            if price:
                inr_amount = order["monthly_usd"] * usd_inr
                units = inr_amount / price
                results.append({
                    "ticker":        ticker,
                    "status":        "dry_run",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    round(order["monthly_usd"], 2),
                    "units":         round(units, 6),
                    "price_native":  price,
                    "currency":      currency,
                    "broker":        "dhan_stub",
                    "order_id":      f"DHAN-{ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
                    "error":         None,
                })
            else:
                results.append({
                    "ticker":        ticker,
                    "status":        "skipped",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    0.0,
                    "units":         0.0,
                    "price_native":  0.0,
                    "currency":      currency,
                    "broker":        "dhan_stub",
                    "order_id":      None,
                    "error":         "no_price",
                })
            continue

        # Fractional shares via notional amount
        try:
            req = MarketOrderRequest(
                symbol        = ticker,
                notional      = order["monthly_usd"],
                side          = OrderSide.BUY,
                time_in_force = TimeInForce.DAY,
            )
            resp = client.submit_order(req)
            price = prices.get(ticker, 0.0)
            units = order["monthly_usd"] / price if price else 0.0

            results.append({
                "ticker":        ticker,
                "status":        "filled",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    round(order["monthly_usd"], 2),
                "units":         round(units, 6),
                "price_native":  price,
                "currency":      "USD",
                "broker":        f"alpaca_{'paper' if paper_mode else 'live'}",
                "order_id":      str(resp.id),
                "error":         None,
            })
        except Exception as exc:
            results.append({
                "ticker":        ticker,
                "status":        "error",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    0.0,
                "units":         0.0,
                "price_native":  prices.get(ticker, 0.0),
                "currency":      "USD",
                "broker":        f"alpaca_{'paper' if paper_mode else 'live'}",
                "order_id":      None,
                "error":         str(exc),
            })

    return results


# ── Node function ─────────────────────────────────────────────────────────────

def broker_connector_node(state: SIPExecutionState) -> dict:
    """
    Node 5 — Broker Connector

    Routes to Alpaca (if keys present and dry_run=False) or paper simulation.
    Always routes BSE/NSE tickers to Dhan stub (paper).
    """
    proposed_orders = state["proposed_orders"]
    dry_run         = state["dry_run"]

    print(f"\n[Node 5] Broker Connector — {'DRY RUN' if dry_run else 'LIVE'} "
          f"mode, {len(proposed_orders)} orders …")

    tickers    = [o["ticker"] for o in proposed_orders]
    prices     = _fetch_prices(tickers)
    usd_inr    = _get_usd_inr()

    print(f"  [Node 5] USD/INR = {usd_inr:.2f}")
    for t, p in prices.items():
        print(f"  [Node 5]   {t:<22}  {p}")

    if dry_run:
        results = _execute_paper(proposed_orders, prices, usd_inr)
        exec_status = "dry_run"
    elif os.environ.get("ALPACA_API_KEY"):
        results = _execute_alpaca(proposed_orders, prices, usd_inr)
        filled  = sum(1 for r in results if r["status"] in ("filled", "dry_run"))
        exec_status = "success" if filled == len(results) else "partial"
    else:
        results = _execute_paper(proposed_orders, prices, usd_inr)
        exec_status = "dry_run"

    filled_count  = sum(1 for r in results if r["status"] in ("filled", "dry_run"))
    skipped_count = sum(1 for r in results if r["status"] == "skipped")
    error_count   = sum(1 for r in results if r["status"] == "error")
    total_usd     = sum(r["filled_usd"] for r in results)

    print(f"  [Node 5] Filled: {filled_count}  Skipped: {skipped_count}  "
          f"Errors: {error_count}  Total: ${total_usd:.2f}")

    return {
        "execution_results": results,
        "execution_status":  exec_status,
        "usd_inr_rate":      usd_inr,
    }
