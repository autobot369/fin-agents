"""
Node 5 — Broker Connector (Executor)
======================================
Executes proposed orders at the broker. Three modes:

  PAPER (default)  : Simulates fills at live yfinance prices. No real money.
  IBKR             : Real orders via Interactive Brokers (ib_insync) for LSE ETFs.
                     Requires env vars: IBKR_HOST (default 127.0.0.1),
                                        IBKR_PORT (7497=paper / 7496=live),
                                        IBKR_CLIENT_ID (default 1)
  DHAN             : Indian broker via DhanHQ API for BSE ETFs.
                     Requires env vars: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN

Routing:
  ticker.endswith(".L")  → Route A: IBKR  (buy in GBP on LSE)
  ticker.endswith(".BO") → Route B: Dhan  (buy in INR on BSE)

dry_run=True  → always paper mode, no broker API calls
dry_run=False → uses IBKR/Dhan if keys present, else falls back to paper
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


# ── DhanHQ security ID mapping (BSE) ─────────────────────────────────────────
# Populate from: https://images.dhan.co/api-data/api-scrip-master.csv
# Column: SEM_SMST_SECURITY_ID for segment=BSE_EQ
_DHAN_SECURITY_IDS: Dict[str, str] = {
    "NIFTYBEES": "532788",  # Nippon India ETF Nifty BeES — BSE scrip code
    "MOM100":    "543590",  # Nifty100 Momentum ETF — BSE scrip code (verify against scrip master)
}


# ── FX rates ──────────────────────────────────────────────────────────────────

def _get_usd_inr() -> float:
    try:
        hist = yf.Ticker("USDINR=X").history(period="5d", auto_adjust=True)
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 2)
    except Exception:
        pass
    return 84.0   # fallback


def _get_usd_gbp() -> float:
    """USD per 1 GBP (e.g. 1.27 means £1 = $1.27)."""
    try:
        hist = yf.Ticker("GBPUSD=X").history(period="5d", auto_adjust=True)
        if not hist.empty:
            return round(float(hist["Close"].iloc[-1]), 4)
    except Exception:
        pass
    return 1.27   # fallback


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
    usd_gbp: float = 1.27,
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
            native_amount = order["monthly_usd"] * usd_inr
            units = native_amount / price
        elif currency == "GBP":
            # Convert USD budget → GBP, then divide by GBP price
            native_amount = order["monthly_usd"] / usd_gbp
            units = native_amount / price
        else:
            # USD — including USD-denominated LSE ETFs (VWRA, IUIT, VVSM)
            units = order["monthly_usd"] / price

        results.append({
            "ticker":        ticker,
            "status":        "dry_run",
            "requested_usd": order["monthly_usd"],
            "filled_usd":    round(order["monthly_usd"], 2),
            "units":         round(units, 6),
            "price_native":  price,
            "currency":      currency,
            "broker":        "paper",
            "order_id":      f"PAPER-{ticker}-{datetime.now().strftime('%Y%m%d%H%M%S')}",
            "error":         None,
        })
    return results


# ── Route A: IBKR execution (LSE / .L tickers) ───────────────────────────────

def _execute_ibkr(
    orders: List[ProposedOrder],
    prices: Dict[str, float],
    usd_gbp: float,
) -> List[ExecutionResult]:
    """
    Execute LSE ETF orders via Interactive Brokers using ib_insync.

    Orders are placed in GBP on the LSE exchange. The USD budget is converted
    to GBP using the live GBP/USD rate before calculating share quantity.

    Falls back to paper mode if IBKR keys are missing or ib_insync is not installed.
    """
    ibkr_host      = os.environ.get("IBKR_HOST", "127.0.0.1")
    ibkr_port      = int(os.environ.get("IBKR_PORT", "7497"))   # 7497=paper, 7496=live
    ibkr_client_id = int(os.environ.get("IBKR_CLIENT_ID", "1"))
    is_paper       = ibkr_port == 7497

    try:
        from ib_insync import IB, Stock, MarketOrder
    except ImportError:
        print("  [Node 5] ib_insync not installed (pip install ib_insync) — falling back to paper")
        return _execute_paper(orders, prices, usd_inr=0.0, usd_gbp=usd_gbp)

    ib = IB()
    try:
        ib.connect(ibkr_host, ibkr_port, clientId=ibkr_client_id, timeout=10)
    except Exception as exc:
        print(f"  [Node 5] IBKR connection failed ({exc}) — falling back to paper")
        return _execute_paper(orders, prices, usd_inr=0.0, usd_gbp=usd_gbp)

    broker_label = f"ibkr_{'paper' if is_paper else 'live'}"
    results: List[ExecutionResult] = []

    try:
        for order in orders:
            ticker   = order["ticker"]
            symbol   = ticker.replace(".L", "")     # "VWRA.L" → "VWRA"
            price    = prices.get(ticker)
            currency = order.get("currency", "GBP")

            # Currency gate: Node 1 captures the actual trading currency from
            # yfinance.  VWRA, IUIT, VVSM all trade in USD on LSE — no FX
            # conversion needed.  GBP-denominated ETFs still need USD→GBP.
            if currency == "USD":
                native_budget    = order["monthly_usd"]
                contract_ccy     = "USD"
                budget_label     = f"${native_budget:.2f}"
            else:
                # GBP (or unknown) — convert USD budget to pounds
                native_budget    = order["monthly_usd"] / usd_gbp
                contract_ccy     = "GBP"
                budget_label     = f"£{native_budget:.2f}"

            if price is None or native_budget <= 0:
                results.append({
                    "ticker":        ticker,
                    "status":        "skipped",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    0.0,
                    "units":         0.0,
                    "price_native":  0.0,
                    "currency":      currency,
                    "broker":        broker_label,
                    "order_id":      None,
                    "error":         "no_price" if price is None else "zero_budget",
                })
                continue

            quantity = int(native_budget / price)   # whole shares (LSE no fractional)
            if quantity < 1:
                results.append({
                    "ticker":        ticker,
                    "status":        "skipped",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    0.0,
                    "units":         0.0,
                    "price_native":  price,
                    "currency":      currency,
                    "broker":        broker_label,
                    "order_id":      None,
                    "error":         f"quantity_zero (budget {budget_label} < price {price:.4f} {contract_ccy})",
                })
                continue

            try:
                contract = Stock(symbol, "LSE", contract_ccy)
                ib.qualifyContracts(contract)
                mkt_order = MarketOrder("BUY", quantity)
                trade = ib.placeOrder(contract, mkt_order)
                ib.sleep(2)   # allow TWS to acknowledge

                filled_native = quantity * price
                if contract_ccy == "USD":
                    filled_usd = filled_native
                else:
                    filled_usd = filled_native * usd_gbp

                results.append({
                    "ticker":        ticker,
                    "status":        "filled",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    round(filled_usd, 2),
                    "units":         quantity,
                    "price_native":  price,
                    "currency":      currency,
                    "broker":        broker_label,
                    "order_id":      str(trade.order.orderId),
                    "error":         None,
                })
            except Exception as exc:
                results.append({
                    "ticker":        ticker,
                    "status":        "error",
                    "requested_usd": order["monthly_usd"],
                    "filled_usd":    0.0,
                    "units":         0.0,
                    "price_native":  price,
                    "currency":      currency,
                    "broker":        broker_label,
                    "order_id":      None,
                    "error":         str(exc),
                })
    finally:
        ib.disconnect()

    return results


# ── Route B: DhanHQ execution (BSE / .BO tickers) ────────────────────────────

def _execute_dhan(
    orders: List[ProposedOrder],
    prices: Dict[str, float],
    usd_inr: float,
) -> List[ExecutionResult]:
    """
    Execute BSE ETF orders via DhanHQ API.

    The USD budget is converted to INR before calculating share quantity.
    Security IDs are resolved via _DHAN_SECURITY_IDS mapping; tickers not in
    the map fall back to paper mode.

    Env vars required: DHAN_CLIENT_ID, DHAN_ACCESS_TOKEN
    """
    dhan_client_id    = os.environ.get("DHAN_CLIENT_ID", "")
    dhan_access_token = os.environ.get("DHAN_ACCESS_TOKEN", "")

    if not dhan_client_id or not dhan_access_token:
        print("  [Node 5] Dhan keys not found — falling back to paper mode")
        return _execute_paper(orders, prices, usd_inr=usd_inr, usd_gbp=1.27)

    try:
        from dhanhq import dhanhq
    except ImportError:
        print("  [Node 5] dhanhq not installed (pip install dhanhq) — falling back to paper")
        return _execute_paper(orders, prices, usd_inr=usd_inr, usd_gbp=1.27)

    dhan = dhanhq(dhan_client_id, dhan_access_token)
    results: List[ExecutionResult] = []

    for order in orders:
        ticker   = order["ticker"]
        symbol   = ticker.replace(".BO", "")       # "NIFTYBEES.BO" → "NIFTYBEES"
        price    = prices.get(ticker)
        currency = order.get("currency", "INR")
        sec_id   = _DHAN_SECURITY_IDS.get(symbol)

        if sec_id is None:
            print(f"  [Node 5] No DhanHQ security ID for {symbol} — using paper fill")
            paper = _execute_paper([order], prices, usd_inr=usd_inr, usd_gbp=1.27)
            results.extend(paper)
            continue

        if price is None:
            results.append({
                "ticker":        ticker,
                "status":        "skipped",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    0.0,
                "units":         0.0,
                "price_native":  0.0,
                "currency":      currency,
                "broker":        "dhan",
                "order_id":      None,
                "error":         "no_price",
            })
            continue

        inr_budget = order["monthly_usd"] * usd_inr
        quantity   = int(inr_budget / price)        # whole shares

        if quantity < 1:
            results.append({
                "ticker":        ticker,
                "status":        "skipped",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    0.0,
                "units":         0.0,
                "price_native":  price,
                "currency":      currency,
                "broker":        "dhan",
                "order_id":      None,
                "error":         f"quantity_zero (budget ₹{inr_budget:.2f} < price ₹{price:.2f})",
            })
            continue

        try:
            resp = dhan.place_order(
                security_id      = sec_id,
                exchange_segment = dhan.BSE,
                transaction_type = dhan.BUY,
                quantity         = quantity,
                order_type       = dhan.MARKET,
                product_type     = dhan.CNC,   # delivery (long-term hold)
                price            = 0,
            )
            order_id    = resp.get("data", {}).get("orderId", "")
            filled_inr  = quantity * price
            filled_usd  = filled_inr / usd_inr

            results.append({
                "ticker":        ticker,
                "status":        "filled",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    round(filled_usd, 2),
                "units":         quantity,
                "price_native":  price,
                "currency":      currency,
                "broker":        "dhan",
                "order_id":      str(order_id),
                "error":         None,
            })
        except Exception as exc:
            results.append({
                "ticker":        ticker,
                "status":        "error",
                "requested_usd": order["monthly_usd"],
                "filled_usd":    0.0,
                "units":         0.0,
                "price_native":  price,
                "currency":      currency,
                "broker":        "dhan",
                "order_id":      None,
                "error":         str(exc),
            })

    return results


# ── Node function ─────────────────────────────────────────────────────────────

def broker_connector_node(state: SIPExecutionState) -> dict:
    """
    Node 5 — Broker Connector

    Routes by ticker suffix:
      .L   → IBKR (ib_insync) — LSE ETFs, bought in GBP
      .BO  → DhanHQ           — BSE ETFs, bought in INR

    Falls back to paper mode when broker keys are absent or libraries missing.
    dry_run=True always uses paper mode regardless of keys.
    """
    proposed_orders = state["proposed_orders"]
    dry_run         = state["dry_run"]

    print(f"\n[Node 5] Broker Connector — {'DRY RUN' if dry_run else 'LIVE'} "
          f"mode, {len(proposed_orders)} orders …")

    tickers = [o["ticker"] for o in proposed_orders]
    prices  = _fetch_prices(tickers)
    usd_inr = _get_usd_inr()
    usd_gbp = _get_usd_gbp()

    print(f"  [Node 5] USD/INR = {usd_inr:.2f}  |  GBP/USD = {usd_gbp:.4f}")
    for t, p in prices.items():
        print(f"  [Node 5]   {t:<22}  {p}")

    if dry_run:
        results    = _execute_paper(proposed_orders, prices, usd_inr, usd_gbp)
        exec_status = "dry_run"
    else:
        # Split orders by broker route
        lse_orders  = [o for o in proposed_orders if o["ticker"].endswith(".L")]
        bse_orders  = [o for o in proposed_orders if o["ticker"].endswith(".BO")]
        other_orders = [
            o for o in proposed_orders
            if not o["ticker"].endswith(".L") and not o["ticker"].endswith(".BO")
        ]

        results: List[ExecutionResult] = []

        if lse_orders:
            print(f"  [Node 5] Routing {len(lse_orders)} LSE order(s) → IBKR")
            results.extend(_execute_ibkr(lse_orders, prices, usd_gbp))

        if bse_orders:
            print(f"  [Node 5] Routing {len(bse_orders)} BSE order(s) → DhanHQ")
            results.extend(_execute_dhan(bse_orders, prices, usd_inr))

        if other_orders:
            print(f"  [Node 5] {len(other_orders)} unrecognised ticker(s) → paper fallback")
            results.extend(_execute_paper(other_orders, prices, usd_inr, usd_gbp))

        filled  = sum(1 for r in results if r["status"] in ("filled", "dry_run"))
        exec_status = "success" if filled == len(results) else "partial"

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
