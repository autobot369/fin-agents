"""
Buy-Only Historical SIP Backtest
==================================
Simulates a Sentiment-Weighted Dynamic SIP over the past N months using
actual market prices from yfinance.

Rules:
  - One installment per calendar month (first available trading day).
  - Units are accumulated — NO selling, NO rebalancing, NO partial exits.
  - BSE/NSE ETFs are priced in INR; USD allocation is converted at the
    prevailing USD/INR rate fetched at simulation time.
  - If price data is unavailable for a ticker in a given month, that
    month's allocation for that ticker is skipped (cash is not re-deployed).
"""

from __future__ import annotations

import warnings
from datetime import date, timedelta
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from simulator.state import AllocationPlan, ETFHolding, MonthlySnapshot, SimulationResult

warnings.filterwarnings("ignore")

# ── SSL fix for macOS Python (python.org installer) ───────────────────────────
def _patch_ssl() -> None:
    import os
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE",      certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass

_patch_ssl()


# ── Exchange rate ─────────────────────────────────────────────────────────────

def get_usd_inr_rate() -> float:
    """Fetch the current USD/INR exchange rate via yfinance."""
    try:
        hist = yf.Ticker("USDINR=X").history(period="5d", auto_adjust=True)
        if not hist.empty:
            return float(hist["Close"].iloc[-1])
    except Exception:
        pass
    print("  [backtest] USD/INR fetch failed — using fallback 83.50")
    return 83.50


# ── Price data helpers ────────────────────────────────────────────────────────

def _fetch_history(ticker: str, start: date, end: date) -> Optional[pd.DataFrame]:
    """Download daily adjusted-close history for a single ticker."""
    try:
        df = yf.download(
            ticker,
            start=start.isoformat(),
            end=(end + timedelta(days=1)).isoformat(),
            progress=False,
            auto_adjust=True,
        )
        if df.empty:
            return None
        # Flatten MultiIndex columns produced by newer yfinance versions
        if isinstance(df.columns, pd.MultiIndex):
            df.columns = df.columns.get_level_values(0)
        return df
    except Exception as exc:
        print(f"    [backtest] {ticker}: fetch error — {exc}")
        return None


def _price_on_or_after(df: pd.DataFrame, target: date) -> Optional[float]:
    """
    Return the Close price on the first trading day >= target.
    Falls back up to 4 business days before target if nothing found ahead.
    """
    if df is None or df.empty:
        return None

    idx_dates = pd.to_datetime(df.index).date

    # Try target date forward (up to 5 days — covers long weekends / holidays)
    for offset in range(6):
        check = target + timedelta(days=offset)
        mask  = idx_dates == check
        if mask.any():
            return float(df.loc[mask, "Close"].iloc[0])

    # Last resort: most recent price before target (end-of-prior-month)
    before = df[idx_dates < target]
    if not before.empty:
        return float(before["Close"].iloc[-1])
    return None


# ── Month utilities ───────────────────────────────────────────────────────────

def _go_back_months(from_date: date, months: int) -> date:
    """First day of the month that is `months` calendar months before from_date."""
    total = from_date.year * 12 + (from_date.month - 1) - months
    return date(total // 12, total % 12 + 1, 1)


def _month_starts(start: date, end: date) -> List[date]:
    """List of first-of-month dates from start (inclusive) to end (exclusive)."""
    result, y, m = [], start.year, start.month
    while True:
        d = date(y, m, 1)
        if d >= end:
            break
        result.append(d)
        m += 1
        if m > 12:
            m, y = 1, y + 1
    return result


# ── Main simulation ───────────────────────────────────────────────────────────

def run_backtest(
    plan: AllocationPlan,
    months: int = 12,
    usd_inr_rate: float = 83.50,
) -> SimulationResult:
    """
    Run a buy-only historical SIP simulation.

    For each month in [today - months, today):
      • Find the first available trading-day closing price for each ETF.
      • Convert USD allocation to INR for BSE ETFs using usd_inr_rate.
      • Buy fractional units = allocation / price.
      • Accumulate units — never sell.

    Returns a SimulationResult with per-ETF holdings and portfolio metrics.
    """
    today    = date.today()
    start    = _go_back_months(today, months)
    allocs   = plan["all"]
    alloc_by = {a["ticker"]: a for a in allocs}
    tickers  = [a["ticker"] for a in allocs]

    # ── Fetch price histories (with a 7-day buffer before start) ─────────────
    buf_start = start - timedelta(days=7)
    print(f"  [backtest] Fetching price history for {len(tickers)} ETFs …")
    price_data: Dict[str, Optional[pd.DataFrame]] = {}
    for t in tickers:
        df = _fetch_history(t, buf_start, today)
        price_data[t] = df
        status = "OK" if df is not None else "NO DATA"
        print(f"    {t:<22}  {status}")

    skipped = [t for t, df in price_data.items() if df is None]

    # ── SIP loop ─────────────────────────────────────────────────────────────
    month_list           = _month_starts(start, today)
    units_held: Dict[str, float] = {t: 0.0 for t in tickers}
    invested:   Dict[str, float] = {t: 0.0 for t in tickers}
    snapshots:  List[MonthlySnapshot] = []
    total_invested_usd   = 0.0

    print(f"  [backtest] Simulating {len(month_list)} monthly instalments (buy-only) …")

    for month_date in month_list:
        units_bought: Dict[str, float] = {}
        price_paid:   Dict[str, float] = {}
        month_usd = 0.0

        for ticker in tickers:
            alloc      = alloc_by[ticker]
            monthly_usd = alloc["monthly_usd"]
            df          = price_data.get(ticker)
            price       = _price_on_or_after(df, month_date)

            if price is None or price <= 0:
                continue                          # skip — no data this month

            if alloc["region"] == "BSE":
                # Convert USD → INR, buy at INR price
                units = (monthly_usd * usd_inr_rate) / price
            else:
                units = monthly_usd / price

            units_held[ticker] += units
            invested[ticker]   += monthly_usd
            units_bought[ticker] = round(units, 6)
            price_paid[ticker]   = round(price, 4)
            month_usd           += monthly_usd

        total_invested_usd += month_usd
        snapshots.append(MonthlySnapshot(
            date          = month_date.isoformat(),
            units_bought  = units_bought,
            price_paid    = price_paid,
            usd_invested  = round(month_usd, 2),
        ))

    # ── Compute current value ─────────────────────────────────────────────────
    holdings: Dict[str, ETFHolding] = {}
    current_value_usd = 0.0

    for ticker in tickers:
        df = price_data.get(ticker)
        if df is None or df.empty or units_held[ticker] == 0:
            continue

        current_price = float(df["Close"].iloc[-1])
        units         = units_held[ticker]
        alloc         = alloc_by[ticker]

        if alloc["region"] == "BSE":
            value_usd = (units * current_price) / usd_inr_rate
        else:
            value_usd = units * current_price

        inv   = invested[ticker]
        pnl   = value_usd - inv
        ret   = (pnl / inv * 100) if inv > 0 else 0.0

        holdings[ticker] = ETFHolding(
            ticker               = ticker,
            bucket               = alloc["bucket"],
            region               = alloc["region"],
            total_units          = round(units, 6),
            current_price_native = round(current_price, 4),
            value_usd            = round(value_usd, 2),
            invested_usd         = round(inv, 2),
            pnl_usd              = round(pnl, 2),
            return_pct           = round(ret, 2),
        )
        current_value_usd += value_usd

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_pnl    = current_value_usd - total_invested_usd
    total_ret    = (total_pnl / total_invested_usd * 100) if total_invested_usd > 0 else 0.0
    years        = len(month_list) / 12
    cagr         = (
        ((current_value_usd / total_invested_usd) ** (1 / years) - 1) * 100
        if total_invested_usd > 0 and years > 0
        else 0.0
    )

    return SimulationResult(
        total_invested_usd = round(total_invested_usd, 2),
        current_value_usd  = round(current_value_usd, 2),
        total_pnl_usd      = round(total_pnl, 2),
        total_return_pct   = round(total_ret, 2),
        cagr               = round(cagr, 2),
        months_simulated   = len(snapshots),
        usd_inr_rate       = usd_inr_rate,
        monthly_snapshots  = snapshots,
        holdings           = holdings,
        skipped_tickers    = skipped,
    )
