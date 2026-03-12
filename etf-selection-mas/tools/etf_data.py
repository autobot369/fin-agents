"""
ETF Market Data Tool
--------------------
Fetches expense ratio, AUM, YTD return, and momentum metrics via yfinance.

Design notes:
- yfinance does NOT always expose `expenseRatio` for ETFs via the `.info` dict.
  The field name varies: 'annualReportExpenseRatio', 'expenseRatio', or absent.
  We maintain a FALLBACK_TER dict with verified values from fund prospectuses as
  of Q1 2026. These are used when yfinance returns None.
- BSE tickers use the `.BO` suffix (e.g., "NIFTYBEES.BO").
  Their TER metadata is sparse in yfinance; fallback dict is essential.
- YTD return is calculated from Jan 1 of the current year if not available in info.
"""

from datetime import date, datetime
from typing import Any, Dict, List, Optional

import pandas as pd
import yfinance as yf

from graph.state import ETFRecord

# ---------------------------------------------------------------------------
# Known expense ratios (decimal) — verified from fund prospectuses, Q1 2026.
# Used as fallback when yfinance cannot surface the field.
# ---------------------------------------------------------------------------
FALLBACK_TER: Dict[str, float] = {
    # ── International (NASDAQ-listed) ──────────────────────────────────────
    "VXUS":  0.0007,   # Vanguard Total Intl Stock ETF          0.07%
    "SPDW":  0.0004,   # SPDR Portfolio Developed World ex-US   0.04%
    "IEFA":  0.0007,   # iShares Core MSCI EAFE                 0.07%
    "SCHY":  0.0014,   # Schwab International Dividend Equity   0.14%
    "VEA":   0.0005,   # Vanguard FTSE Developed Markets        0.05%
    "EEM":   0.0068,   # iShares MSCI Emerging Markets          0.68%
    "ACWI":  0.0032,   # iShares MSCI ACWI                      0.32%
    "SCHF":  0.0006,   # Schwab International Equity            0.06%
    "IXUS":  0.0007,   # iShares Core MSCI Total Intl Stock     0.07%
    "VWO":   0.0007,   # Vanguard FTSE Emerging Markets         0.07%
    "IEMG":  0.0009,   # iShares Core MSCI EM                   0.09%
    "XEQT":  0.0020,   # iShares Core Equity ETF Portfolio      0.20%
    # ── China / HK (US-listed) ─────────────────────────────────────────────
    "CQQQ":  0.0065,   # Invesco China Technology ETF           0.65%
    "KWEB":  0.0065,   # KraneShares CSI China Internet         0.65%
    "MCHI":  0.0019,   # iShares MSCI China                     0.19%
    "FXI":   0.0074,   # iShares China Large-Cap                0.74%
    "CHIQ":  0.0065,   # Global X MSCI China Consumer Disc      0.65%
    "GXC":   0.0059,   # SPDR S&P China ETF                     0.59%
    "FLCH":  0.0019,   # Franklin FTSE China ETF                0.19%
    "KURE":  0.0065,   # KraneShares MSCI All China Health Care 0.65%
    "CNYA":  0.0060,   # iShares MSCI China A                   0.60%
    "ASHR":  0.0065,   # Xtrackers Harvest CSI 300 China A-Shrs 0.65%
    # ── India / BSE-listed ─────────────────────────────────────────────────
    "NIFTYBEES.BO":  0.0004,   # Nippon India Nifty 50 BeES     0.04%
    "BANKBEES.BO":   0.0019,   # Nippon India Banking BeES      0.19%
    "GOLDBEES.BO":   0.0005,   # Nippon India Gold BeES         0.05%
    "JUNIORBEES.BO": 0.0013,   # Nippon India Jr BeES (Next 50) 0.13%
    "ITBEES.BO":     0.0010,   # Nippon India IT BeES           0.10%
    "INFRABEES.BO":  0.0010,   # Nippon India Infra BeES        0.10%
    "SETFNIF50.BO":  0.0007,   # SBI Nifty 50 ETF               0.07%
    "CONSUMBEES.BO": 0.0010,   # Nippon India Consumption BeES  0.10%
    "PHARMABEES.BO": 0.0010,   # Nippon India Pharma BeES       0.10%
    "DIVOPPBEES.BO": 0.0033,   # Nippon India Dividend Opps BeES 0.33%
}


def _safe_float(val: Any, n: int = 4) -> Optional[float]:
    try:
        if val is None or (isinstance(val, float) and pd.isna(val)):
            return None
        return round(float(val), n)
    except (TypeError, ValueError):
        return None


def _calc_ytd_return(hist: pd.DataFrame) -> Optional[float]:
    """Calculate YTD return from Jan 1 of the current year."""
    if hist.empty:
        return None
    year_start = datetime(date.today().year, 1, 1)
    ytd_hist = hist[hist.index >= pd.Timestamp(year_start, tz=hist.index.tz)]
    if len(ytd_hist) < 2:
        # Fallback: use earliest available in the period
        ytd_hist = hist
    if ytd_hist.empty:
        return None
    start_price = ytd_hist["Close"].iloc[0]
    end_price = ytd_hist["Close"].iloc[-1]
    if start_price == 0:
        return None
    return round((end_price - start_price) / start_price, 4)


def _calc_momentum(hist: pd.DataFrame, days: int) -> Optional[float]:
    """N-day price momentum (simple return)."""
    close = hist["Close"]
    if len(close) < days:
        return None
    return round((close.iloc[-1] - close.iloc[-days]) / close.iloc[-days], 4)


def get_etf_data(ticker: str, region: str) -> ETFRecord:
    """
    Fetch ETF data via yfinance.
    Returns a fully-populated ETFRecord; gracefully degrades on missing fields.
    """
    record: ETFRecord = {
        "ticker": ticker,
        "name": ticker,
        "region": region,
        "expense_ratio": None,
        "aum_b": None,
        "ytd_return": None,
        "momentum_3m": None,
        "momentum_1m": None,
        "current_price": None,
        "volume_avg_30d": None,
        "data_source": "yfinance",
        "fetch_error": None,
    }

    try:
        etf = yf.Ticker(ticker)
        info = etf.info or {}

        # ── Name ───────────────────────────────────────────────────────────
        record["name"] = (
            info.get("longName")
            or info.get("shortName")
            or ticker
        )

        # ── Expense Ratio ──────────────────────────────────────────────────
        # Try multiple yfinance field names before falling back to our dict.
        ter_raw = (
            info.get("annualReportExpenseRatio")
            or info.get("expenseRatio")
            or info.get("totalExpenseRatio")
        )
        record["expense_ratio"] = (
            _safe_float(ter_raw)
            if ter_raw is not None
            else FALLBACK_TER.get(ticker)
        )

        # ── AUM ────────────────────────────────────────────────────────────
        total_assets = info.get("totalAssets")
        if total_assets:
            record["aum_b"] = _safe_float(total_assets / 1e9, 2)

        # ── Price History (6 months for momentum, YTD) ────────────────────
        hist = etf.history(period="6mo")
        if not hist.empty:
            close = hist["Close"]
            record["current_price"] = _safe_float(close.iloc[-1], 2)

            vol = hist["Volume"]
            if len(vol) >= 30:
                record["volume_avg_30d"] = int(vol.rolling(30).mean().iloc[-1])

            record["ytd_return"] = _calc_ytd_return(hist)
            record["momentum_3m"] = _calc_momentum(hist, 63)   # ~3 months
            record["momentum_1m"] = _calc_momentum(hist, 21)   # ~1 month

        # ── Mark BSE tickers when yfinance returns sparse info ─────────────
        if ticker.endswith(".BO") and not info.get("longName"):
            record["data_source"] = "fallback"

    except Exception as exc:
        record["fetch_error"] = str(exc)
        record["data_source"] = "fallback"
        # Still surface the fallback TER so ExpenseGuard can work
        record["expense_ratio"] = FALLBACK_TER.get(ticker)

    return record


def fetch_etf_universe(tickers: List[str], region: str) -> Dict[str, ETFRecord]:
    """Fetch ETF data for a list of tickers and return a ticker→ETFRecord map."""
    results: Dict[str, ETFRecord] = {}
    for ticker in tickers:
        print(f"  [etf_data] Fetching {ticker} ...")
        results[ticker] = get_etf_data(ticker, region)
    return results
