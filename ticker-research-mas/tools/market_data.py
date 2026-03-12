"""
Fetches historical prices, technicals, fundamentals, and index futures basis via yfinance.
"""

from typing import Any, Dict, Optional, Tuple

import pandas as pd
import yfinance as yf

# Maps stock sector → (front-month futures symbol, spot index symbol, index name)
# Nasdaq 100 futures for tech/comm; S&P 500 futures as the universal default.
_SECTOR_TO_FUTURES: Dict[str, Tuple[str, str, str]] = {
    "Technology": ("NQ=F", "^NDX", "Nasdaq 100"),
    "Communication Services": ("NQ=F", "^NDX", "Nasdaq 100"),
    "Consumer Cyclical": ("NQ=F", "^NDX", "Nasdaq 100"),
}
_DEFAULT_FUTURES: Tuple[str, str, str] = ("ES=F", "^GSPC", "S&P 500")


def _safe_round(val: Optional[float], n: int = 2) -> Optional[float]:
    if val is None or (isinstance(val, float) and pd.isna(val)):
        return None
    return round(float(val), n)


def get_market_data(ticker: str) -> Dict[str, Any]:
    """
    Fetches comprehensive market data for a stock, including index futures basis.
    All futures fields are None (gracefully) if the data is unavailable.
    """
    stock = yf.Ticker(ticker)
    info = stock.info

    # Soft guard — just check we have something usable
    if not info or not info.get("longName"):
        raise ValueError(f"No data found for ticker '{ticker}'. Check the symbol.")

    # ── Determine relevant futures contract by sector ─────────────────────
    sector = info.get("sector", "N/A")
    futures_symbol, spot_symbol, index_name = _SECTOR_TO_FUTURES.get(
        sector, _DEFAULT_FUTURES
    )

    # ── Stock history (6 months) ──────────────────────────────────────────
    hist = stock.history(period="6mo")
    if hist.empty:
        raise ValueError(f"No historical price data returned for '{ticker}'.")

    close = hist["Close"]
    volume = hist["Volume"]
    current_price = close.iloc[-1]
    price_6mo_ago = close.iloc[0]

    # ── RSI (14-period, Wilder EMA method) ───────────────────────────────
    delta = close.diff()
    avg_gain = delta.clip(lower=0).ewm(com=13, adjust=False).mean()
    avg_loss = (-delta.clip(upper=0)).ewm(com=13, adjust=False).mean()
    rs = avg_gain / avg_loss.replace(0, float("nan"))
    rsi = (100 - 100 / (1 + rs)).iloc[-1]

    # ── MACD (12/26/9) ───────────────────────────────────────────────────
    ema12 = close.ewm(span=12, adjust=False).mean()
    ema26 = close.ewm(span=26, adjust=False).mean()
    macd_line = ema12 - ema26
    signal_line = macd_line.ewm(span=9, adjust=False).mean()
    macd_histogram = macd_line - signal_line

    # ── Moving Averages ───────────────────────────────────────────────────
    ma20 = close.rolling(20).mean()
    ma50 = close.rolling(50).mean()
    ma200 = close.rolling(200).mean()

    # ── Bollinger Bands (20-period, 2σ) ──────────────────────────────────
    bb_mid = ma20.iloc[-1]
    bb_std = close.rolling(20).std().iloc[-1]
    bb_upper = bb_mid + 2 * bb_std
    bb_lower = bb_mid - 2 * bb_std
    bb_pos = (current_price - bb_lower) / (bb_upper - bb_lower) if (bb_upper - bb_lower) != 0 else float("nan")

    # ── Futures & Spot (60d history so we can compute 30-day volume avg) ─
    futures_price = spot_price = futures_basis = basis_pct = None
    futures_volume = futures_volume_avg30 = futures_volume_ratio = None
    basis_state = volatility_level = None

    try:
        f_hist = yf.Ticker(futures_symbol).history(period="60d")
        s_hist = yf.Ticker(spot_symbol).history(period="5d")

        if not f_hist.empty and not s_hist.empty:
            futures_price = float(f_hist["Close"].iloc[-1])
            spot_price = float(s_hist["Close"].iloc[-1])
            futures_basis = futures_price - spot_price
            basis_pct = (futures_basis / spot_price) * 100
            basis_state = "Contango" if futures_basis > 0 else "Backwardation"

            # Volume: compare today vs 30-day rolling average
            f_vol_series = f_hist["Volume"]
            futures_volume = int(f_vol_series.iloc[-1])
            futures_volume_avg30 = int(f_vol_series.rolling(30).mean().iloc[-1])
            futures_volume_ratio = round(
                futures_volume / futures_volume_avg30, 2
            ) if futures_volume_avg30 > 0 else 1.0

            # Volatility level driven by ratio vs 30-day average
            if futures_volume_ratio >= 2.0:
                volatility_level = "SPIKE"
            elif futures_volume_ratio >= 1.5:
                volatility_level = "ELEVATED"
            else:
                volatility_level = "NORMAL"

    except Exception as e:
        print(f"[market_data]  Futures fetch warning ({futures_symbol}): {e}")

    return {
        "ticker": ticker,
        "company_name": info.get("longName", ticker),
        "sector": sector,
        "industry": info.get("industry", "N/A"),

        # ── Anchor Signals ────────────────────────────────────────────────
        "futures_symbol": futures_symbol,
        "spot_symbol": spot_symbol,
        "index_name": index_name,
        "futures_price": _safe_round(futures_price),
        "spot_price": _safe_round(spot_price),
        "futures_basis": _safe_round(futures_basis),
        "basis_pct": _safe_round(basis_pct, 4),
        "basis_state": basis_state,                     # "Contango" | "Backwardation" | None
        "futures_volume": futures_volume,
        "futures_volume_avg30": futures_volume_avg30,
        "futures_volume_ratio": futures_volume_ratio,   # today / 30d avg
        "volatility_level": volatility_level,           # NORMAL | ELEVATED | SPIKE

        # ── Price & Range ─────────────────────────────────────────────────
        "current_price": _safe_round(current_price),
        "price_change_6mo_pct": _safe_round((current_price - price_6mo_ago) / price_6mo_ago * 100),
        "high_52w": info.get("fiftyTwoWeekHigh"),
        "low_52w": info.get("fiftyTwoWeekLow"),

        # ── Volume ────────────────────────────────────────────────────────
        "volume_latest": int(volume.iloc[-1]),
        "volume_avg_30d": int(volume.rolling(30).mean().iloc[-1]),

        # ── Trend ─────────────────────────────────────────────────────────
        "ma20": _safe_round(ma20.iloc[-1]),
        "ma50": _safe_round(ma50.iloc[-1]),
        "ma200": _safe_round(ma200.iloc[-1]),

        # ── Momentum ─────────────────────────────────────────────────────
        "rsi_14": _safe_round(rsi),
        "macd_bullish": bool(macd_line.iloc[-1] > signal_line.iloc[-1]),
        "macd_histogram": _safe_round(macd_histogram.iloc[-1], 4),
        "bb_position_pct": _safe_round(bb_pos * 100, 1),

        # ── Fundamentals ─────────────────────────────────────────────────
        "market_cap_b": _safe_round(info.get("marketCap", 0) / 1e9) if info.get("marketCap") else None,
        "pe_trailing": info.get("trailingPE"),
        "pe_forward": info.get("forwardPE"),
        "peg_ratio": info.get("pegRatio"),
        "price_to_book": info.get("priceToBook"),
        "price_to_sales": info.get("priceToSalesTrailing12Months"),
        "revenue_growth_yoy": info.get("revenueGrowth"),
        "earnings_growth_yoy": info.get("earningsQuarterlyGrowth"),
        "profit_margin": info.get("profitMargins"),
        "operating_margin": info.get("operatingMargins"),
        "roe": info.get("returnOnEquity"),
        "debt_to_equity": info.get("debtToEquity"),
        "free_cashflow_b": _safe_round(info.get("freeCashflow", 0) / 1e9) if info.get("freeCashflow") else None,
        "short_percent_float": info.get("shortPercentOfFloat"),
        "analyst_target_price": info.get("targetMeanPrice"),
        "analyst_recommendation": info.get("recommendationKey"),
    }
