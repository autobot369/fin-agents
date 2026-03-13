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

from simulator.state import (
    AllocationPlan, ETFHolding, MonthlySnapshot, SimulationResult,
    BacktestPosition, BacktestMonthEntry, HistoricalBacktestResult,
)

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


# ── Month-by-month historical backtest ────────────────────────────────────────

def run_historical_backtest(
    start_date: date,
    end_date: date,
    sip_amount: float = 500.0,
    day_of_month: int = 1,
    top_n: int = 10,
    core_count: int = 5,
    core_pct: float = 0.70,
    ter_threshold: float = 0.007,       # decimal — e.g. 0.007 = 0.70%
    usd_inr_rate: float = 83.50,
    progress_callback=None,
) -> HistoricalBacktestResult:
    """
    Month-by-month historical SIP backtest with per-month ETF research.

    For each calendar month in [start_date, end_date):
      1. Gemini discovers the ETF universe (date-isolated prompt — no future events).
      2. yfinance fetches metrics bounded to as_of_date.
      3. News is filtered to articles published on or before as_of_date.
      4. Gemini (or VADER fallback) scores sentiment for that date only.
      5. Allocation is computed and units are bought at the historical price.

    Args:
        start_date:        First month to invest (inclusive).
        end_date:          Stop before this date (exclusive).
        sip_amount:        Monthly SIP budget in USD.
        top_n:             Total ETFs to invest in (core + satellite).
        core_count:        Number of top ETFs in the core bucket.
        core_pct:          Fraction of SIP allocated to core (e.g. 0.70).
        ter_threshold:     Max expense ratio as decimal (0.007 = 0.70%).
        usd_inr_rate:      USD/INR rate used for BSE ticker conversions.
        progress_callback: Optional callable(msg: str) for progress updates.

    Returns:
        HistoricalBacktestResult with per-month entries and aggregate holdings.
    """
    import calendar as _cal
    from simulator.allocator import compute_allocation
    from sip_execution_mas.agents.regional_researcher import (
        _build_india_tier, _build_us_tier, _build_hk_proxy_tier,
        _fetch_yfinance_batch, _fetch_news, _FALLBACK,
        _adv_usd, _ADV_MIN_USD, _recommend_broker, _calc_entry_cost,
    )
    from sip_execution_mas.agents.signal_scorer import score_etfs

    def _log(msg: str) -> None:
        print(msg)
        if progress_callback:
            progress_callback(msg)

    month_list = _month_starts(start_date, end_date)
    _log("[backtest] {} months to simulate: {} → {}".format(
        len(month_list), start_date.isoformat(), end_date.isoformat(),
    ))

    # Pre-seed the price cache with the fallback universe so common tickers
    # don't need repeated downloads across months. Any Gemini-discovered ticker
    # not in the fallback is fetched lazily on first encounter.
    buf_start = start_date - timedelta(days=10)
    fallback_tickers = [r["ticker"] for recs in _FALLBACK.values() for r in recs]
    _log("[backtest] Pre-fetching price history for {} seed ETFs …".format(len(fallback_tickers)))
    price_cache: dict = {}
    for t in fallback_tickers:
        price_cache[t] = _fetch_history(t, buf_start, date.today())
    _log("[backtest] Price cache ready — starting month loop …\n")

    # ── Month loop ────────────────────────────────────────────────────────────
    units_held:      dict = {}   # ticker → cumulative units
    invested:        dict = {}   # ticker → cumulative USD invested
    last_alloc_meta: dict = {}   # ticker → latest ETFAllocation metadata
    monthly_entries = []
    skipped_months  = []
    total_invested_usd = 0.0

    for month_date in month_list:
        month_str = month_date.strftime("%Y-%m")
        # Clamp day_of_month to the last day of the month (handles Feb, 30-day months)
        last_day  = _cal.monthrange(month_date.year, month_date.month)[1]
        buy_date  = date(month_date.year, month_date.month, min(day_of_month, last_day))
        _log("[backtest] ── {} (buy date: {}) ──────────────────────────────────────".format(
            month_str, buy_date.isoformat()
        ))

        # ── Phase 1: Gemini discovers universe as of buy_date ────────────────
        try:
            india_recs = _build_india_tier(ter_threshold, as_of_date=buy_date)
            us_recs    = _build_us_tier(as_of_date=month_date)
            hk_recs    = _build_hk_proxy_tier(as_of_date=month_date)

            market_recs = {"BSE": india_recs, "US": us_recs, "HKCN": hk_recs}
            all_tickers = [r["ticker"] for recs in market_recs.values() for r in recs]

            # ── Phase 2: yfinance enrichment (date-bounded) ───────────────────
            raw        = _fetch_yfinance_batch(all_tickers, as_of_date=buy_date)
            macro_news = _fetch_news(
                {m: [r["ticker"] for r in recs] for m, recs in market_recs.items()},
                as_of_date=buy_date,
            )

            # ── Phase 3: ETFRecord + TER / liquidity filter ───────────────────
            all_etf_data: dict     = {}
            filtered_tickers: list = []

            for market, recs in market_recs.items():
                for rec in recs:
                    ticker      = rec["ticker"]
                    r           = raw.get(ticker, {})
                    error       = r.get("error")
                    is_hk_local = ticker.endswith(".HK")
                    is_proxy    = rec.get("is_proxy", not is_hk_local and market == "HKCN")

                    ter = r.get("ter") if not error else None
                    if ter is None and rec.get("ter_pct"):
                        ter = rec["ter_pct"] / 100.0

                    price   = r.get("price") if not error else None
                    aum_b   = (r.get("aum_b") if not error else None) or rec.get("aum_b_est")
                    avg_vol = r.get("avg_vol") if not error else None
                    adv     = _adv_usd(ticker, avg_vol, price)
                    liq_ok  = (adv is None) or (adv >= _ADV_MIN_USD)

                    broker     = _recommend_broker(ticker, is_hkex_local=is_hk_local)
                    entry_cost = (
                        _calc_entry_cost(
                            expense_ratio=ter if ter is not None else 0.005,
                            broker=broker,
                            is_hk_local=is_hk_local,
                        ) if ter is not None else None
                    )

                    currency = "INR" if market == "BSE" else ("HKD" if is_hk_local else "USD")
                    category = rec.get("category", rec.get("theme", "ETF"))
                    if is_hk_local:
                        category = f"{category} [HKD — FX conversion required]"
                    elif currency == "INR":
                        category = f"{category} [INR — FX via Dhan]"

                    all_etf_data[ticker] = {
                        "ticker":        ticker,
                        "name":          rec["name"],
                        "region":        market,
                        "market":        "NSE" if market == "BSE" else ("HKEX" if is_hk_local else "NASDAQ"),
                        "category":      category,
                        "expense_ratio": ter,
                        "aum_b":         aum_b,
                        "ytd_return":    r.get("ytd") if not error else None,
                        "momentum_3m":   r.get("mom3m") if not error else None,
                        "momentum_1m":   r.get("mom1m") if not error else None,
                        "current_price": price,
                        "currency":      currency,
                        "is_proxy":      is_proxy,
                        "proxy_for":     rec.get("proxy_for"),
                    }

                    if not liq_ok or (ter is not None and ter > ter_threshold):
                        continue
                    filtered_tickers.append(ticker)

            if not filtered_tickers:
                _log("[backtest] {} no ETFs after filtering — skipping".format(month_str))

                skipped_months.append(month_str)
                continue

            # ── Phase 4: Score via Gemini / VADER (date-isolated) ─────────────
            sentiment_scores, boom_triggers, macro_summary = score_etfs(
                filtered_tickers=filtered_tickers,
                all_etf_data=all_etf_data,
                all_macro_news=macro_news,
                reference_date=buy_date,
            )

            # ── Phase 5: Build ranked list for allocator ──────────────────────
            rankings: list = []
            for ticker in filtered_tickers:
                etf           = all_etf_data[ticker]
                ter           = etf.get("expense_ratio")
                expense_score = max(0.0, 1.0 - ter / ter_threshold) if ter is not None else 0.50
                sentiment     = sentiment_scores.get(ticker, 0.50)
                consensus     = round(0.60 * sentiment + 0.40 * expense_score, 4)
                region        = etf["region"]

                rankings.append({
                    "ticker":              ticker,
                    "name":                etf["name"],
                    "region":              region,
                    "market_label":        "NSE" if region == "BSE" else "NASDAQ",
                    "category":            etf.get("category", "—"),
                    "trade_on":            "Dhan" if region == "BSE" else "Alpaca",
                    "expense_ratio":       ter,
                    "momentum_3m":         etf.get("momentum_3m"),
                    "sentiment_score":     round(sentiment, 4),
                    "expense_score":       round(expense_score, 4),
                    "consensus_score":     consensus,
                    "sentiment_rationale": macro_summary[:120],
                })
            rankings.sort(key=lambda x: x["consensus_score"], reverse=True)

        except Exception as exc:
            _log("[backtest] {} research failed ({}) — skipping".format(month_str, exc))
            skipped_months.append(month_str)
            continue

        # Lazily populate price cache for any newly discovered tickers
        new_tickers = [r["ticker"] for r in rankings if r["ticker"] not in price_cache]
        for t in new_tickers:
            price_cache[t] = _fetch_history(t, buf_start, date.today())

        # ── Allocation ────────────────────────────────────────────────────────
        plan = compute_allocation(
            rankings   = rankings,
            sip_amount = sip_amount,
            top_n      = top_n,
            core_count = core_count,
            core_pct   = core_pct,
        )

        # ── Buy each ETF at the first available price on/after month_date ─────
        positions: list = []
        month_usd = 0.0

        for alloc in plan["all"]:
            ticker = alloc["ticker"]
            df     = price_cache.get(ticker)
            if df is not None and not df.empty:
                df_as_of = df[pd.to_datetime(df.index).date <= buy_date + timedelta(days=6)]
            else:
                df_as_of = df

            buy_price = _price_on_or_after(df_as_of, buy_date) if df_as_of is not None else None
            if buy_price is None or buy_price <= 0:
                _log("  [backtest] {} {}: no price data — skipped".format(month_str, ticker))
                continue

            monthly_usd = alloc["monthly_usd"]
            if alloc["region"] == "BSE":
                units = (monthly_usd * usd_inr_rate) / buy_price
            else:
                units = monthly_usd / buy_price

            units_held[ticker]      = units_held.get(ticker, 0.0) + units
            invested[ticker]        = invested.get(ticker, 0.0) + monthly_usd
            last_alloc_meta[ticker] = alloc
            month_usd              += monthly_usd

            positions.append(BacktestPosition(
                ticker          = ticker,
                name            = alloc["name"],
                bucket          = alloc["bucket"],
                region          = alloc["region"],
                category        = alloc["category"],
                consensus_score = alloc["consensus_score"],
                sentiment_score = alloc["sentiment_score"],
                expense_score   = alloc["expense_score"],
                weight          = alloc["weight"],
                monthly_usd     = round(monthly_usd, 2),
                price_native    = round(buy_price, 4),
                units_bought    = round(units, 6),
                currency        = alloc["currency"],
            ))

        total_invested_usd += month_usd
        monthly_entries.append(BacktestMonthEntry(
            month               = month_str,
            date                = buy_date.isoformat(),
            sip_amount          = sip_amount,
            core_budget         = plan["core_budget"],
            satellite_budget    = plan["satellite_budget"],
            total_invested_usd  = round(month_usd, 2),
            positions           = positions,
            boom_triggers       = boom_triggers,
            macro_summary       = macro_summary,
            scorer              = "gemini",
            usd_inr_rate        = usd_inr_rate,
        ))
        _log("[backtest] {} done: {} positions, ${:.2f} invested\n".format(
            month_str, len(positions), month_usd,
        ))

    # ── Compute current value using latest prices ─────────────────────────────
    _log("[backtest] Computing current portfolio value …")
    holdings: dict = {}
    current_value_usd = 0.0

    for ticker, units in units_held.items():
        if units <= 0:
            continue
        df = price_cache.get(ticker)
        if df is None or df.empty:
            continue
        current_price = float(df["Close"].iloc[-1])
        alloc_meta    = last_alloc_meta.get(ticker, {})
        region        = alloc_meta.get("region", "US")

        if region == "BSE":
            value_usd = (units * current_price) / usd_inr_rate
        else:
            value_usd = units * current_price

        inv = invested.get(ticker, 0.0)
        pnl = value_usd - inv
        ret = (pnl / inv * 100) if inv > 0 else 0.0

        holdings[ticker] = {
            "ticker":               ticker,
            "name":                 alloc_meta.get("name", ticker),
            "bucket":               alloc_meta.get("bucket", "?"),
            "region":               region,
            "category":             alloc_meta.get("category", "—"),
            "currency":             alloc_meta.get("currency", "USD"),
            "total_units":          round(units, 6),
            "current_price_native": round(current_price, 4),
            "value_usd":            round(value_usd, 2),
            "invested_usd":         round(inv, 2),
            "pnl_usd":              round(pnl, 2),
            "return_pct":           round(ret, 2),
        }
        current_value_usd += value_usd

    # ── Summary metrics ───────────────────────────────────────────────────────
    total_pnl  = current_value_usd - total_invested_usd
    total_ret  = (total_pnl / total_invested_usd * 100) if total_invested_usd > 0 else 0.0
    years      = len(monthly_entries) / 12
    cagr       = (
        ((current_value_usd / total_invested_usd) ** (1 / years) - 1) * 100
        if years > 0 and total_invested_usd > 0 and current_value_usd > 0
        else 0.0
    )

    _log("[backtest] Complete: {}/{} months run, ${:.2f} invested, ${:.2f} value, {:.2f}% return".format(
        len(monthly_entries), len(month_list),
        total_invested_usd, current_value_usd, total_ret,
    ))

    return HistoricalBacktestResult(
        start_date         = start_date.isoformat(),
        end_date           = end_date.isoformat(),
        months_run         = len(monthly_entries),
        sip_amount         = sip_amount,
        total_invested_usd = round(total_invested_usd, 2),
        current_value_usd  = round(current_value_usd, 2),
        total_pnl_usd      = round(total_pnl, 2),
        total_return_pct   = round(total_ret, 2),
        cagr               = round(cagr, 2),
        usd_inr_rate       = usd_inr_rate,
        monthly_entries    = monthly_entries,
        holdings           = holdings,
        skipped_months     = skipped_months,
    )
