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

import time
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

def _fixed_bucket_alloc(
    tickers: List[str],
    consensus_scores: dict,
    sip_amount: float,
    all_etf_data: dict,
    sentiment_scores: dict,
    expense_scores: dict,
    macro_summary: str,
) -> List[dict]:
    """
    Fixed 70/30 Core/Satellite allocation using pseudo-Sharpe consensus scores.
    All 14 locked tickers always receive a position — none are dropped.
    """
    from sip_execution_mas.agents.regional_researcher import (
        _CORE_UNIVERSE_TICKERS, _SATELLITE_UNIVERSE_TICKERS,
    )

    core_budget      = round(sip_amount * 0.70, 2)
    satellite_budget = round(sip_amount * 0.30, 2)

    def _alloc_bucket(bucket_tickers, budget, bucket_name):
        total = sum(max(consensus_scores.get(t, 0.01), 0.01) for t in bucket_tickers) or 1.0
        allocs = []
        for t in bucket_tickers:
            etf    = all_etf_data.get(t, {})
            score  = max(consensus_scores.get(t, 0.01), 0.01)
            usd    = round(budget * score / total, 2)
            region = etf.get("region", "US")
            allocs.append({
                "ticker":          t,
                "name":            etf.get("name", t),
                "bucket":          bucket_name,
                "region":          region,
                "market":          etf.get("market", "NSE" if region == "BSE" else "NASDAQ"),
                "category":        etf.get("category", "—"),
                "currency":        "INR" if region == "BSE" else "USD",
                "consensus_score": round(score / total, 4),
                "sentiment_score": round(sentiment_scores.get(t, 0.50), 4),
                "expense_score":   round(expense_scores.get(t, 0.50), 4),
                "monthly_usd":     usd,
                "weight":          round(usd / sip_amount, 5),
                "sentiment_rationale": macro_summary[:120],
            })
        # Snap rounding drift to top ticker in bucket
        drift = round(budget - sum(a["monthly_usd"] for a in allocs), 2)
        if drift != 0 and allocs:
            allocs[0]["monthly_usd"] = round(allocs[0]["monthly_usd"] + drift, 2)
            allocs[0]["weight"]      = round(allocs[0]["monthly_usd"] / sip_amount, 5)
        return allocs

    core_tickers      = [t for t in tickers if t in _CORE_UNIVERSE_TICKERS]
    satellite_tickers = [t for t in tickers if t in _SATELLITE_UNIVERSE_TICKERS]

    allocs  = _alloc_bucket(core_tickers,      core_budget,      "core")
    allocs += _alloc_bucket(satellite_tickers, satellite_budget, "satellite")
    return allocs


def run_historical_backtest(
    start_date: date,
    end_date: date,
    sip_amount: float = 500.0,
    day_of_month: int = 1,
    ter_threshold: float = 0.007,       # decimal — used in expense_score formula
    usd_inr_rate: float = 83.50,
    progress_callback=None,
    gemini_ratelimit_delay: float = 5.0,
    use_llm: bool = True,
) -> HistoricalBacktestResult:
    """
    Month-by-month historical SIP backtest with locked 14-ETF universe.

    For each calendar month in [start_date, end_date):
      1. Uses the locked 14-ETF universe (no discovery — same tickers every month).
      2. yfinance fetches metrics bounded to as_of_date.
      3. News is filtered to articles published on or before as_of_date.
      4. Gemini (or VADER fallback) scores sentiment for that date only.
         Sentiment rotates satellite weights — core weights are stable.
      5. Crash-Accumulator VA: if panic + negative momentum, effective SIP
         scales by 1.20× (Tier 1) or 1.50× (Tier 2) — mirrors Node 4 logic.
      6. Fixed 70/30 bucket allocation using effective_sip.
      7. Hard rules 1/2/3/5 validated (no duplicate-month check in backtest).
      8. Units are bought at the historical price for that month.

    Args:
        start_date:              First month to invest (inclusive).
        end_date:                Stop before this date (exclusive).
        sip_amount:              Base monthly SIP budget in USD (before VA scaling).
        ter_threshold:           Expense ratio ceiling used in expense_score formula
                                 (decimal, e.g. 0.007 = 0.70%).
        usd_inr_rate:            USD/INR rate for BSE ticker conversions.
        progress_callback:       Optional callable(msg: str) for progress updates.
        gemini_ratelimit_delay:  Seconds to sleep after each Gemini call to stay
                                 within API rate limits (default 5.0 s ≈ 12 RPM).
                                 Set to 0 to disable. Ignored when use_llm=False.
        use_llm:                 True (default) → use Gemini for sentiment scoring.
                                 False → use VADER only (faster, no API calls).

    Returns:
        HistoricalBacktestResult with per-month entries and aggregate holdings.
    """
    import calendar as _cal
    from sip_execution_mas.agents.regional_researcher import (
        _LOCKED_UNIVERSE,
        _fetch_yfinance_batch, _fetch_news,
        _recommend_broker, _adv_usd,
    )
    from sip_execution_mas.agents.signal_scorer import score_etfs
    from sip_execution_mas.agents.risk_auditor import (
        _detect_va_condition,
        VA_MULTIPLIER_TIER1, VA_MULTIPLIER_TIER2,
    )

    def _log(msg: str) -> None:
        print(msg)
        if progress_callback:
            progress_callback(msg)

    month_list = _month_starts(start_date, end_date)
    _log("[backtest] {} months to simulate: {} → {}".format(
        len(month_list), start_date.isoformat(), end_date.isoformat(),
    ))

    # Pre-seed the price cache with all 14 locked tickers
    buf_start      = start_date - timedelta(days=10)
    locked_tickers = [r["ticker"] for r in _LOCKED_UNIVERSE]
    _log("[backtest] Pre-fetching price history for {} locked ETFs …".format(len(locked_tickers)))
    price_cache: dict = {}
    for t in locked_tickers:
        price_cache[t] = _fetch_history(t, buf_start, date.today())
    _log("[backtest] Price cache ready — starting month loop …\n")

    # Group locked universe by region for ETFRecord building
    _market_recs: dict = {}
    for entry in _LOCKED_UNIVERSE:
        _market_recs.setdefault(entry["region"], []).append(entry)

    # ── Month loop ────────────────────────────────────────────────────────────
    units_held:      dict = {}   # ticker → cumulative units
    invested:        dict = {}   # ticker → cumulative USD invested
    last_alloc_meta: dict = {}   # ticker → latest allocation metadata
    monthly_entries = []
    skipped_months  = []
    total_invested_usd = 0.0

    for month_date in month_list:
        month_str = month_date.strftime("%Y-%m")
        last_day  = _cal.monthrange(month_date.year, month_date.month)[1]
        buy_date  = date(month_date.year, month_date.month, min(day_of_month, last_day))
        _log("[backtest] ── {} (buy date: {}) ──────────────────────────────────────".format(
            month_str, buy_date.isoformat()
        ))

        try:
            # ── Phase 1: Build ETFRecord for all 14 locked ETFs ──────────────
            raw        = _fetch_yfinance_batch(locked_tickers, as_of_date=buy_date)
            macro_news = _fetch_news(as_of_date=buy_date)

            all_etf_data: dict = {}
            for market, recs in _market_recs.items():
                for rec in recs:
                    ticker  = rec["ticker"]
                    r       = raw.get(ticker, {})
                    error   = r.get("error")

                    ter = r.get("ter") if not error else None
                    if ter is None and rec.get("ter_pct"):
                        ter = rec["ter_pct"] / 100.0

                    price   = r.get("price")   if not error else None
                    avg_vol = r.get("avg_vol") if not error else None
                    aum_b   = r.get("aum_b")   if not error else None
                    adv     = _adv_usd(ticker, avg_vol, price)
                    broker  = _recommend_broker(ticker)
                    currency = "INR" if market == "BSE" else "USD"
                    category = rec.get("category", "ETF")
                    if currency == "INR":
                        category = f"{category} [INR — via Dhan]"

                    all_etf_data[ticker] = {
                        "ticker":                 ticker,
                        "name":                   rec["name"],
                        "region":                 market,
                        "market":                 rec.get("market", "NSE" if market == "BSE" else "NASDAQ"),
                        "category":               category,
                        "expense_ratio":          ter,
                        "aum_b":                  aum_b,
                        "ytd_return":             r.get("ytd")           if not error else None,
                        "momentum_3m":            r.get("mom3m")         if not error else None,
                        "momentum_1m":            r.get("mom1m")         if not error else None,
                        "trailing_volatility_3m": r.get("vol_3m")        if not error else None,
                        "forward_pe":             r.get("forward_pe")    if not error else None,
                        "beta":                   r.get("beta")          if not error else None,
                        "dividend_yield":         r.get("dividend_yield") if not error else None,
                        "current_price":          price,
                        "currency":               currency,
                        "is_proxy":               False,
                        "proxy_for":              None,
                    }

            filtered_tickers = locked_tickers   # all 14 always included

            # ── Phase 2: Score via Gemini / VADER (date-isolated) ─────────────
            sentiment_scores, boom_triggers, macro_summary, scorer_source = score_etfs(
                filtered_tickers = filtered_tickers,
                all_etf_data     = all_etf_data,
                all_macro_news   = macro_news,
                reference_date   = buy_date,
                _force_vader     = not use_llm,
            )

            # Rate-limit pacing — sleep after Gemini calls to avoid 429
            if use_llm and scorer_source == "gemini" and gemini_ratelimit_delay > 0:
                time.sleep(gemini_ratelimit_delay)

            # ── Phase 3: Pseudo-Sharpe consensus — same formula as production ──
            _BT_VOL_FLOOR = 0.05
            _raw_cs: dict = {}
            expense_scores: dict = {}
            for ticker in filtered_tickers:
                etf           = all_etf_data[ticker]
                ter           = etf.get("expense_ratio")
                exp_score     = max(0.0, 1.0 - ter / ter_threshold) if ter is not None else 0.50
                sentiment     = sentiment_scores.get(ticker, 0.50)
                vol           = etf.get("trailing_volatility_3m")
                vol_dec       = max(_BT_VOL_FLOOR, float(vol)) if vol else _BT_VOL_FLOOR
                _raw_cs[ticker]        = (0.60 * sentiment + 0.40 * exp_score) / vol_dec
                expense_scores[ticker] = round(exp_score, 4)

            _max_raw = max(_raw_cs.values()) if _raw_cs else 1.0
            if _max_raw <= 0:
                _max_raw = 1.0
            consensus_scores = {t: round(_raw_cs[t] / _max_raw, 4) for t in filtered_tickers}

            # ── Phase 3b: Crash-Accumulator VA — mirrors Node 4 logic ─────────
            va_mult, va_reason = _detect_va_condition(
                tickers          = filtered_tickers,
                sentiment_scores = sentiment_scores,
                all_etf_data     = all_etf_data,
                boom_triggers    = boom_triggers,
            )
            effective_sip = round(sip_amount * va_mult, 2)
            va_triggered  = va_mult > 1.0

            if va_triggered:
                tier_label = (
                    "TIER 2 — Generational Crash" if va_mult >= VA_MULTIPLIER_TIER2
                    else "TIER 1 — Standard Dip"
                )
                _log(
                    f"  [backtest] ⚡ VALUE-AVERAGING {tier_label} — "
                    f"SIP ${sip_amount:.2f} → ${effective_sip:.2f} "
                    f"(×{va_mult:.2f})  |  {va_reason}"
                )

            # ── Phase 4: Fixed 70/30 bucket allocation (using effective SIP) ──
            allocs = _fixed_bucket_alloc(
                tickers          = filtered_tickers,
                consensus_scores = consensus_scores,
                sip_amount       = effective_sip,
                all_etf_data     = all_etf_data,
                sentiment_scores = sentiment_scores,
                expense_scores   = expense_scores,
                macro_summary    = macro_summary,
            )

            # ── Phase 4b: Hard rule validation (Rules 1/2/3/5 — no Rule 4) ───
            _MAX_POS_PCT = 0.15
            _MAX_REG_PCT = 0.50
            _rule_violations: list = []
            if not allocs:
                _rule_violations.append("RULE_5: No allocations generated")
            else:
                for _a in allocs:
                    if _a["monthly_usd"] > effective_sip * _MAX_POS_PCT:
                        _rule_violations.append(
                            f"RULE_1: {_a['ticker']} ${_a['monthly_usd']:.2f} "
                            f"> {_MAX_POS_PCT*100:.0f}% cap"
                        )
                    if _a["monthly_usd"] < 1.0:
                        _rule_violations.append(
                            f"RULE_3: {_a['ticker']} ${_a['monthly_usd']:.2f} < $1.00"
                        )
                _reg_totals: dict = {}
                for _a in allocs:
                    _reg_totals[_a["region"]] = (
                        _reg_totals.get(_a["region"], 0.0) + _a["monthly_usd"]
                    )
                for _reg, _tot in _reg_totals.items():
                    if _tot > effective_sip * _MAX_REG_PCT:
                        _rule_violations.append(
                            f"RULE_2: Region {_reg} ${_tot:.2f} "
                            f"> {_MAX_REG_PCT*100:.0f}% cap"
                        )
            if _rule_violations:
                for _rv in _rule_violations:
                    _log(f"  [backtest] ⚠ RULE VIOLATION (informational): {_rv}")

        except Exception as exc:
            _log("[backtest] {} research failed ({}) — skipping".format(month_str, exc))
            skipped_months.append(month_str)
            continue

        # ── Buy each ETF at the first available price on/after buy_date ───────
        positions: list = []
        month_usd = 0.0

        for alloc in allocs:
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
            effective_sip       = effective_sip,
            core_budget         = round(effective_sip * 0.70, 2),
            satellite_budget    = round(effective_sip * 0.30, 2),
            total_invested_usd  = round(month_usd, 2),
            positions           = positions,
            boom_triggers       = boom_triggers,
            macro_summary       = macro_summary,
            scorer              = scorer_source,
            va_triggered        = va_triggered,
            va_multiplier       = va_mult,
            usd_inr_rate        = usd_inr_rate,
        ))
        va_tag = f" [VA ×{va_mult:.2f}]" if va_triggered else ""
        _log("[backtest] {} done: {} positions, ${:.2f} invested (base SIP ${:.2f}{})\n".format(
            month_str, len(positions), month_usd, sip_amount, va_tag,
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
