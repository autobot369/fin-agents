"""
Node 1 — Regional Alpha & Cost Orchestrator
=============================================
Uses a locked 12-ETF universe split into two permanent buckets.
Universe discovery has been removed — the 12 tickers never change.

Node 1 responsibilities:
  1. Build ETFRecord for all 12 locked tickers via yfinance.
  2. Fetch recent macro news (DuckDuckGo + yfinance.news) for Node 2
     (signal_scorer). News drives satellite weight rotation — it is the
     only variable input per month.
  3. Always passes all 14 tickers as filtered_tickers (no TER/liquidity cull).

Locked buckets:
  Core (70% of SIP, 7 ETFs):
    VTI · SPLG · SPDW · SPEM · FLIN · NIFTYBEES.NS · QUAL
  Satellite (30% of SIP, 7 ETFs):
    XLK · QQQM · SOXQ · ICLN · USCA · ESGV · XLY

Backtest mode (as_of_date set):
  yfinance data and news are bounded to as_of_date for historical accuracy.
  Universe and buckets are identical to production — no discovery calls.
"""
from __future__ import annotations

import math
import time
from datetime import date as _Date, datetime as _Datetime, timedelta
from typing import Any, Dict, List, Optional

import yfinance as yf

from sip_execution_mas.graph.state import ETFRecord, SIPExecutionState


# ── Broker cost constants ─────────────────────────────────────────────────────

_BROKERAGE_MIN: Dict[str, float] = {
    "Dhan":   0.00,   # Zero brokerage for BSE/NSE ETFs
    "Alpaca": 0.00,   # Commission-free US ETFs
}
_TRADE_SIZE  = 500.0        # Reference trade size in USD


# ── Locked universe (12 ETFs, v3) ────────────────────────────────────────────
# Core (5): Broad-market anchors — low-cost beta + India strategic overweight
# Satellite (7): Thematic growth — semis, tech, small-cap value, uranium,
#                cybersecurity, India small-cap, biotech

_CORE_UNIVERSE: List[Dict[str, Any]] = [
    {"ticker": "USCA",         "name": "BNY Mellon US Large Cap Core Equity ETF",  "ter_pct": 0.07, "category": "broad_market",       "market": "NYSE",   "region": "US"},
    {"ticker": "SPDW",         "name": "SPDR Portfolio Developed World ex-US ETF", "ter_pct": 0.04, "category": "developed_markets",   "market": "NYSE",   "region": "US"},
    {"ticker": "SPEM",         "name": "SPDR Portfolio Emerging Markets ETF",      "ter_pct": 0.07, "category": "emerging_markets",    "market": "NYSE",   "region": "US"},
    {"ticker": "FLIN",         "name": "Franklin FTSE India ETF",                  "ter_pct": 0.19, "category": "india",               "market": "NYSE",   "region": "US"},
    {"ticker": "NIFTYBEES.NS", "name": "Nippon India ETF Nifty 50 BeES",           "ter_pct": 0.04, "category": "india",               "market": "NSE",    "region": "BSE"},
]

_SATELLITE_UNIVERSE: List[Dict[str, Any]] = [
    {"ticker": "SOXQ", "name": "Invesco PHLX Semiconductor ETF",              "ter_pct": 0.19, "category": "semiconductors",  "market": "NASDAQ", "region": "US"},
    {"ticker": "XLK",  "name": "Technology Select Sector SPDR Fund",          "ter_pct": 0.10, "category": "technology",      "market": "NYSE",   "region": "US"},
    {"ticker": "AVUV", "name": "Avantis U.S. Small Cap Value ETF",             "ter_pct": 0.25, "category": "small_cap_value", "market": "NYSE",   "region": "US"},
    {"ticker": "URNM", "name": "Sprott Uranium Miners ETF",                    "ter_pct": 0.83, "category": "uranium_energy",  "market": "NYSE",   "region": "US"},
    {"ticker": "CIBR", "name": "First Trust NASDAQ Cybersecurity ETF",         "ter_pct": 0.60, "category": "cybersecurity",   "market": "NASDAQ", "region": "US"},
    {"ticker": "SMIN", "name": "iShares MSCI India Small-Cap ETF",             "ter_pct": 0.74, "category": "india_small_cap", "market": "NYSE",   "region": "US"},
    {"ticker": "XBI",  "name": "SPDR S&P Biotech ETF",                         "ter_pct": 0.35, "category": "biotech",         "market": "NYSE",   "region": "US"},
]

_LOCKED_UNIVERSE            = _CORE_UNIVERSE + _SATELLITE_UNIVERSE
_CORE_UNIVERSE_TICKERS      = frozenset(r["ticker"] for r in _CORE_UNIVERSE)
_SATELLITE_UNIVERSE_TICKERS = frozenset(r["ticker"] for r in _SATELLITE_UNIVERSE)


# ── Thematic news categories ───────────────────────────────────────────────────
# Each category maps to 2 targeted DDGS search queries.  More specific queries
# give Gemini higher-quality signal than a single broad market search.

_CATEGORY_QUERIES: Dict[str, List[str]] = {
    "TECH_SEMIS": [
        "hyperscaler data center capex AI infrastructure semiconductor demand 2026",
        "TSMC chip supply chain NVIDIA AI accelerator NASDAQ cybersecurity spending",
    ],
    "INDIA_EM": [
        "FII foreign institutional investor flows India NSE RBI monetary policy 2026",
        "emerging market supply chain relocation India trade manufacturing small-cap",
    ],
    "URANIUM_ENERGY": [
        "uranium spot price nuclear energy AI data center power demand 2026",
        "Sprott uranium miners energy transition nuclear reactor capacity",
    ],
    "BIOTECH": [
        "FDA drug approval biotech GLP-1 weight loss RNA therapy 2026",
        "biotech M&A pipeline equal-weight XBI rate sensitivity innovation",
    ],
    "QUALITY_CORE": [
        "US large-cap equity broad market Federal Reserve outlook 2026",
        "developed world international equities Europe Japan dividend growth",
    ],
}

# Ticker → thematic category (drives targeted DDGS queries and VADER grouping)
_TICKER_CATEGORY: Dict[str, str] = {
    "SOXQ":         "TECH_SEMIS",
    "XLK":          "TECH_SEMIS",
    "CIBR":         "TECH_SEMIS",
    "FLIN":         "INDIA_EM",
    "NIFTYBEES.NS": "INDIA_EM",
    "SPEM":         "INDIA_EM",
    "SMIN":         "INDIA_EM",
    "URNM":         "URANIUM_ENERGY",
    "XBI":          "BIOTECH",
    "USCA":         "QUALITY_CORE",
    "SPDW":         "QUALITY_CORE",
    "AVUV":         "QUALITY_CORE",
}


# ── Cost attribution helpers ──────────────────────────────────────────────────

def _recommend_broker(ticker: str) -> str:
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return "Dhan"
    return "Alpaca"


def _calc_entry_cost(expense_ratio: float, broker: str, trade_size: float = _TRADE_SIZE) -> float:
    """Monthly drag: (brokerage_min / trade_size) + (expense_ratio / 12)"""
    return (_BROKERAGE_MIN.get(broker, 0.0) / trade_size) + (expense_ratio / 12)


def _adv_usd(
    ticker: str,
    avg_volume: Optional[int],
    price: Optional[float],
    usd_inr: float = 84.0,
) -> Optional[float]:
    """Average daily volume in USD equivalent."""
    if avg_volume is None or price is None or price <= 0:
        return None
    adv_native = avg_volume * price
    if ticker.endswith(".NS") or ticker.endswith(".BO"):
        return adv_native / usd_inr
    return adv_native


# ── yfinance enrichment ───────────────────────────────────────────────────────

def _fetch_yfinance_batch(
    tickers: List[str],
    as_of_date: Optional[_Date] = None,
) -> Dict[str, Dict[str, Any]]:
    """
    Fetch price metrics and static info for each ticker.
    as_of_date=None  → current data (production, uses period="1y").
    as_of_date set   → historical slice ending at as_of_date (backtest).
    """
    results: Dict[str, Dict[str, Any]] = {}
    for ticker in tickers:
        try:
            tk = yf.Ticker(ticker)
            if as_of_date:
                start = (as_of_date - timedelta(days=400)).isoformat()
                end   = (as_of_date + timedelta(days=1)).isoformat()
                hist  = tk.history(start=start, end=end, auto_adjust=True)
            else:
                hist = tk.history(period="1y", auto_adjust=True)

            info: Dict[str, Any] = {}
            try:
                info = tk.info or {}
            except Exception:
                pass

            if hist.empty:
                results[ticker] = {"error": "no_history"}
                continue

            close = hist["Close"]
            price = float(close.iloc[-1])

            mom3m = mom1m = ytd = vol_3m = None
            if len(close) >= 63:
                mom3m = round((price / float(close.iloc[-63]) - 1) * 100, 2)
                log_returns = close.pct_change().dropna().tail(63)
                if len(log_returns) >= 20:
                    vol_3m = round(float(log_returns.std()) * math.sqrt(252), 4)
            if len(close) >= 21:
                mom1m = round((price / float(close.iloc[-21]) - 1) * 100, 2)
            if len(close) >= 5:
                ytd = round((price / float(close.iloc[0]) - 1) * 100, 2)

            ter = None
            for key in ("annualReportExpenseRatio", "expenseRatio", "totalExpenseRatio"):
                v = info.get(key)
                if v and isinstance(v, (int, float)) and v > 0:
                    ter = float(v)
                    break

            aum   = info.get("totalAssets")
            aum_b = round(float(aum) / 1e9, 2) if aum else None

            avg_vol = info.get("averageVolume") or info.get("averageDailyVolume10Day")

            # Hard fundamental metrics for Gemini valuation anchoring
            _fpe = info.get("forwardPE")
            _beta = info.get("beta")
            _div = info.get("dividendYield") or info.get("trailingAnnualDividendYield")
            forward_pe    = round(float(_fpe),  2) if _fpe  and isinstance(_fpe,  (int, float)) else None
            beta          = round(float(_beta), 3) if _beta and isinstance(_beta, (int, float)) else None
            dividend_yield = round(float(_div),  4) if _div  and isinstance(_div,  (int, float)) else None

            results[ticker] = {
                "price":         round(price, 4),
                "ytd":           ytd,
                "mom3m":         mom3m,
                "mom1m":         mom1m,
                "vol_3m":        vol_3m,
                "ter":           ter,
                "aum_b":         aum_b,
                "avg_vol":       int(avg_vol) if avg_vol else None,
                "forward_pe":    forward_pe,
                "beta":          beta,
                "dividend_yield": dividend_yield,
                "source":        "yfinance",
            }
        except Exception as exc:
            results[ticker] = {"error": str(exc)}
        time.sleep(0.05)
    return results


# ── Thematic news fetch ───────────────────────────────────────────────────────

def _fetch_news(
    as_of_date: Optional[_Date] = None,
    max_per_category: int = 6,
) -> Dict[str, List[str]]:
    """
    Fetch thematic macro headlines per category group using targeted DDGS queries.
    Returns {"TECH_SEMIS": [...], "GREEN_ESG": [...], "INDIA_EM": [...], "QUALITY_CORE": [...]}.

    Category-specific queries give Gemini more precise signal than a single
    broad market search — e.g. hyperscaler capex news for XLK/SOXQ vs.
    carbon pricing news for ICLN/ESGV.

    as_of_date=None  → latest news (production).
    as_of_date set   → headlines filtered to articles published on or before
                       as_of_date (backtest — no future leakage).
    """
    news: Dict[str, List[str]] = {cat: [] for cat in _CATEGORY_QUERIES}

    cutoff_dt: Optional[_Datetime] = None
    if as_of_date:
        cutoff_dt = _Datetime.combine(as_of_date, _Datetime.max.time())

    try:
        try:
            from ddgs import DDGS
        except ImportError:
            from duckduckgo_search import DDGS

        with DDGS() as ddgs:
            for category, queries in _CATEGORY_QUERIES.items():
                headlines: List[str] = []
                for query in queries:
                    # Append month/year for backtest date isolation
                    if as_of_date:
                        query = f"{query} {as_of_date.strftime('%B %Y')}"
                    try:
                        fetch_n = max_per_category * 3 if as_of_date else max_per_category
                        results = list(ddgs.news(query, max_results=fetch_n))
                        for r in results:
                            if as_of_date and cutoff_dt:
                                date_str = r.get("date", "")
                                if date_str:
                                    try:
                                        pub_dt = _Datetime.fromisoformat(
                                            date_str.replace("Z", "+00:00")
                                        ).replace(tzinfo=None)
                                        if pub_dt > cutoff_dt:
                                            continue
                                    except Exception:
                                        pass
                            title = r.get("title", "")
                            if title and title not in headlines:
                                headlines.append(title)
                        time.sleep(0.3 if as_of_date else 0.1)
                    except Exception:
                        pass
                news[category] = headlines[:max_per_category]
    except Exception:
        pass   # all categories stay as empty lists

    return news


# ── Node function ─────────────────────────────────────────────────────────────

def regional_researcher_node(state: SIPExecutionState) -> dict:
    """
    Node 1 — Regional Alpha & Cost Orchestrator

    Always uses the locked 12-ETF universe — no discovery calls.
    Fetches yfinance metrics and macro news for all 12 tickers.
    News feeds Node 2 (signal_scorer) which rotates satellite weights.

    Production (as_of_date=None): live yfinance data, latest news.
    Backtest  (as_of_date set):   historical yfinance + news bounded to as_of_date.
    """
    as_of_date = state.get("as_of_date")   # None in production

    suffix = f" [as-of {as_of_date}]" if as_of_date else ""
    print(f"\n[Node 1] Building locked 12-ETF universe{suffix} …")

    all_tickers = [r["ticker"] for r in _LOCKED_UNIVERSE]

    # ── yfinance enrichment + thematic news ───────────────────────────────────
    raw        = _fetch_yfinance_batch(all_tickers, as_of_date=as_of_date)
    macro_news = _fetch_news(as_of_date=as_of_date)

    # Group by region for the ETFRecord loop
    market_recs: Dict[str, List[Dict[str, Any]]] = {}
    for entry in _LOCKED_UNIVERSE:
        market_recs.setdefault(entry["region"], []).append(entry)

    etf_data: Dict[str, ETFRecord] = {}
    filtered: List[str]            = []

    for market, recs in market_recs.items():
        for rec in recs:
            ticker  = rec["ticker"]
            r       = raw.get(ticker, {})
            error   = r.get("error")

            # TER: yfinance first, then locked universe estimate
            ter = r.get("ter") if not error else None
            if ter is None and rec.get("ter_pct"):
                ter = rec["ter_pct"] / 100.0

            price   = r.get("price")   if not error else None
            aum_b   = r.get("aum_b")   if not error else None
            avg_vol = r.get("avg_vol") if not error else None
            adv     = _adv_usd(ticker, avg_vol, price)

            broker     = _recommend_broker(ticker)
            entry_cost = (
                _calc_entry_cost(
                    expense_ratio=ter if ter is not None else 0.005,
                    broker=broker,
                ) if ter is not None else None
            )

            currency = "INR" if market == "BSE" else "USD"
            category = rec.get("category", "ETF")
            if currency == "INR":
                category = f"{category} [INR — via Dhan]"

            record: ETFRecord = {
                "ticker":                 ticker,
                "name":                   rec["name"],
                "region":                 market,
                "market":                 rec.get("market", "NSE" if market == "BSE" else "NASDAQ"),
                "category":               category,
                "expense_ratio":          ter,
                "aum_b":                  aum_b,
                "ytd_return":             r.get("ytd")    if not error else None,
                "momentum_3m":            r.get("mom3m")  if not error else None,
                "momentum_1m":            r.get("mom1m")  if not error else None,
                "trailing_volatility_3m": r.get("vol_3m") if not error else None,
                "forward_pe":             r.get("forward_pe")     if not error else None,
                "beta":                   r.get("beta")           if not error else None,
                "dividend_yield":         r.get("dividend_yield") if not error else None,
                "current_price":          price,
                "currency":               currency,
                "data_source":            "yfinance" if not error else "locked_estimate",
                "fetch_error":            error,
                "recommended_broker":     broker,
                "adv_usd":                adv,
                "liquidity_ok":           True,   # locked universe — always included
                "est_entry_cost_pct":     entry_cost,
                "is_proxy":               False,
                "proxy_for":              None,
            }
            etf_data[ticker] = record
            filtered.append(ticker)   # all 12 always pass

    print(f"[Node 1] {len(filtered)} ETFs ready "
          f"({len(_CORE_UNIVERSE_TICKERS)} core + {len(_SATELLITE_UNIVERSE_TICKERS)} satellite)")

    return {
        "all_etf_data":     etf_data,
        "all_macro_news":   macro_news,
        "filtered_tickers": filtered,
    }
