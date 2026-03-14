"""
No-LLM ETF Ranker
==================
Generates rankings.json using only free, local tools — zero API keys.

Pipeline:
  1. Fetch ETF metrics via yfinance (price, momentum, YTD, TER, AUM)
  2. Fetch macro news headlines via DuckDuckGo Search (DDGS)
  3. Score headlines with VADER sentiment (local rule-based NLP)
  4. Compute consensus_score = 60% sentiment + 40% expense_efficiency
  5. Rank and save to outputs/rankings.json

Same output format as etf-selection-mas so the allocator and scheduler
consume it transparently.

Usage:
  python -m simulator.ranker              # rank and save
  python -m simulator.ranker --ter 0.50  # stricter TER ceiling
  python -m simulator.ranker --top 20    # rank top-20
"""

from __future__ import annotations

import json
import warnings
from datetime import date, datetime, timedelta
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

warnings.filterwarnings("ignore")

# ── SSL fix for macOS Python (python.org installer) ───────────────────────────
# Ensures yfinance / requests use the certifi CA bundle instead of the system
# keychain, which is not linked by default on macOS Python 3.x installs.
def _patch_ssl() -> None:
    import os
    try:
        import certifi
        os.environ.setdefault("SSL_CERT_FILE",      certifi.where())
        os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
    except ImportError:
        pass  # certifi not installed — SSL may still work on some systems

_patch_ssl()

# ── Locked ETF Universe (v2 — matches production MAS) ─────────────────────────
# Core bucket (70% of SIP) — broad market anchors + quality factor
CORE_UNIVERSE: List[str] = [
    "VTI", "SPLG", "SPDW", "SPEM", "FLIN", "NIFTYBEES.NS", "QUAL",
]
# Satellite bucket (30% of SIP) — sectors, tech, green energy
SATELLITE_UNIVERSE: List[str] = [
    "XLK", "QQQM", "SOXQ", "ICLN", "USCA", "ESGV", "XLY",
]

# Ranker region mapping (INTL = US-listed, BSE = NSE India)
INTL_UNIVERSE: List[str] = [t for t in CORE_UNIVERSE + SATELLITE_UNIVERSE if not t.endswith(".NS")]
HKCN_UNIVERSE: List[str] = []   # no HK/China exposure in locked universe
BSE_UNIVERSE:  List[str] = ["NIFTYBEES.NS"]
ALL_TICKERS = INTL_UNIVERSE + BSE_UNIVERSE

# ── TER fallback registry (fund prospectuses, Q1 2026) ───────────────────────

TER_REGISTRY: Dict[str, float] = {
    # Core — broad market
    "VTI":          0.0003,
    "SPLG":         0.0002,
    "SPDW":         0.0004,
    "SPEM":         0.0007,
    "FLIN":         0.0019,
    "QUAL":         0.0015,
    # Core — India NSE
    "NIFTYBEES.NS": 0.0004,
    # Satellite — technology & sectors
    "XLK":  0.0010,
    "QQQM": 0.0015,
    "SOXQ": 0.0019,
    "ICLN": 0.0042,
    "USCA": 0.0010,
    "ESGV": 0.0009,
    "XLY":  0.0010,
}

# ── ETF metadata (category, trade_on) ────────────────────────────────────────

ETF_META: Dict[str, Dict[str, str]] = {
    # Core — broad market
    "VTI":          {"category": "US Total Market",      "trade_on": "Alpaca",         "bucket": "core"},
    "SPLG":         {"category": "US Large-Cap (S&P 500)","trade_on": "Alpaca",         "bucket": "core"},
    "SPDW":         {"category": "Developed ex-US",      "trade_on": "Alpaca",         "bucket": "core"},
    "SPEM":         {"category": "Emerging Markets",     "trade_on": "Alpaca",         "bucket": "core"},
    "FLIN":         {"category": "India (US-listed)",    "trade_on": "Alpaca",         "bucket": "core"},
    "QUAL":         {"category": "US Quality Factor",    "trade_on": "Alpaca",         "bucket": "core"},
    # Core — India NSE
    "NIFTYBEES.NS": {"category": "Nifty 50",             "trade_on": "Dhan",           "bucket": "core"},
    # Satellite — technology & sectors
    "XLK":  {"category": "Technology Sector",    "trade_on": "Alpaca", "bucket": "satellite"},
    "QQQM": {"category": "NASDAQ-100",           "trade_on": "Alpaca", "bucket": "satellite"},
    "SOXQ": {"category": "Semiconductors",       "trade_on": "Alpaca", "bucket": "satellite"},
    "ICLN": {"category": "Clean Energy",         "trade_on": "Alpaca", "bucket": "satellite"},
    "USCA": {"category": "US ESG Leaders",       "trade_on": "Alpaca", "bucket": "satellite"},
    "ESGV": {"category": "US ESG Broad",         "trade_on": "Alpaca", "bucket": "satellite"},
    "XLY":  {"category": "Consumer Discretionary","trade_on": "Alpaca", "bucket": "satellite"},
}

# ── News queries per region ───────────────────────────────────────────────────

NEWS_QUERIES: Dict[str, List[str]] = {
    "INTL": [
        "US ETF market outlook semiconductor AI",
        "S&P 500 broad market ETF momentum",
        "Fed interest rate global equities 2026",
        "clean energy ESG ETF performance",
        "technology sector NASDAQ ETF rally",
    ],
    "BSE": [
        "India Nifty 50 ETF outlook",
        "RBI interest rate India 2026",
        "India GDP growth Nifty rally",
        "NSE ETF FII flows India market",
        "India stock market performance",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Fetch ETF metrics
# ══════════════════════════════════════════════════════════════════════════════

def _region(ticker: str) -> str:
    if ticker in BSE_UNIVERSE:   return "BSE"
    if ticker in HKCN_UNIVERSE:  return "HKCN"
    return "INTL"


def _market_label(ticker: str) -> str:
    if ticker.endswith(".NS"):
        return "NSE"
    meta = ETF_META.get(ticker, {})
    return "NYSE" if meta.get("trade_on") == "Alpaca" and ticker in ("SPLG", "SPDW", "SPEM", "FLIN", "XLK", "XLY") else "NASDAQ"


def fetch_etf_metrics(tickers: List[str]) -> Dict[str, Dict[str, Any]]:
    """
    Fetch price-based metrics for every ticker via yfinance.
    Returns dict: ticker → metric dict.
    """
    import yfinance as yf

    today    = date.today()
    jan1     = date(today.year, 1, 1)
    d3m_ago  = today - timedelta(days=92)
    d1m_ago  = today - timedelta(days=31)

    records: Dict[str, Dict[str, Any]] = {}

    print(f"  [ranker] Fetching metrics for {len(tickers)} ETFs …")
    for ticker in tickers:
        rec: Dict[str, Any] = {
            "ticker":                 ticker,
            "name":                   ticker,
            "region":                 _region(ticker),
            "market":                 _market_label(ticker),
            "expense_ratio":          TER_REGISTRY.get(ticker),
            "aum_b":                  None,
            "ytd_return":             None,
            "momentum_3m":            None,
            "momentum_1m":            None,
            "trailing_volatility_3m": None,
            "current_price":          None,
        }
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(start=jan1.isoformat(), auto_adjust=True)

            if not hist.empty:
                import math as _math
                price_now  = float(hist["Close"].iloc[-1])
                rec["current_price"] = round(price_now, 4)

                # YTD return
                price_jan  = float(hist["Close"].iloc[0])
                if price_jan > 0:
                    rec["ytd_return"] = round((price_now - price_jan) / price_jan, 5)

                # 3-month momentum + trailing volatility (annualised)
                hist3m = hist[hist.index.date >= d3m_ago]
                if len(hist3m) >= 2:
                    p3m = float(hist3m["Close"].iloc[0])
                    rec["momentum_3m"] = round((price_now - p3m) / p3m, 5) if p3m > 0 else None
                    log_ret = hist3m["Close"].pct_change().dropna()
                    if len(log_ret) >= 20:
                        rec["trailing_volatility_3m"] = round(
                            float(log_ret.std()) * _math.sqrt(252), 4
                        )

                # 1-month momentum
                hist1m = hist[hist.index.date >= d1m_ago]
                if len(hist1m) >= 2:
                    p1m = float(hist1m["Close"].iloc[0])
                    rec["momentum_1m"] = round((price_now - p1m) / p1m, 5) if p1m > 0 else None

            # Name & TER from .info (best-effort)
            try:
                info = t.info
                rec["name"] = info.get("longName") or info.get("shortName") or ticker
                if rec["expense_ratio"] is None:
                    ter = info.get("annualReportExpenseRatio") or info.get("expenseRatio")
                    if ter:
                        rec["expense_ratio"] = float(ter)
            except Exception:
                pass

            status = f"price={rec['current_price']}  ytd={rec['ytd_return']}  3m={rec['momentum_3m']}"
        except Exception as exc:
            status = f"ERROR: {exc}"

        print(f"    {ticker:<22}  {status}")
        records[ticker] = rec

    return records


# ══════════════════════════════════════════════════════════════════════════════
#  Step 2 — Fetch news headlines
# ══════════════════════════════════════════════════════════════════════════════

def fetch_news(max_per_query: int = 5) -> Dict[str, List[str]]:
    """
    Pull recent news headlines from DuckDuckGo for each region.
    Returns dict: region → list of headline strings.
    """
    try:
        from ddgs import DDGS
    except ImportError:
        try:
            from duckduckgo_search import DDGS
        except ImportError:
            print("  [ranker] ddgs not installed — skipping news (run: pip install ddgs)")
            return {r: [] for r in NEWS_QUERIES}

    headlines: Dict[str, List[str]] = {r: [] for r in NEWS_QUERIES}
    ddgs = DDGS()

    print(f"  [ranker] Fetching news headlines …")
    for region, queries in NEWS_QUERIES.items():
        for q in queries:
            try:
                results = ddgs.news(q, max_results=max_per_query)
                for r in results:
                    title = r.get("title", "")
                    body  = r.get("body", "")
                    if title:
                        headlines[region].append(f"{title}. {body}"[:300])
            except Exception as exc:
                print(f"    [ranker] news query '{q}' failed: {exc}")
        print(f"    {region:<6}  {len(headlines[region])} headlines")

    return headlines


# ══════════════════════════════════════════════════════════════════════════════
#  Step 3 — VADER sentiment scoring
# ══════════════════════════════════════════════════════════════════════════════

def score_sentiment_vader(headlines: Dict[str, List[str]]) -> Dict[str, float]:
    """
    Score each region's headlines with VADER and return a normalised [0,1] score.
    VADER compound score is in [-1, +1]; we map it to [0.1, 0.9].
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer

    sia = SentimentIntensityAnalyzer()
    region_scores: Dict[str, float] = {}

    for region, texts in headlines.items():
        if not texts:
            region_scores[region] = 0.50   # neutral fallback
            continue

        compounds = [sia.polarity_scores(t)["compound"] for t in texts]
        avg_compound = sum(compounds) / len(compounds)
        # Map [-1, +1] → [0.10, 0.90]
        normalised = round(0.10 + (avg_compound + 1) / 2 * 0.80, 4)
        region_scores[region] = normalised
        print(f"    {region:<6}  VADER avg={avg_compound:+.3f}  →  score={normalised:.4f}")

    return region_scores


# ══════════════════════════════════════════════════════════════════════════════
#  Step 4 — Compute expense & consensus scores
# ══════════════════════════════════════════════════════════════════════════════

_VOL_FLOOR = 0.05   # 5% annualised — matches portfolio_optimizer._VOL_FLOOR

def compute_scores(
    metrics: Dict[str, Dict[str, Any]],
    region_sentiment: Dict[str, float],
    ter_threshold: float = 0.007,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Returns (sentiment_scores, expense_scores, consensus_scores) per ticker.

    Consensus formula (pseudo-Sharpe rank, matches portfolio_optimizer):
        numerator_i  = 0.60 × sentiment_i + 0.40 × expense_i
        raw_i        = numerator_i / max(vol_i, VOL_FLOOR)
        consensus_i  = raw_i / max_j(raw_j)   → normalised to [0, 1]

    Sentiment = VADER regional score propagated to each ticker in the region,
    plus a momentum boost of ±0.10 based on 3-month return.
    """
    sentiment_scores: Dict[str, float] = {}
    expense_scores:   Dict[str, float] = {}
    raw_consensus:    Dict[str, float] = {}

    for ticker, rec in metrics.items():
        region   = rec["region"]
        sent     = region_sentiment.get(region, 0.50)

        # Momentum boost: up to ±0.10 based on 3m momentum
        mom3m     = rec.get("momentum_3m") or 0.0
        mom_boost = max(-0.10, min(0.10, mom3m * 0.5))
        sent_adj  = round(min(0.95, max(0.05, sent + mom_boost)), 4)

        ter = rec.get("expense_ratio")
        exp = round(max(0.0, 1.0 - ter / ter_threshold), 4) if ter is not None else 0.50

        vol     = rec.get("trailing_volatility_3m")
        vol_dec = max(_VOL_FLOOR, float(vol)) if vol else _VOL_FLOOR

        sentiment_scores[ticker] = sent_adj
        expense_scores[ticker]   = exp
        raw_consensus[ticker]    = (0.60 * sent_adj + 0.40 * exp) / vol_dec

    # Batch normalise to [0, 1]
    max_raw = max(raw_consensus.values()) if raw_consensus else 1.0
    if max_raw <= 0:
        max_raw = 1.0
    consensus_scores = {t: round(v / max_raw, 4) for t, v in raw_consensus.items()}

    return sentiment_scores, expense_scores, consensus_scores


# ══════════════════════════════════════════════════════════════════════════════
#  Step 5 — Build rationale (template, no LLM)
# ══════════════════════════════════════════════════════════════════════════════

def _rationale(ticker: str, rec: Dict, sent: float, cons: float) -> str:
    region = rec["region"]
    mom3m  = rec.get("momentum_3m") or 0.0
    ter    = rec.get("expense_ratio")

    if cons >= 0.75:
        strength = "Strong"
    elif cons >= 0.60:
        strength = "Moderate"
    else:
        strength = "Cautious"

    region_label = {
        "INTL": "global developed-market",
        "HKCN": "China/HK tech and broad-market",
        "BSE":  "India equity",
    }.get(region, "")

    mom_str = (
        f" with {abs(mom3m)*100:.1f}% {'tailwind' if mom3m > 0 else 'headwind'} over 3 months"
        if abs(mom3m) > 0.01 else ""
    )
    cost_str = (
        f"; ultra-low TER of {ter*100:.3f}%"
        if ter and ter < 0.002 else
        (f"; TER {ter*100:.3f}%" if ter else "")
    )

    return (
        f"{strength} {region_label} sentiment score ({sent:.2f}){mom_str}{cost_str}."
    )


# ══════════════════════════════════════════════════════════════════════════════
#  Main ranker
# ══════════════════════════════════════════════════════════════════════════════

def run_ranker(
    ter_threshold: float = 0.007,
    top_n: int = 20,
    output_path: Optional[str] = None,
    news_per_query: int = 5,
) -> Dict[str, Any]:
    """
    Full no-LLM ranking pipeline. Returns the rankings dict and saves to JSON.
    """
    if output_path is None:
        out_dir     = Path(__file__).resolve().parent.parent / "etf-selection-mas" / "outputs"
        out_dir.mkdir(exist_ok=True)
        output_path = str(out_dir / "rankings.json")

    print(f"\n{'='*60}")
    print(f"  NO-LLM ETF RANKER  —  {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
    print(f"  Universe : {len(ALL_TICKERS)} ETFs  |  TER ceiling: {ter_threshold*100:.2f}%")
    print(f"{'='*60}")

    # 1. Metrics
    print(f"\n[1/4] Fetching ETF metrics via yfinance …")
    metrics = fetch_etf_metrics(ALL_TICKERS)

    # 2. TER filter
    print(f"\n[2/4] Applying TER filter (<= {ter_threshold*100:.2f}%) …")
    filtered = [
        t for t, r in metrics.items()
        if r.get("expense_ratio") is None or r["expense_ratio"] <= ter_threshold
    ]
    pruned   = [t for t in ALL_TICKERS if t not in filtered]
    print(f"  Passed: {len(filtered)}  |  Pruned: {len(pruned)}  {pruned if pruned else ''}")

    # 3. News + VADER
    print(f"\n[3/4] Fetching news and scoring with VADER …")
    headlines        = fetch_news(max_per_query=news_per_query)
    region_sentiment = score_sentiment_vader(headlines)

    # 4. Scores
    print(f"\n[4/4] Computing consensus scores …")
    sent_scores, exp_scores, cons_scores = compute_scores(
        {t: metrics[t] for t in filtered},
        region_sentiment,
        ter_threshold,
    )

    # Sort and build output entries
    ranked_tickers = sorted(filtered, key=lambda t: (
        round(cons_scores.get(t, 0), 4),
        round(metrics[t].get("momentum_3m") or 0, 4),
    ), reverse=True)[:top_n]

    entries = []
    for rank, ticker in enumerate(ranked_tickers, 1):
        rec  = metrics[ticker]
        meta = ETF_META.get(ticker, {"category": "—", "trade_on": "—"})
        sent = sent_scores.get(ticker, 0.5)
        cons = cons_scores.get(ticker, 0.5)
        entries.append({
            "rank":               rank,
            "ticker":             ticker,
            "name":               rec["name"],
            "market_label":       rec["market"],
            "region":             rec["region"],
            "category":           meta["category"],
            "trade_on":           meta["trade_on"],
            "expense_ratio":      rec.get("expense_ratio"),
            "aum_b":              rec.get("aum_b"),
            "ytd_return":         rec.get("ytd_return"),
            "momentum_3m":        rec.get("momentum_3m"),
            "momentum_1m":        rec.get("momentum_1m"),
            "current_price":      rec.get("current_price"),
            "sentiment_score":    sent,
            "expense_score":      exp_scores.get(ticker, 0.5),
            "consensus_score":    cons,
            "sentiment_rationale":_rationale(ticker, rec, sent, cons),
        })

    payload = {
        "generated_at":       date.today().isoformat(),
        "generator":          "no-llm-vader",
        "ter_threshold":      ter_threshold,
        "boom_triggers_fired":[],
        "region_sentiment":   region_sentiment,
        "rankings":           entries,
    }

    with open(output_path, "w", encoding="utf-8") as f:
        json.dump(payload, f, indent=2, default=str)

    # Terminal summary
    print(f"\n  {'Rank':<5} {'Ticker':<22} {'Region':<6} {'Sentiment':>9} {'Expense':>8} {'Consensus':>10}")
    print(f"  {'─'*65}")
    for e in entries:
        print(
            f"  #{e['rank']:<4} {e['ticker']:<22} {e['region']:<6} "
            f"{e['sentiment_score']:>9.4f} {e['expense_score']:>8.4f} {e['consensus_score']:>10.4f}"
        )
    print(f"\n  Rankings saved → {output_path}\n")

    return payload


# ── CLI ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import argparse
    p = argparse.ArgumentParser(description="No-LLM ETF Ranker (VADER + yfinance + DDGS)")
    p.add_argument("--ter",    type=float, default=0.70,
                   help="TER ceiling in %% (default: 0.70)")
    p.add_argument("--top",    type=int,   default=20,
                   help="Number of ETFs to include in ranking (default: 20)")
    p.add_argument("--output", type=str,   default=None,
                   help="Output path for rankings.json")
    args = p.parse_args()
    run_ranker(ter_threshold=args.ter / 100, top_n=args.top, output_path=args.output)
