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

# ── ETF Universe ──────────────────────────────────────────────────────────────

INTL_UNIVERSE: List[str] = [
    "VXUS", "SPDW", "IEFA", "SCHY", "VEA",
    "EEM",  "ACWI", "SCHF", "IXUS", "VWO", "IEMG",
]
HKCN_UNIVERSE: List[str] = [
    "CQQQ", "KWEB", "MCHI", "FXI",  "CHIQ",
    "GXC",  "FLCH", "KURE", "CNYA", "ASHR",
]
BSE_UNIVERSE: List[str] = [
    "NIFTYBEES.NS", "BANKBEES.NS",  "GOLDBEES.NS",   "JUNIORBEES.NS",
    "ITBEES.NS",    "INFRABEES.NS", "SETFNIF50.NS",  "CONSUMBEES.NS",
    "PHARMABEES.NS","DIVOPPBEES.NS",
]
ALL_TICKERS = INTL_UNIVERSE + HKCN_UNIVERSE + BSE_UNIVERSE

# ── TER fallback registry (from fund prospectuses, Q1 2026) ──────────────────

TER_REGISTRY: Dict[str, float] = {
    # INTL
    "VXUS": 0.0007, "SPDW": 0.0003, "IEFA": 0.0007, "SCHY": 0.0006,
    "VEA":  0.0005, "EEM":  0.0068, "ACWI": 0.0033, "SCHF": 0.0006,
    "IXUS": 0.0007, "VWO":  0.0008, "IEMG": 0.0009,
    # HKCN
    "CQQQ": 0.0065, "KWEB": 0.0069, "MCHI": 0.0059, "FXI":  0.0074,
    "CHIQ": 0.0065, "GXC":  0.0059, "FLCH": 0.0019, "KURE": 0.0069,
    "CNYA": 0.0055, "ASHR": 0.0065,
    # BSE (NSE ETFs — very low cost)
    "NIFTYBEES.NS":  0.0004, "BANKBEES.NS":  0.0010, "GOLDBEES.NS":  0.0010,
    "JUNIORBEES.NS": 0.0013, "ITBEES.NS":    0.0015, "INFRABEES.NS": 0.0010,
    "SETFNIF50.NS":  0.0007, "CONSUMBEES.NS":0.0010, "PHARMABEES.NS":0.0010,
    "DIVOPPBEES.NS": 0.0010,
}

# ── ETF metadata (category, trade_on) ────────────────────────────────────────

ETF_META: Dict[str, Dict[str, str]] = {
    # INTL
    "VXUS": {"category": "Total International", "trade_on": "US Broker"},
    "SPDW": {"category": "Developed ex-US",     "trade_on": "US Broker"},
    "IEFA": {"category": "Developed ex-US",     "trade_on": "US Broker"},
    "SCHY": {"category": "Intl High Dividend",  "trade_on": "US Broker"},
    "VEA":  {"category": "Developed Markets",   "trade_on": "US Broker"},
    "EEM":  {"category": "Emerging Markets",    "trade_on": "US Broker"},
    "ACWI": {"category": "All-World",           "trade_on": "US Broker"},
    "SCHF": {"category": "Developed ex-US",     "trade_on": "US Broker"},
    "IXUS": {"category": "Total International", "trade_on": "US Broker"},
    "VWO":  {"category": "Emerging Markets",    "trade_on": "US Broker"},
    "IEMG": {"category": "Emerging Markets",    "trade_on": "US Broker"},
    # HKCN
    "CQQQ": {"category": "China Tech",          "trade_on": "US Broker"},
    "KWEB": {"category": "China Internet",      "trade_on": "US Broker"},
    "MCHI": {"category": "China Broad",         "trade_on": "US Broker"},
    "FXI":  {"category": "China Large Cap",     "trade_on": "US Broker"},
    "CHIQ": {"category": "China Consumer",      "trade_on": "US Broker"},
    "GXC":  {"category": "China Broad",         "trade_on": "US Broker"},
    "FLCH": {"category": "China Broad",         "trade_on": "US Broker"},
    "KURE": {"category": "China Healthcare",    "trade_on": "US Broker"},
    "CNYA": {"category": "China A-Shares",      "trade_on": "US Broker"},
    "ASHR": {"category": "China A-Shares",      "trade_on": "US Broker"},
    # BSE
    "NIFTYBEES.NS":  {"category": "Nifty 50",        "trade_on": "Zerodha, Groww, Upstox"},
    "BANKBEES.NS":   {"category": "Banking",          "trade_on": "Zerodha, Groww, Upstox"},
    "GOLDBEES.NS":   {"category": "Gold",             "trade_on": "Zerodha, Groww, Upstox"},
    "JUNIORBEES.NS": {"category": "Nifty Next 50",    "trade_on": "Zerodha, Groww, Upstox"},
    "ITBEES.NS":     {"category": "IT Sector",        "trade_on": "Zerodha, Groww, Upstox"},
    "INFRABEES.NS":  {"category": "Infrastructure",   "trade_on": "Zerodha, Groww, Upstox"},
    "SETFNIF50.NS":  {"category": "Nifty 50",         "trade_on": "Zerodha, Groww, Upstox"},
    "CONSUMBEES.NS": {"category": "FMCG / Consumer",  "trade_on": "Zerodha, Groww, Upstox"},
    "PHARMABEES.NS": {"category": "Pharma",           "trade_on": "Zerodha, Groww, Upstox"},
    "DIVOPPBEES.NS": {"category": "Dividend Opp.",    "trade_on": "Zerodha, Groww, Upstox"},
}

# ── News queries per region ───────────────────────────────────────────────────

NEWS_QUERIES: Dict[str, List[str]] = {
    "INTL": [
        "international ETF outlook",
        "global equity markets momentum",
        "emerging markets ETF rally",
        "Fed interest rate global equities",
        "developed markets ETF performance",
    ],
    "HKCN": [
        "China ETF stock market outlook",
        "China GDP economic growth",
        "China tech stocks rally",
        "PBOC monetary policy stimulus",
        "Hong Kong stocks market",
    ],
    "BSE": [
        "India Nifty 50 ETF outlook",
        "RBI interest rate India",
        "India GDP growth Nifty",
        "India stock market rally",
        "NSE ETF performance India",
    ],
}


# ══════════════════════════════════════════════════════════════════════════════
#  Step 1 — Fetch ETF metrics
# ══════════════════════════════════════════════════════════════════════════════

def _region(ticker: str) -> str:
    if ticker in INTL_UNIVERSE:  return "INTL"
    if ticker in HKCN_UNIVERSE:  return "HKCN"
    return "BSE"


def _market_label(ticker: str) -> str:
    return "NSE" if ticker.endswith(".NS") else "NASDAQ"


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
            "ticker":       ticker,
            "name":         ticker,
            "region":       _region(ticker),
            "market":       _market_label(ticker),
            "expense_ratio": TER_REGISTRY.get(ticker),
            "aum_b":        None,
            "ytd_return":   None,
            "momentum_3m":  None,
            "momentum_1m":  None,
            "current_price":None,
        }
        try:
            t    = yf.Ticker(ticker)
            hist = t.history(start=jan1.isoformat(), auto_adjust=True)

            if not hist.empty:
                price_now  = float(hist["Close"].iloc[-1])
                rec["current_price"] = round(price_now, 4)

                # YTD return
                price_jan  = float(hist["Close"].iloc[0])
                if price_jan > 0:
                    rec["ytd_return"] = round((price_now - price_jan) / price_jan, 5)

                # 3-month momentum
                hist3m = hist[hist.index.date >= d3m_ago]
                if len(hist3m) >= 2:
                    p3m = float(hist3m["Close"].iloc[0])
                    rec["momentum_3m"] = round((price_now - p3m) / p3m, 5) if p3m > 0 else None

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

def compute_scores(
    metrics: Dict[str, Dict[str, Any]],
    region_sentiment: Dict[str, float],
    ter_threshold: float = 0.007,
) -> Tuple[Dict[str, float], Dict[str, float], Dict[str, float]]:
    """
    Returns (sentiment_scores, expense_scores, consensus_scores) per ticker.
    """
    sentiment_scores: Dict[str, float] = {}
    expense_scores:   Dict[str, float] = {}
    consensus_scores: Dict[str, float] = {}

    for ticker, rec in metrics.items():
        region   = rec["region"]
        sent     = region_sentiment.get(region, 0.50)

        # Momentum boost: add up to ±0.10 based on 3m momentum
        mom3m    = rec.get("momentum_3m") or 0.0
        mom_boost = max(-0.10, min(0.10, mom3m * 0.5))
        sent_adj  = round(min(0.95, max(0.05, sent + mom_boost)), 4)

        ter  = rec.get("expense_ratio")
        if ter is None:
            exp = 0.50                    # neutral if unknown
        else:
            exp = round(max(0.0, 1.0 - ter / ter_threshold), 4)

        cons = round(0.60 * sent_adj + 0.40 * exp, 4)

        sentiment_scores[ticker] = sent_adj
        expense_scores[ticker]   = exp
        consensus_scores[ticker] = cons

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
