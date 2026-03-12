"""
Free News Fetcher — DDGS + yfinance.Ticker.news
================================================
100% free. No API key required. Two sources are merged per query:

  Source A — DuckDuckGo News (DDGS)
  ----------------------------------
  Used for macro-level queries:
    • "Market news for {TICKER} ETF"
    • "Monthly outlook for {MARKET} equities March 2026"
    • Thematic macro queries (Fed soft landing, China GDP, RBI pivot, etc.)

  Source B — yfinance.Ticker.news
  --------------------------------
  Used for ticker-specific news embedded in the Yahoo Finance feed.
  yfinance ≥ 0.2.40 uses a nested `content` object; older builds use flat fields.
  Both formats are normalised to the same schema.

Output schema (every article):
  {
    "title"     : str,
    "body"      : str,          # snippet / summary, max 500 chars
    "date"      : str,          # ISO-8601 or empty string
    "source"    : str,          # publisher / provider name
    "url"       : str,
    "relevance" : float,        # 0.0–1.0; DDGS default 0.6, yfinance 0.7
    "fetch_src" : "ddgs"|"yfinance"
  }

Rate-limiting notes:
  DDGS is a scraping wrapper — successive rapid calls can trigger temporary
  blocks. A 0.4 s sleep is inserted between DDGS query bursts.
  yfinance calls are local (cached or direct) and carry no rate risk here.
"""

import time
from datetime import datetime, timezone
from typing import Any, Dict, List

import yfinance as yf
from duckduckgo_search import DDGS

# ---------------------------------------------------------------------------
# Macro query templates — one set per market region.
# Each query targets a specific "boom hypothesis" that the SentimentScorer
# will evaluate against the BOOM_TRIGGERS catalogue.
# ---------------------------------------------------------------------------
MACRO_QUERIES: Dict[str, List[str]] = {
    "INTL": [
        "international ETF monthly outlook March 2026",
        "Fed soft landing global equity impact 2026",
        "MSCI World developed markets momentum March 2026",
        "US dollar weakness emerging markets rally 2026",
    ],
    "HKCN": [
        "China GDP 2026 economic growth Q1 forecast",
        "China tech re-rating AI localization DeepSeek 2026",
        "PBOC stimulus monetary easing March 2026",
        "Hong Kong China internet stocks rally March 2026",
    ],
    "BSE": [
        "RBI pivot interest rate cut India 2026",
        "India inflation CPI target RBI monetary policy March 2026",
        "India GDP growth Nifty 50 equity outlook Q1 2026",
        "Fed vs RBI interest rate differential India equities 2026",
    ],
}

# Thematic cross-market queries — run once regardless of market segment.
CROSS_MARKET_QUERIES: List[str] = [
    "Fed vs RBI interest rates global ETF impact 2026",
    "emerging market ETF inflows March 2026 boom",
    "global equity rotation developed vs emerging markets 2026",
]

# Seconds to pause between DDGS queries to avoid scraper throttling.
_DDGS_SLEEP_S: float = 0.4


# ---------------------------------------------------------------------------
# Normalisation helpers
# ---------------------------------------------------------------------------

def _unix_to_iso(ts: Any) -> str:
    """Convert a Unix timestamp (int/float) to ISO-8601 string."""
    try:
        return datetime.fromtimestamp(int(ts), tz=timezone.utc).strftime("%Y-%m-%d")
    except (TypeError, ValueError, OSError):
        return ""


def _normalise_ddgs(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalise a single DuckDuckGo search result."""
    return {
        "title":    raw.get("title", "").strip(),
        "body":     raw.get("body", "")[:500].strip(),
        "date":     raw.get("date", ""),
        "source":   raw.get("source", ""),
        "url":      raw.get("url", ""),
        "relevance": 0.60,
        "fetch_src": "ddgs",
    }


def _normalise_yf_article(raw: Dict[str, Any]) -> Dict[str, Any]:
    """
    Normalise a yfinance Ticker.news item.

    yfinance ≥ 0.2.40 wraps article data under a nested `content` key:
        raw["content"]["title"], raw["content"]["summary"], etc.
    Older builds expose flat keys:
        raw["title"], raw["publisher"], raw["providerPublishTime"], etc.
    We handle both gracefully.
    """
    content = raw.get("content", {})          # new-style nested object

    if content:
        # ── New yfinance format (≥ 0.2.40) ───────────────────────────────
        title   = content.get("title", "").strip()
        body    = content.get("summary", "")[:500].strip()
        pub_raw = content.get("pubDate", "")
        # pubDate is already an ISO string in the new format
        date    = pub_raw[:10] if pub_raw else ""
        source  = (content.get("provider") or {}).get("displayName", "Yahoo Finance")
        url     = (content.get("canonicalUrl") or {}).get("url", "")
    else:
        # ── Legacy yfinance format (< 0.2.40) ────────────────────────────
        title   = raw.get("title", "").strip()
        body    = raw.get("summary", "")[:500].strip()
        ts      = raw.get("providerPublishTime")
        date    = _unix_to_iso(ts) if ts else ""
        source  = raw.get("publisher", "Yahoo Finance")
        url     = raw.get("link", "")

    return {
        "title":     title,
        "body":      body,
        "date":      date,
        "source":    source,
        "url":       url,
        "relevance": 0.70,   # Ticker-matched news scores slightly higher
        "fetch_src": "yfinance",
    }


# ---------------------------------------------------------------------------
# Core fetch functions
# ---------------------------------------------------------------------------

def fetch_ddgs_news(
    query: str,
    max_results: int = 8,
    timelimit: str = "m",          # "d"=day, "w"=week, "m"=month
) -> List[Dict[str, Any]]:
    """
    Run a single DuckDuckGo news search and return normalised results.

    Args:
        query:       Search string.
        max_results: Maximum number of articles (default 8).
        timelimit:   DuckDuckGo time filter — "m" = last month (matches March 2026 cycle).

    Returns:
        List of normalised article dicts; empty list on error.
    """
    try:
        with DDGS() as ddgs:
            raw_results = ddgs.news(
                query,
                max_results=max_results,
                timelimit=timelimit,
            )
        return [_normalise_ddgs(r) for r in raw_results if r.get("title")]
    except Exception as exc:
        print(f"  [news_search/ddgs] ERROR '{query[:50]}': {exc}")
        return [{
            "title":    f"[DDGS fetch error] {query[:50]}",
            "body":     str(exc),
            "date":     "",
            "source":   "error",
            "url":      "",
            "relevance": 0.0,
            "fetch_src": "ddgs",
        }]


def fetch_yfinance_news(ticker: str, max_results: int = 6) -> List[Dict[str, Any]]:
    """
    Fetch news from yfinance.Ticker.news for a specific ticker.

    Args:
        ticker:      ETF ticker symbol (e.g. "KWEB", "NIFTYBEES.NS").
        max_results: Maximum articles to return (default 6).

    Returns:
        List of normalised article dicts; empty list on fetch failure.
    """
    try:
        etf = yf.Ticker(ticker)
        raw_news = etf.news or []
        articles = [_normalise_yf_article(a) for a in raw_news[:max_results]]
        # Drop empty titles (malformed entries)
        return [a for a in articles if a["title"] and "fetch error" not in a["title"].lower()]
    except Exception as exc:
        print(f"  [news_search/yfinance] ERROR '{ticker}': {exc}")
        return []


# ---------------------------------------------------------------------------
# High-level: fetch all news for a market segment
# ---------------------------------------------------------------------------

def fetch_market_news(
    market: str,
    tickers: List[str],
    include_cross_market: bool = True,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Fetch all news for a market segment by combining:
      1. DDGS macro queries (from MACRO_QUERIES[market])
      2. DDGS ticker-level queries  → "Market news for {TICKER} ETF"
      3. yfinance Ticker.news       → per-ticker Yahoo Finance feed
      4. Cross-market thematic queries (optional)

    Returns:
        Dict mapping query/ticker key → list of normalised article dicts.
    """
    results: Dict[str, List[Dict[str, Any]]] = {}
    seen_titles: set = set()

    def _dedupe(articles: List[Dict]) -> List[Dict]:
        out = []
        for a in articles:
            t = a["title"].lower().strip()
            if t and t not in seen_titles:
                seen_titles.add(t)
                out.append(a)
        return out

    # ── 1. DDGS macro queries for this market ─────────────────────────────
    for query in MACRO_QUERIES.get(market, []):
        print(f"  [news/ddgs] {query[:65]}")
        arts = fetch_ddgs_news(query, max_results=8)
        results[query] = _dedupe(arts)
        time.sleep(_DDGS_SLEEP_S)

    # ── 2. Cross-market thematic queries (run once per pipeline) ──────────
    if include_cross_market:
        for query in CROSS_MARKET_QUERIES:
            if query not in results:
                print(f"  [news/ddgs:global] {query[:65]}")
                arts = fetch_ddgs_news(query, max_results=6)
                results[query] = _dedupe(arts)
                time.sleep(_DDGS_SLEEP_S)

    # ── 3. Ticker-level DDGS + yfinance news ──────────────────────────────
    # Cap at 12 tickers to keep runtime reasonable; prioritise by TER later.
    for ticker in tickers[:12]:
        ticker_key = f"Market news for {ticker} ETF"
        print(f"  [news/ticker] {ticker}")

        # DDGS ticker query
        ddgs_arts = fetch_ddgs_news(ticker_key, max_results=5)
        time.sleep(_DDGS_SLEEP_S)

        # yfinance native news
        yf_arts = fetch_yfinance_news(ticker, max_results=5)

        combined = _dedupe(ddgs_arts + yf_arts)
        if combined:
            results[ticker_key] = combined

    total = sum(len(v) for v in results.values())
    print(f"  [news_search] {market}: {total} articles across {len(results)} queries")
    return results


# ---------------------------------------------------------------------------
# Convenience: fetch news for a single ticker (used by researcher in brief mode)
# ---------------------------------------------------------------------------

def fetch_ticker_news_combined(ticker: str) -> List[Dict[str, Any]]:
    """
    Fetch merged DDGS + yfinance news for one ticker.
    Useful for spot-checks or per-ticker enrichment in the Arbiter.
    """
    ddgs_arts = fetch_ddgs_news(f"Market news for {ticker} ETF", max_results=5)
    yf_arts   = fetch_yfinance_news(ticker, max_results=5)
    seen: set = set()
    merged = []
    for a in ddgs_arts + yf_arts:
        t = a["title"].lower().strip()
        if t and t not in seen:
            seen.add(t)
            merged.append(a)
    return merged
