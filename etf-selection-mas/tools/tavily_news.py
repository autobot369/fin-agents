"""
Tavily News Tool
----------------
Fetches targeted macro news via the Tavily Search API.

Query strategy per market:
  INTL  → Fed policy, developed-market macro, USD trends
  HKCN  → China GDP 2026, PBOC, AI localization, tech regulation
  BSE   → India CPI/inflation, RBI rates, India GDP, Nifty outlook

Each result is normalized to:
  {title, body, date, source, url, relevance_score}

The Tavily `search_depth="advanced"` mode is used to get full article bodies
rather than snippet-only results, which is essential for the SentimentScorer's
Claude-based analysis.
"""

import os
from typing import Any, Dict, List

from tavily import TavilyClient

# ---------------------------------------------------------------------------
# Macro query templates — each maps to a "boom/bust" hypothesis the
# SentimentScorer will evaluate.
# ---------------------------------------------------------------------------
MACRO_QUERIES: Dict[str, List[str]] = {
    "INTL": [
        "Fed interest rate decision March 2026 global equity impact",
        "MSCI World developed markets outlook Q1 2026",
        "US dollar strength emerging markets 2026",
        "international ETF inflows March 2026",
    ],
    "HKCN": [
        "China GDP 2026 economic growth Q1 forecast",
        "China AI localization DeepSeek technology sector 2026",
        "PBOC monetary policy stimulus March 2026",
        "Hong Kong China tech stocks rally 2026",
        "China consumer spending retail sales March 2026",
    ],
    "BSE": [
        "India inflation CPI 2026 RBI interest rate decision",
        "India GDP growth forecast Q1 2026",
        "BSE Nifty 50 outlook Indian equity market March 2026",
        "RBI vs Fed interest rate differential India 2026",
        "India manufacturing PMI FDI inflows 2026",
    ],
}


def _normalize_result(raw: Dict[str, Any]) -> Dict[str, Any]:
    """Normalize a Tavily search result to our internal schema."""
    return {
        "title": raw.get("title", "").strip(),
        "body": (raw.get("content") or raw.get("raw_content") or "")[:600].strip(),
        "date": raw.get("published_date", ""),
        "source": raw.get("source", ""),
        "url": raw.get("url", ""),
        "relevance_score": round(float(raw.get("score", 0.0)), 3),
    }


def fetch_macro_news(
    market: str,
    extra_queries: List[str] | None = None,
    max_results_per_query: int = 5,
) -> Dict[str, List[Dict[str, Any]]]:
    """
    Run Tavily searches for all macro queries relevant to `market`.

    Returns:
        {query_string: [normalized_result, ...]}

    Raises:
        ValueError if TAVILY_API_KEY is not set.
    """
    api_key = os.getenv("TAVILY_API_KEY")
    if not api_key:
        raise ValueError(
            "TAVILY_API_KEY environment variable is not set. "
            "Get your key at https://tavily.com"
        )

    client = TavilyClient(api_key=api_key)
    queries = list(MACRO_QUERIES.get(market, []))
    if extra_queries:
        queries.extend(extra_queries)

    results: Dict[str, List[Dict[str, Any]]] = {}

    for query in queries:
        print(f"  [tavily] Searching: \"{query}\"")
        try:
            response = client.search(
                query=query,
                search_depth="advanced",      # full article body
                max_results=max_results_per_query,
                include_raw_content=False,
                include_answer=True,          # Tavily's AI answer — used as a summary
            )
            raw_results = response.get("results", [])
            ai_answer = response.get("answer", "")

            normalized = [_normalize_result(r) for r in raw_results]

            # Prepend Tavily's synthesized answer as a synthetic "article"
            # so the SentimentScorer can use it as a high-quality summary signal.
            if ai_answer:
                normalized.insert(0, {
                    "title": f"[Tavily Summary] {query}",
                    "body": ai_answer[:600],
                    "date": "",
                    "source": "tavily_ai",
                    "url": "",
                    "relevance_score": 1.0,   # synthetic — always high priority
                })

            results[query] = normalized

        except Exception as exc:
            print(f"  [tavily] ERROR for query '{query}': {exc}")
            results[query] = [{
                "title": f"[fetch error] {query}",
                "body": str(exc),
                "date": "",
                "source": "error",
                "url": "",
                "relevance_score": 0.0,
            }]

    return results
