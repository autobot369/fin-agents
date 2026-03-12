"""
Fetches recent market news via DuckDuckGo search.
"""

from typing import Dict, List

from duckduckgo_search import DDGS


def fetch_recent_news(
    ticker: str,
    company_name: str,
    max_results: int = 12,
) -> List[Dict[str, str]]:
    """
    Returns a list of recent news items, each with title, body, date, source.
    Runs two targeted queries:
      1. Stock-specific news (earnings, guidance, analyst actions)
      2. Macro / sector headwinds relevant to the company
    """
    news_items: List[Dict[str, str]] = []
    seen_titles: set = set()

    queries = [
        f"{ticker} {company_name} stock earnings guidance analyst forecast",
        f"{company_name} industry outlook macro headwinds competition",
    ]

    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.news(query, max_results=max_results // 2, timelimit="m")
                for r in results:
                    title = r.get("title", "").strip()
                    if title and title not in seen_titles:
                        seen_titles.add(title)
                        news_items.append(
                            {
                                "title": title,
                                "body": r.get("body", "")[:400],
                                "date": r.get("date", ""),
                                "source": r.get("source", ""),
                            }
                        )
            except Exception as e:
                news_items.append(
                    {
                        "title": f"[News fetch error for query '{query}']",
                        "body": str(e),
                        "date": "",
                        "source": "",
                    }
                )

    return news_items
