"""
Earnings Call Sentiment Analyzer
---------------------------------
1. Searches DuckDuckGo for the most recent earnings call transcript / summary.
2. Passes retrieved snippets to Claude for structured NLP sentiment analysis.

Returns a dict with management tone, key phrases, guidance direction, and a
1–10 sentiment score — ready to drop into the Researcher's ground_truth JSON.
"""

import json
import re
from typing import Any, Dict

from duckduckgo_search import DDGS
from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

_llm = ChatAnthropic(model="claude-opus-4-6", max_tokens=1200)

_SYSTEM = """You are a financial NLP specialist analyzing earnings call language.
Given search snippets about a company's most recent earnings call, extract structured
sentiment signals. Return ONLY a valid JSON object — no markdown, no extra text."""

_SCHEMA = """{
  "tone": "Confident|Cautious|Defensive|Evasive|Mixed",
  "key_phrases": ["exact quote 1", "exact quote 2", "exact quote 3"],
  "unusual_focus_topics": ["topic that got disproportionate management attention"],
  "guidance_direction": "Raised|Maintained|Lowered|Withdrawn|Not given",
  "sentiment_score": <integer 1-10, where 1=very bearish and 10=very bullish>,
  "one_line_summary": "single sentence synthesis"
}"""


def fetch_earnings_sentiment(ticker: str, company_name: str) -> Dict[str, Any]:
    """
    Returns a structured dict with management tone and sentiment from the most
    recent earnings call. Falls back to a neutral dict if content is unavailable.
    """
    snippets: list[str] = []
    seen: set[str] = set()

    queries = [
        f"{ticker} earnings call transcript management remarks guidance 2025",
        f"{company_name} CEO CFO quarterly results outlook commentary 2025",
        f"{ticker} earnings call key takeaways analyst Q&A 2025",
    ]

    with DDGS() as ddgs:
        for query in queries:
            try:
                results = ddgs.text(query, max_results=4, timelimit="y")
                for r in results:
                    title = r.get("title", "").strip()
                    body = r.get("body", "").strip()
                    if body and len(body) > 80 and title not in seen:
                        seen.add(title)
                        snippets.append(f"[{title}]\n{body[:500]}")
            except Exception as e:
                snippets.append(f"[Search error: {e}]")

    if not snippets:
        return {
            "tone": "Unknown",
            "key_phrases": [],
            "unusual_focus_topics": [],
            "guidance_direction": "Not given",
            "sentiment_score": None,
            "one_line_summary": "No earnings call content found via search.",
        }

    combined = "\n\n".join(snippets[:6])

    prompt = f"""Ticker: {ticker}
Company: {company_name}

EARNINGS CALL SEARCH SNIPPETS:
{combined}

Analyze the management tone. Return a JSON object matching this schema exactly:
{_SCHEMA}

If you cannot determine a value from the snippets, use null.
For key_phrases, only quote text that actually appears in the snippets."""

    try:
        response = _llm.invoke(
            [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
        )
        raw = (
            response.content
            if isinstance(response.content, str)
            else str(response.content)
        )
        # Extract JSON robustly
        match = re.search(r"\{.*\}", raw, re.DOTALL)
        if match:
            return json.loads(match.group())
        return {
            "tone": "Unknown",
            "key_phrases": [],
            "unusual_focus_topics": [],
            "guidance_direction": "Not given",
            "sentiment_score": None,
            "one_line_summary": "JSON parse failed — raw: " + raw[:200],
        }
    except Exception as e:
        return {
            "tone": "Unknown",
            "key_phrases": [],
            "unusual_focus_topics": [],
            "guidance_direction": "Not given",
            "sentiment_score": None,
            "one_line_summary": f"Sentiment analysis error: {e}",
        }
