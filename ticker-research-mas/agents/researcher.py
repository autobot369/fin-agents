"""
Researcher Agent Node
---------------------
Fetches yfinance market data + DuckDuckGo news + SEC filings + earnings call
sentiment, then asks Claude to synthesize a structured research brief.

Also emits a `ground_truth` dict — the verified quantitative anchor that the
Fact-Checker and Arbiter treat as authoritative and immutable.
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import MarketState
from tools.earnings_sentiment import fetch_earnings_sentiment
from tools.market_data import get_market_data # <-- UPDATED IMPORT
from tools.news_search import fetch_recent_news
from tools.sec_filings import fetch_risk_factors

_llm = ChatAnthropic(
    model="claude-opus-4-6",
    max_tokens=3500,
    thinking={"type": "adaptive"},
)

_SYSTEM = """You are a senior equity research analyst producing a pre-debate research brief.
Your job is to present raw market data and news OBJECTIVELY — do NOT take a bullish or
bearish stance. Highlight key facts, figures, and catalysts on BOTH sides so that
downstream Bull and Bear agents can build their theses from a common factual foundation.
Be precise: always quote the exact numbers from the data provided."""


def _build_ground_truth(market_data: dict, sec: dict, sentiment: dict) -> dict:
    """
    Distill verified, machine-readable numbers into a single authoritative JSON.
    This is the quantitative anchor — downstream agents must not contradict these values.
    """
    md = market_data
    return {
        # ── Identity ──────────────────────────────────────────────────────
        "company_name": md.get("company_name"),
        "sector": md.get("sector"),
        # ── Anchor Signal A: Futures Basis ────────────────────────────────
        "index_name": md.get("index_name"),
        "futures_symbol": md.get("futures_symbol"),
        "futures_price": md.get("futures_price"),
        "spot_price": md.get("spot_price"),
        "futures_basis": md.get("futures_basis"),
        "basis_pct": md.get("basis_pct"),
        "basis_state": md.get("basis_state"),        # "Contango" | "Backwardation"
        # ── Anchor Signal B: Volume-Induced Volatility ────────────────────
        "futures_volume": md.get("futures_volume"),
        "futures_volume_avg30": md.get("futures_volume_avg30"),
        "futures_volume_ratio": md.get("futures_volume_ratio"),  # today / 30d avg
        "volatility_level": md.get("volatility_level"),          # NORMAL | ELEVATED | SPIKE
        # ── Price & Range ─────────────────────────────────────────────────
        "current_price": md.get("current_price"),
        "high_52w": md.get("high_52w"),
        "low_52w": md.get("low_52w"),
        "price_change_6mo_pct": md.get("price_change_6mo_pct"),
        # ── Trend ─────────────────────────────────────────────────────────
        "ma200": md.get("ma200"),
        # ── Momentum ─────────────────────────────────────────────────────
        "rsi_14": md.get("rsi_14"),
        "macd_bullish": md.get("macd_bullish"),
        "bb_position_pct": md.get("bb_position_pct"),
        # ── Valuation ────────────────────────────────────────────────────
        "pe_trailing": md.get("pe_trailing"),
        "pe_forward": md.get("pe_forward"),
        "peg_ratio": md.get("peg_ratio"),
        "price_to_book": md.get("price_to_book"),
        # ── Fundamentals ─────────────────────────────────────────────────
        "revenue_growth_yoy": md.get("revenue_growth_yoy"),
        "earnings_growth_yoy": md.get("earnings_growth_yoy"),
        "profit_margin": md.get("profit_margin"),
        "market_cap_b": md.get("market_cap_b"),
        # ── Analyst & Short Interest ──────────────────────────────────────
        "analyst_target_price": md.get("analyst_target_price"),
        "analyst_recommendation": md.get("analyst_recommendation"),
        "short_percent_float": md.get("short_percent_float"),
        # ── Alternative Data ──────────────────────────────────────────────
        "sec_filing_type": sec.get("filing_type"),
        "sec_risk_factors_excerpt": sec.get("risk_factors", "")[:800],
        "earnings_call_tone": sentiment.get("tone"),
        "earnings_call_guidance": sentiment.get("guidance_direction"),
        "earnings_call_sentiment_score": sentiment.get("sentiment_score"),
    }


def researcher_node(state: MarketState) -> dict:
    ticker = state["ticker"]
    print(f"\n{'='*60}")
    print(f"[RESEARCHER]  Fetching data for {ticker} ...")
    print(f"{'='*60}")

    # ── Core market data (now includes futures data) ─────────────────────
    market_data = get_market_data(ticker) # <-- UPDATED FUNCTION CALL
    company_name = market_data["company_name"]

    # ── News ──────────────────────────────────────────────────────────────
    print(f"[RESEARCHER]  Fetching news ...")
    news = fetch_recent_news(ticker, company_name)

    # ── SEC filings (risk factors) ────────────────────────────────────────
    print(f"[RESEARCHER]  Fetching SEC filings ...")
    sec = fetch_risk_factors(ticker)
    if sec.get("filing_type"):
        print(f"[RESEARCHER]  SEC {sec['filing_type']} parsed ({sec.get('filing_date', '?')})")
    else:
        print(f"[RESEARCHER]  SEC: {sec.get('risk_factors', 'unavailable')[:80]}")

    # ── Earnings call sentiment ───────────────────────────────────────────
    print(f"[RESEARCHER]  Analyzing earnings call sentiment ...")
    sentiment = fetch_earnings_sentiment(ticker, company_name)
    score = sentiment.get("sentiment_score")
    tone = sentiment.get("tone", "Unknown")
    print(f"[RESEARCHER]  Earnings tone: {tone}  |  Score: {score}/10")

    # ── Ground Truth JSON (now includes futures data) ───────────────────
    ground_truth = _build_ground_truth(market_data, sec, sentiment)

    # ── Build news block ──────────────────────────────────────────────────
    news_block = "\n".join(
        f"  [{i+1}] ({item['date']}) {item['source']}: {item['title']}\n"
        f"      {item['body']}"
        for i, item in enumerate(news)
    )

    # ── SEC risk factors block ────────────────────────────────────────────
    sec_block = (
        f"Source: {sec.get('filing_type', 'N/A')} filed {sec.get('filing_date', 'N/A')}\n"
        f"{sec.get('risk_factors', 'Not available')[:1500]}"
    )

    # ── Earnings sentiment block ──────────────────────────────────────────
    sentiment_block = json.dumps(sentiment, indent=2)

    prompt = f"""Ticker: {ticker}
Company: {company_name}  |  Sector: {market_data['sector']}  |  Industry: {market_data['industry']}

=== QUANTITATIVE MARKET DATA (Ground Truth) ===
{json.dumps(market_data, indent=2, default=str)}

=== RECENT NEWS (last 30 days) ===
{news_block}

=== SEC FILING — RISK FACTORS (most recent {sec.get('filing_type', '10-Q/10-K')}) ===
{sec_block}

=== EARNINGS CALL SENTIMENT ANALYSIS ===
{sentiment_block}

Produce a structured research brief with these sections:
1. **Price Action & Trend** — momentum, MAs, distance from 52w high/low
2. **Technical Signals** — RSI, MACD, Bollinger position, volume anomalies
3. **Fundamental Snapshot** — valuation multiples vs. sector norms, growth, margins, balance sheet
4. **Key Bullish Catalysts** — specific data points supporting upside
5. **Key Bearish Risks** — specific data points supporting downside
6. **Analyst Consensus** — target price, recommendation, short interest
7. **SEC Risk Factors** — top 3 material risks disclosed in the most recent filing
8. **Management Tone** — earnings call sentiment, guidance direction, and notable language
9. **Futures Basis Signal** — state whether {market_data.get('index_name', 'the index')} futures ({market_data.get('futures_symbol')}) are in Contango or Backwardation vs spot, report the exact basis ({market_data.get('basis_pct', 'N/A')}%), and note the current futures volume vs its 30-day average (ratio: {market_data.get('futures_volume_ratio', 'N/A')}x, level: {market_data.get('volatility_level', 'N/A')}). Do NOT interpret — just state the facts."""

    response = _llm.invoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
    )

    summary = _extract_text(response)
    print(f"\n[RESEARCHER]  Brief complete ({len(summary)} chars)")

    # The entire market data payload is passed along for broad access.
    # The curated 'ground_truth' is used for high-signal scoring and fact-checking.
    return {
        "research_data": {
            "market_data": market_data,
            "news": news,
            "sec_filings": sec,
            "earnings_sentiment": sentiment,
            "summary": summary,
        },
        "ground_truth": ground_truth,
        "debate_round": 0,
        "debate_history": [],
    }


def _extract_text(response) -> str:
    """Return plain text, ignoring thinking blocks."""
    if hasattr(response, "content") and isinstance(response.content, list):
        parts = [
            block.get("text", "") if isinstance(block, dict) else getattr(block, "text", "")
            for block in response.content
            if (isinstance(block, dict) and block.get("type") == "text")
            or (hasattr(block, "type") and block.type == "text")
        ]
        return "\n".join(parts).strip()
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content.strip()
    return str(response)
