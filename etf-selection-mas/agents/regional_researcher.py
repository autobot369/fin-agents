"""
Multi-Market Researcher Agent
------------------------------
Processes ALL THREE market segments (INTL, HKCN, BSE) in a single node pass,
accumulating ETF data and macro news into the shared `all_etf_data` /
`all_macro_news` state fields.

News source: 100% free — DDGS + yfinance.Ticker.news (no Tavily dependency).

Pipeline per market segment:
  1. Fetch ETF quantitative data (expense ratio, AUM, momentum) via yfinance.
  2. Fetch macro + ticker-level news via tools/news_search.py.
  3. Aggregate into all_etf_data / all_macro_news.

After all three markets are fetched, one LLM call produces a combined macro
brief covering the global outlook as of March 2026.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ETFSelectionState
from tools.bse_data import get_nse_tickers
from tools.etf_data import fetch_etf_universe
from tools.news_search import fetch_market_news

# ---------------------------------------------------------------------------
# ETF Universe — candidates per region (includes all spec-required tickers)
# ---------------------------------------------------------------------------
ETF_UNIVERSES = {
    "INTL": [
        "VXUS", "SPDW", "IEFA", "SCHY",          # spec-required
        "VEA", "EEM", "ACWI", "SCHF", "IXUS", "VWO", "IEMG",
    ],
    "HKCN": [
        "CQQQ", "KWEB", "MCHI", "FXI", "CHIQ",   # spec-required
        "GXC", "FLCH", "KURE", "CNYA", "ASHR",
    ],
    "BSE": get_nse_tickers(),                      # 10 NSE .NS tickers from registry
}

# Map region → exchange label used in output report
REGION_TO_MARKET = {
    "INTL": "NASDAQ",
    "HKCN": "NASDAQ",
    "BSE":  "NSE",
}

_llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=2500)

_SYSTEM = """\
You are a senior macro strategist writing a concise global research brief for a
quantitative ETF selection model. Today is March 2026.

Cover all three market segments: International ex-US (INTL), China/HK (HKCN),
and India BSE/NSE (BSE). Be factual, cite figures where available, and flag
data gaps. Under 500 words total. Structure as three short sections."""


def _build_etf_table(etf_data: dict, region: str) -> str:
    rows = [f"=== {region} ETFs ===",
            f"{'Ticker':<20} {'TER%':>6} {'AUM_B':>7} {'YTD%':>7} {'Mom3M%':>7} {'Mom1M%':>7}",
            "-" * 60]
    for ticker, rec in etf_data.items():
        if rec.get("region") != region:
            continue
        ter = f"{rec['expense_ratio']*100:.3f}" if rec.get("expense_ratio") else "N/A"
        aum = f"{rec['aum_b']:.1f}"             if rec.get("aum_b")         else "N/A"
        ytd = f"{rec['ytd_return']*100:.2f}"    if rec.get("ytd_return")    else "N/A"
        m3  = f"{rec['momentum_3m']*100:.2f}"   if rec.get("momentum_3m")   else "N/A"
        m1  = f"{rec['momentum_1m']*100:.2f}"   if rec.get("momentum_1m")   else "N/A"
        rows.append(f"{ticker:<20} {ter:>6} {aum:>7} {ytd:>7} {m3:>7} {m1:>7}")
    return "\n".join(rows)


def _build_news_snippet(macro_news: dict, max_per_query: int = 3) -> str:
    lines = []
    for query, articles in list(macro_news.items())[:12]:
        lines.append(f"\n[{query[:60]}]")
        for a in articles[:max_per_query]:
            lines.append(f"  • {a['title'][:90]}")
    return "\n".join(lines)


def multi_market_researcher_node(state: ETFSelectionState) -> dict:
    """
    LangGraph node: Multi-Market Researcher.

    Reads  : state["ter_threshold"]
    Writes : state["all_etf_data"], state["all_macro_news"],
             state["researcher_notes"]
    """
    print(f"\n{'='*60}")
    print("[MULTI_MARKET_RESEARCHER]  Processing INTL + HKCN + BSE ...")
    print(f"{'='*60}")

    all_etf_data: dict = {}
    all_macro_news: dict = {}
    cross_market_done = False   # inject cross-market queries only once

    for region, tickers in ETF_UNIVERSES.items():
        market_label = REGION_TO_MARKET[region]
        print(f"\n  ── {region} ({len(tickers)} tickers, exchange={market_label}) ──")

        # ── 1. yfinance quantitative data ─────────────────────────────────
        etf_records = fetch_etf_universe(tickers, region=region)

        # Stamp market field on every record
        for rec in etf_records.values():
            rec["market"] = market_label

        all_etf_data.update(etf_records)

        errors = [t for t, r in etf_records.items() if r.get("fetch_error")]
        if errors:
            print(f"  [WARN] yfinance errors: {errors}")

        # ── 2. Free news (DDGS + yfinance.Ticker.news) ───────────────────
        news = fetch_market_news(
            market=region,
            tickers=tickers,
            include_cross_market=not cross_market_done,  # only once
        )
        cross_market_done = True
        all_macro_news.update(news)

        total_arts = sum(len(v) for v in news.values())
        print(f"  Fetched {total_arts} articles for {region}")

    print(f"\n  Total ETFs    : {len(all_etf_data)}")
    print(f"  Total queries : {len(all_macro_news)}")
    print(f"  Total articles: {sum(len(v) for v in all_macro_news.values())}")

    # ── 3. LLM combined macro brief ───────────────────────────────────────
    print("\n[MULTI_MARKET_RESEARCHER]  Writing global macro brief ...")

    tables = "\n\n".join(
        _build_etf_table(all_etf_data, region)
        for region in ["INTL", "HKCN", "BSE"]
    )
    news_snippet = _build_news_snippet(all_macro_news)

    prompt = f"""Date Context: March 2026
Global ETF pipeline — processing INTL (NASDAQ ex-US), HKCN (China/HK), BSE (India NSE).

=== ETF QUANTITATIVE DATA ===
{tables}

=== RECENT NEWS HEADLINES ===
{news_snippet}

Write three concise sections: (1) INTL macro backdrop, (2) China/HK macro backdrop,
(3) India macro backdrop. Highlight any active boom/bust catalysts per region."""

    response = _llm.invoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
    )
    researcher_notes = _extract_text(response)
    print(f"[MULTI_MARKET_RESEARCHER]  Brief: {len(researcher_notes)} chars")

    return {
        "all_etf_data":   all_etf_data,
        "all_macro_news": all_macro_news,
        "researcher_notes": researcher_notes,
    }


def _extract_text(response) -> str:
    if hasattr(response, "content") and isinstance(response.content, list):
        parts = [
            b.get("text", "") if isinstance(b, dict) else getattr(b, "text", "")
            for b in response.content
            if (isinstance(b, dict) and b.get("type") == "text")
            or (hasattr(b, "type") and b.type == "text")
        ]
        return "\n".join(parts).strip()
    if hasattr(response, "content") and isinstance(response.content, str):
        return response.content.strip()
    return str(response)
