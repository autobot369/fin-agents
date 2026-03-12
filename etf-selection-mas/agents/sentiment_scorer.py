"""
Sentiment Scorer Agent  (v2 — Global Top-20 Edition)
======================================================
Converts raw DDGS / yfinance macro news into per-ETF sentiment scores and
generates a one-sentence boom rationale for every ETF that makes the top-20.

Two-stage pipeline (unchanged logic, updated boom triggers):

  Stage 1 — Article-level scoring (Claude)
  -----------------------------------------
  Batches of articles are sent to Claude, which assigns a score in [-1, +1]
  and confirms which BOOM_TRIGGERS are evidenced by the text.

  Stage 2 — ETF-level aggregation
  --------------------------------
  Scores are aggregated per region, boom-trigger boosts are applied, and
  results are min-max normalised to [0.0, 1.0].

Updated boom triggers vs v1:
  NEW  china_tech_rerating   — replaces china_ai_localization (broader scope)
  NEW  rbi_pivot             — replaces rbi_rate_cut (captures full pivot cycle)
  NEW  fed_soft_landing      — replaces fed_pause_cut (soft-landing narrative)
  KEPT india_inflation_target, china_gdp_beat, em_broad_rally
  KEPT china_tech_crackdown, india_fiscal_slippage (bust triggers)
  NEW  us_recession_risk     — bust trigger (risk-off, hurts all non-US)

Reads  : state["all_macro_news"], state["all_filtered_tickers"],
         state["all_etf_data"]
Writes : state["all_sentiment_scores"], state["boom_triggers_fired"],
         state["raw_article_scores"], state["sentiment_narrative"]
"""

import json
from typing import Any, Dict, List, Tuple

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from graph.state import ETFSelectionState

_llm = ChatAnthropic(model="claude-sonnet-4-6", max_tokens=4096)

# ---------------------------------------------------------------------------
# BOOM TRIGGER CATALOGUE  (March 2026 macro environment)
# ---------------------------------------------------------------------------
BOOM_TRIGGERS: Dict[str, Dict[str, Any]] = {

    # ══ BULLISH TRIGGERS ═══════════════════════════════════════════════════

    "china_tech_rerating": {
        "keywords": [
            "china tech re-rating", "deepseek", "china ai rally", "alibaba rebound",
            "tencent recovery", "kweb rally", "cqqq rally", "china internet surge",
            "china tech valuation re-rate", "baidu ai breakthrough", "huawei chips",
            "regulatory thaw china tech", "china big tech bull",
        ],
        "boost": 0.35,
        "applies_to": ["HKCN"],
        "hypothesis": (
            "China tech re-rating: regulatory overhang eases as AI localization "
            "accelerates post-DeepSeek, driving multiple expansion across KWEB/CQQQ/MCHI."
        ),
        "confidence_threshold": 0.10,
    },

    "rbi_pivot": {
        "keywords": [
            "rbi pivot", "rbi rate cut", "rbi repo rate reduction", "rbi easing cycle",
            "reserve bank india dovish", "india rate cut 2026", "rbi accommodative",
            "india monetary easing", "rbi cuts repo", "india interest rate cut",
        ],
        "boost": 0.30,
        "applies_to": ["BSE"],
        "hypothesis": (
            "RBI pivot to easing unlocks P/E expansion for Indian equities — "
            "directly boosts BANKBEES (rate-sensitive), INFRABEES, and JUNIORBEES."
        ),
        "confidence_threshold": 0.10,
    },

    "fed_soft_landing": {
        "keywords": [
            "fed soft landing", "us soft landing", "federal reserve soft landing",
            "fed pause rate cut", "fomc hold cut", "powell soft landing",
            "us no recession 2026", "us economy soft landing", "fed dovish pivot",
            "fed funds rate stable", "inflation under control fed",
        ],
        "boost": 0.25,
        "applies_to": ["INTL", "HKCN", "BSE"],
        "hypothesis": (
            "Fed soft-landing narrative: stable rates + controlled US inflation "
            "weakens USD and triggers rotation into international and EM ETFs globally."
        ),
        "confidence_threshold": 0.10,
    },

    "india_inflation_target": {
        "keywords": [
            "india cpi 2.75", "india inflation 2.75%", "india inflation target met",
            "india cpi below 3", "india retail inflation march 2026",
            "india price stability 2026", "rbi inflation mandate",
        ],
        "boost": 0.25,
        "applies_to": ["BSE"],
        "hypothesis": (
            "India CPI hitting the 4% ±2% band mid-point (2.75%) gives RBI "
            "headroom to cut, compressing discount rates for Indian equities."
        ),
        "confidence_threshold": 0.05,
    },

    "china_gdp_beat": {
        "keywords": [
            "china gdp beat", "china gdp 5%", "china q1 gdp 2026", "china growth target",
            "china economic expansion", "china gdp above forecast", "china gdp surprise",
            "china gdp outperforms", "china economy accelerates",
        ],
        "boost": 0.25,
        "applies_to": ["HKCN", "INTL"],
        "hypothesis": (
            "China GDP surpassing the 5% 2026 target ends the property-led slowdown "
            "narrative, re-rating China-heavy ETFs and boosting EM sentiment broadly."
        ),
        "confidence_threshold": 0.10,
    },

    "em_broad_rally": {
        "keywords": [
            "emerging market rally", "em inflows", "em etf inflows 2026",
            "developing market outperformance", "em equity march 2026",
            "global em rally", "em rotation 2026", "em allocation increase",
        ],
        "boost": 0.15,
        "applies_to": ["INTL", "BSE"],
        "hypothesis": (
            "Broad EM inflow cycle as global allocators rotate from expensive US "
            "equities — lifts VXUS/IEFA/EEM (INTL) and NIFTYBEES/BANKBEES (BSE)."
        ),
        "confidence_threshold": 0.05,
    },

    # ══ BEARISH TRIGGERS ═══════════════════════════════════════════════════

    "china_tech_crackdown": {
        "keywords": [
            "china tech crackdown", "china regulation tighten", "delisting risk china",
            "vie structure ban", "china sec enforcement", "ant group fine 2026",
            "didi regulatory action", "china internet regulation",
        ],
        "boost": -0.20,
        "applies_to": ["HKCN"],
        "hypothesis": (
            "Renewed Chinese tech regulation or US delisting threats reverse the "
            "re-rating thesis, compressing multiples on KWEB/CQQQ/FXI."
        ),
        "confidence_threshold": -0.10,
    },

    "india_fiscal_slippage": {
        "keywords": [
            "india fiscal deficit widens", "india debt gdp rises", "india fiscal slippage",
            "india bond yield spike", "india current account deficit",
            "india government spending overrun",
        ],
        "boost": -0.15,
        "applies_to": ["BSE"],
        "hypothesis": (
            "India fiscal slippage raises sovereign yields and crowds out private "
            "investment, capping BSE ETF upside and delaying RBI cuts."
        ),
        "confidence_threshold": -0.05,
    },

    "us_recession_risk": {
        "keywords": [
            "us recession 2026", "us hard landing", "federal reserve over-tightening",
            "us gdp contraction", "us employment falls", "us recession risk rising",
            "us slowdown 2026", "global recession risk",
        ],
        "boost": -0.15,
        "applies_to": ["INTL", "HKCN"],
        "hypothesis": (
            "US recession risk triggers global risk-off, hurting export-oriented "
            "INTL and HKCN ETFs through trade/demand channel."
        ),
        "confidence_threshold": -0.10,
    },
}

# ---------------------------------------------------------------------------
# Region routing: which region does a given query string primarily affect?
# ---------------------------------------------------------------------------
QUERY_REGION_MAP: Dict[str, str] = {
    "china":              "HKCN",
    "hong kong":          "HKCN",
    "pboc":               "HKCN",
    "kweb":               "HKCN",
    "cqqq":               "HKCN",
    "india":              "BSE",
    "rbi":                "BSE",
    "nifty":              "BSE",
    "nse":                "BSE",
    "fed ":               "ALL",
    "federal reserve":    "ALL",
    "soft landing":       "ALL",
    "msci world":         "INTL",
    "developed market":   "INTL",
    "international etf":  "INTL",
    "emerging market":    "INTL",
    "us dollar":          "ALL",
    "em inflows":         "INTL",
}

_SYSTEM_SCORER = """\
You are a quantitative macro analyst scoring news articles for ETF market impact.
For each article:

1. SENTIMENT_SCORE: assign -1.0 (very bearish) to +1.0 (very bullish) in 0.05 steps.
   Use 0.0 for truly neutral / irrelevant articles.

2. BOOM_TRIGGERS_CONFIRMED: list only trigger keys explicitly evidenced by the article
   text. Do NOT infer — require concrete statements (numbers, policy decisions, data).

3. REASON: one precise sentence citing the key fact that drove your score.

Return ONLY valid JSON — no markdown fences, no prose outside the object:
{
  "articles": [
    {
      "title": "<title>",
      "sentiment_score": <float>,
      "boom_triggers_confirmed": ["<key>", ...],
      "reason": "<sentence>"
    }
  ],
  "regional_summary": "<2-3 sentence macro narrative for this query region>"
}
"""


def _region_for_query(query: str) -> str:
    q = query.lower()
    for kw, region in QUERY_REGION_MAP.items():
        if kw in q:
            return region
    return "ALL"


def _catalogue_string() -> str:
    lines = []
    for key, spec in BOOM_TRIGGERS.items():
        sign = "BULLISH" if spec["boost"] > 0 else "BEARISH"
        lines.append(
            f"  {key} ({sign} {spec['boost']:+.2f} → {spec['applies_to']}): "
            f"{spec['hypothesis'][:90]}"
        )
    return "\n".join(lines)


def _score_batch(
    query: str,
    articles: List[Dict[str, Any]],
    catalogue: str,
) -> Tuple[List[Dict[str, Any]], str]:
    if not articles:
        return [], ""

    block = "\n\n".join(
        f"[{i+1}] Title: {a['title']}\n"
        f"Source: {a.get('source','?')}  Date: {a.get('date','?')}\n"
        f"Body: {a.get('body','')[:400]}"
        for i, a in enumerate(articles[:8])
    )

    prompt = (
        f'Query context: "{query}"\n\n'
        f"BOOM TRIGGER CATALOGUE:\n{catalogue}\n\n"
        f"ARTICLES:\n{block}\n\n"
        f"Score each article and return only the JSON."
    )

    resp = _llm.invoke([SystemMessage(content=_SYSTEM_SCORER), HumanMessage(content=prompt)])
    raw = _extract_text(resp)

    # Strip markdown fences if model wraps in ```json ... ```
    if raw.startswith("```"):
        raw = raw.split("```")[1]
        if raw.startswith("json"):
            raw = raw[4:]
        raw = raw.strip()

    try:
        parsed = json.loads(raw)
        return parsed.get("articles", []), parsed.get("regional_summary", "")
    except json.JSONDecodeError:
        print(f"  [scorer] JSON parse failed — query: {query[:55]}")
        return [], ""


def _aggregate(
    all_scored: Dict[str, List[Dict]],
    filtered_tickers: List[str],
    etf_data: Dict[str, Any],
) -> Dict[str, float]:
    """
    Aggregate article scores to per-ETF sentiment scores.
    Returns min-max normalised scores in [0.0, 1.0].
    """
    region_pools: Dict[str, List[float]] = {"INTL": [], "HKCN": [], "BSE": []}

    for query, articles in all_scored.items():
        region = _region_for_query(query)
        for art in articles:
            score = float(art.get("sentiment_score", 0.0))
            # yfinance articles get 1.3× weight (ticker-matched source)
            weight = 1.3 if art.get("fetch_src") == "yfinance" else 1.0
            weighted = score * weight

            if region == "ALL":
                for r in region_pools:
                    region_pools[r].append(weighted * 0.85)
            elif region in region_pools:
                region_pools[region].append(weighted)

    def _mean(lst):
        return sum(lst) / len(lst) if lst else 0.0

    base = {r: _mean(v) for r, v in region_pools.items()}

    # Resolve confirmed triggers
    confirmed: Dict[str, bool] = {k: False for k in BOOM_TRIGGERS}
    for articles in all_scored.values():
        for art in articles:
            art_score = float(art.get("sentiment_score", 0.0))
            for key in art.get("boom_triggers_confirmed", []):
                if key not in BOOM_TRIGGERS:
                    continue
                spec = BOOM_TRIGGERS[key]
                thresh = spec["confidence_threshold"]
                if (spec["boost"] > 0 and art_score >= thresh) or \
                   (spec["boost"] < 0 and art_score <= thresh):
                    confirmed[key] = True

    # Per-ticker scoring
    raw_scores: Dict[str, float] = {}
    for ticker in filtered_tickers:
        ticker_region = etf_data.get(ticker, {}).get("region", "ALL")
        score = base.get(ticker_region, 0.0)

        for key, fired in confirmed.items():
            if not fired:
                continue
            spec = BOOM_TRIGGERS[key]
            if ticker_region in spec["applies_to"] or "ALL" in spec["applies_to"]:
                score += spec["boost"]

        raw_scores[ticker] = max(-1.0, min(1.0, score))

    if not raw_scores:
        return {}

    lo, hi = min(raw_scores.values()), max(raw_scores.values())
    spread = hi - lo
    if spread < 1e-9:
        return {t: 0.5 for t in raw_scores}
    return {t: round((v - lo) / spread, 4) for t, v in raw_scores.items()}


def sentiment_scorer_node(state: ETFSelectionState) -> dict:
    """
    LangGraph node: Sentiment Scorer.

    Reads  : state["all_macro_news"], state["all_filtered_tickers"],
             state["all_etf_data"]
    Writes : state["all_sentiment_scores"], state["boom_triggers_fired"],
             state["raw_article_scores"], state["sentiment_narrative"]
    """
    macro_news       = state.get("all_macro_news", {})
    filtered_tickers = state.get("all_filtered_tickers", [])
    etf_data         = state.get("all_etf_data", {})

    print(f"\n{'='*60}")
    print(f"[SENTIMENT_SCORER]  Scoring {len(macro_news)} query batches "
          f"across {len(filtered_tickers)} tickers ...")
    print(f"{'='*60}")

    catalogue = _catalogue_string()
    all_scored: Dict[str, List[Dict]] = {}
    narratives: List[str] = []

    for query, articles in macro_news.items():
        scored, narrative = _score_batch(query, articles, catalogue)
        # Carry fetch_src forward for weighting in _aggregate
        for i, sa in enumerate(scored):
            if i < len(articles):
                sa["fetch_src"] = articles[i].get("fetch_src", "ddgs")
        all_scored[query] = scored
        if narrative:
            narratives.append(narrative)

        avg = (sum(a.get("sentiment_score", 0) for a in scored) / len(scored)
               if scored else 0.0)
        print(f"  [{_region_for_query(query):<4}] {query[:55]:<55}  avg={avg:+.2f}")

    # Collect fired boom triggers
    fired_triggers: List[str] = []
    for arts in all_scored.values():
        for art in arts:
            for key in art.get("boom_triggers_confirmed", []):
                if key in BOOM_TRIGGERS and key not in fired_triggers:
                    fired_triggers.append(key)

    print(f"\n  Boom triggers fired: {fired_triggers or ['none']}")

    # Aggregate to per-ETF normalised scores
    sentiment_scores = _aggregate(all_scored, filtered_tickers, etf_data)

    print(f"\n  Top sentiment scores:")
    for t, s in sorted(sentiment_scores.items(), key=lambda x: -x[1])[:10]:
        print(f"    {t:<22} {s:.4f}")

    # Build narrative
    narrative_md = _build_narrative(fired_triggers, sentiment_scores, narratives)

    return {
        "raw_article_scores":   all_scored,
        "all_sentiment_scores": sentiment_scores,
        "boom_triggers_fired":  fired_triggers,
        "sentiment_narrative":  narrative_md,
    }


def _build_narrative(
    fired: List[str],
    scores: Dict[str, float],
    llm_summaries: List[str],
) -> str:
    lines = ["## Sentiment Scorer — Global (March 2026)\n\n",
             "### Active Boom/Bust Triggers\n\n"]
    if fired:
        for key in fired:
            spec = BOOM_TRIGGERS.get(key, {})
            sign = "BULLISH" if spec.get("boost", 0) > 0 else "BEARISH"
            lines.append(
                f"- **`{key}`** [{sign} {spec.get('boost',0):+.2f}]"
                f" → `{spec.get('applies_to', [])}`  \n"
                f"  _{spec.get('hypothesis', '')}_\n\n"
            )
    else:
        lines.append("_No high-confidence triggers detected._\n\n")

    if llm_summaries:
        lines.append("### Regional Macro Summaries\n\n")
        for s in llm_summaries:
            lines.append(f"> {s}\n\n")

    lines.append("### Normalised Sentiment Scores\n\n")
    lines.append("| Ticker | Region | Score |\n|--------|--------|-------|\n")
    for t, s in sorted(scores.items(), key=lambda x: -x[1]):
        lines.append(f"| {t} | — | {s:.4f} |\n")

    return "".join(lines)


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
