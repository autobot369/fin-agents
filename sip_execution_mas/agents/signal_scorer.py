"""
Node 2 — Signal Scorer (Gemini)
================================
Gemini reads ETF metrics + macro news headlines and assigns:
  - sentiment_score per ETF (0.0 – 1.0)
  - boom_triggers  : list of region/macro conditions that fired
  - macro_summary  : 2–3 sentence macro outlook

Falls back to VADER if GEMINI_API_KEY is missing or Gemini fails.
"""
from __future__ import annotations

import json
import os
from datetime import date as _Date
from typing import Any, Dict, List, Optional, Tuple

from sip_execution_mas.graph.state import SIPExecutionState

# ── Gemini client initialisation ──────────────────────────────────────────────

def _get_gemini_model():
    import google.generativeai as genai
    api_key = os.environ.get("GEMINI_API_KEY")
    if not api_key:
        raise EnvironmentError("GEMINI_API_KEY not set")
    genai.configure(api_key=api_key)
    return genai.GenerativeModel(
        model_name="gemini-2.0-flash",
        generation_config=genai.GenerationConfig(
            response_mime_type="application/json",
            temperature=0.1,
        ),
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


# ── VADER fallback ────────────────────────────────────────────────────────────

def _vader_fallback(
    filtered_tickers: List[str],
    all_macro_news: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Regional VADER sentiment → propagated to each ETF in that region.
    Returns ticker → sentiment_score (0.0 – 1.0).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    region_scores: Dict[str, float] = {}
    for region, headlines in all_macro_news.items():
        if not headlines:
            region_scores[region] = 0.50
            continue
        compounds = [sia.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(compounds) / len(compounds)
        # Normalise [-1, 1] → [0.10, 0.90]
        region_scores[region] = round(0.10 + (avg + 1) / 2 * 0.80, 4)

    scores: Dict[str, float] = {}
    for ticker in filtered_tickers:
        if ticker.endswith(".NS") or ticker.endswith(".BO"):
            scores[ticker] = region_scores.get("BSE", 0.50)
        elif ticker in ("CQQQ", "KWEB", "MCHI", "FXI", "CHIQ",
                        "GXC", "FLCH", "KURE", "CNYA", "ASHR"):
            scores[ticker] = region_scores.get("HKCN", 0.50)
        else:
            scores[ticker] = region_scores.get("INTL", 0.50)
    return scores


# ── Gemini prompt builder ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a quantitative ETF signal scorer for a systematic investment system.

Given ETF metrics and macro news headlines, assign a sentiment score (0.0 to 1.0) to each ETF:
  - 0.0 – 0.35 : Bearish / risk-off
  - 0.35 – 0.65: Neutral / mixed
  - 0.65 – 1.0 : Bullish / risk-on

Also identify boom triggers: macro conditions that strongly boost or suppress a region.
Examples of boom triggers: "CHINA_STIMULUS", "INDIA_GROWTH_OUTPERFORM", "EMERGING_MARKETS_RALLY",
"US_RATE_CUT", "GLOBAL_RISK_OFF", "CHINA_TECH_CRACKDOWN".

Return ONLY valid JSON in exactly this format:
{
  "scores": {
    "TICKER1": 0.72,
    "TICKER2": 0.45
  },
  "boom_triggers": ["TRIGGER1", "TRIGGER2"],
  "macro_summary": "2-3 sentence macro outlook."
}

Rules:
- Every ticker in the input must appear in "scores"
- Scores must be between 0.0 and 1.0
- boom_triggers is an empty list [] if no notable macro conditions
- macro_summary must be 2-3 sentences maximum
"""


def _build_user_prompt(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    all_macro_news: Dict[str, List[str]],
) -> str:
    lines = ["=== ETF METRICS ==="]
    for ticker in filtered_tickers:
        rec = all_etf_data.get(ticker, {})
        ter = rec.get("expense_ratio")
        ter_str = f"{ter*100:.2f}%" if ter else "N/A"
        mom3 = rec.get("momentum_3m")
        mom3_str = f"{mom3:+.1f}%" if mom3 is not None else "N/A"
        ytd = rec.get("ytd_return")
        ytd_str = f"{ytd:+.1f}%" if ytd is not None else "N/A"
        region = rec.get("region", "?")
        lines.append(
            f"{ticker:<22} region={region}  TER={ter_str}  "
            f"3m={mom3_str}  YTD={ytd_str}"
        )

    lines.append("\n=== MACRO NEWS HEADLINES ===")
    for region, headlines in all_macro_news.items():
        lines.append(f"\n[{region}]")
        for h in headlines[:8]:   # cap at 8 headlines per region
            lines.append(f"  • {h}")

    lines.append(f"\n=== TICKERS TO SCORE ===")
    lines.append(", ".join(filtered_tickers))

    return "\n".join(lines)


# ── Extracted scorer — callable from both node and backtest ───────────────────

def score_etfs(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    all_macro_news: Dict[str, List[str]],
    reference_date: Optional[_Date] = None,
) -> Tuple[Dict[str, float], List[str], str]:
    """
    Score ETF sentiment via Gemini (with optional date isolation) or VADER.

    Args:
        filtered_tickers: Tickers to score.
        all_etf_data:     ETF metrics dict (from regional_researcher output).
        all_macro_news:   Headlines per region (from regional_researcher output).
        reference_date:   None → production (no date constraint in prompt).
                          date → backtest (Gemini instructed to evaluate as-of that date).

    Returns:
        (sentiment_scores, boom_triggers, macro_summary)
    """
    # Build system prompt — inject date header for backtest mode
    if reference_date:
        month_year  = reference_date.strftime("%B %Y")
        iso         = reference_date.isoformat()
        date_header = (
            "SIMULATION MODE: You are scoring as of {my} ({iso}).\n"
            "Score based ONLY on the provided news articles and ETF metrics.\n"
            "Do NOT incorporate market events, earnings, or macro changes from after {iso}.\n\n"
        ).format(my=month_year, iso=iso)
        system_prompt = date_header + _SYSTEM_PROMPT
    else:
        system_prompt = _SYSTEM_PROMPT

    sentiment_scores: Dict[str, float] = {}
    boom_triggers: List[str] = []
    macro_summary = ""
    used_fallback = False

    try:
        model = _get_gemini_model()
        user_content = _build_user_prompt(filtered_tickers, all_etf_data, all_macro_news)
        full_prompt  = system_prompt + "\n\n" + user_content

        response = model.generate_content(full_prompt)
        parsed   = _parse_json(response.text)

        raw_scores = parsed.get("scores", {})
        for ticker in filtered_tickers:
            val = raw_scores.get(ticker)
            sentiment_scores[ticker] = (
                max(0.0, min(1.0, float(val))) if val is not None else 0.50
            )

        boom_triggers = parsed.get("boom_triggers", [])
        macro_summary = parsed.get("macro_summary", "")

    except EnvironmentError as exc:
        print(f"  [scorer] ⚠  {exc} — falling back to VADER")
        used_fallback = True
    except Exception as exc:
        print(f"  [scorer] ⚠  Gemini error: {exc} — falling back to VADER")
        used_fallback = True

    if used_fallback:
        sentiment_scores = _vader_fallback(filtered_tickers, all_macro_news)
        macro_summary    = "Macro scores computed via VADER sentiment (Gemini unavailable)."
        boom_triggers    = []

    source = "VADER fallback" if used_fallback else "Gemini"
    date_tag = f" [as-of {reference_date}]" if reference_date else ""
    print(f"  [scorer] Scores via {source}{date_tag}")
    if boom_triggers:
        print(f"  [scorer] Boom triggers: {', '.join(boom_triggers)}")
    print(f"  [scorer] Macro: {macro_summary[:120]}")

    return sentiment_scores, boom_triggers, macro_summary


# ── Node function (thin wrapper around score_etfs) ────────────────────────────

def signal_scorer_node(state: SIPExecutionState) -> dict:
    """
    Node 2 — Signal Scorer.
    Delegates to score_etfs(); threads as_of_date from state for backtest support.
    """
    filtered_tickers = state["filtered_tickers"]
    all_etf_data     = state["all_etf_data"]
    all_macro_news   = state["all_macro_news"]
    reference_date   = state.get("as_of_date")   # None in production

    print(f"\n[Node 2] Signal Scorer — scoring {len(filtered_tickers)} ETFs …")

    sentiment_scores, boom_triggers, macro_summary = score_etfs(
        filtered_tickers = filtered_tickers,
        all_etf_data     = all_etf_data,
        all_macro_news   = all_macro_news,
        reference_date   = reference_date,
    )

    return {
        "sentiment_scores": sentiment_scores,
        "boom_triggers":    boom_triggers,
        "macro_summary":    macro_summary,
    }
