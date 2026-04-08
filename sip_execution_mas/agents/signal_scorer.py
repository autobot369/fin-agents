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
import time
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
        model_name="gemini-2.5-flash",
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


def _is_rate_limit_error(exc: Exception) -> bool:
    """Detect Gemini 429 / ResourceExhausted errors from any SDK version."""
    exc_str = str(exc).upper()
    if "429" in exc_str or "RESOURCE_EXHAUSTED" in exc_str or "RATE_LIMIT" in exc_str:
        return True
    try:
        from google.api_core.exceptions import ResourceExhausted
        if isinstance(exc, ResourceExhausted):
            return True
    except ImportError:
        pass
    return False


# ── Ticker → thematic category (shared by prompt builder and VADER fallback) ──
# v4 universe: 5 ETFs across 3 thematic categories

_TICKER_CATEGORY: Dict[str, str] = {
    "VWRA.L":       "QUALITY_CORE",
    "NIFTYBEES.NS": "INDIA_EM",
    "IUIT.L":       "TECH_SEMIS",
    "WSML.L":       "QUALITY_CORE",
    "MOM100.NS":    "INDIA_EM",
}


# ── VADER fallback ────────────────────────────────────────────────────────────

def _vader_fallback(
    filtered_tickers: List[str],
    all_macro_news: Dict[str, List[str]],
) -> Dict[str, float]:
    """
    Category-aware VADER sentiment fallback (used when Gemini is unavailable).
    Scores each category group from its thematic headlines, then maps each
    ticker to its category. Falls back to overall average when a category
    has no headlines.
    Returns ticker → sentiment_score (0.0 – 1.0).
    """
    from vaderSentiment.vaderSentiment import SentimentIntensityAnalyzer
    sia = SentimentIntensityAnalyzer()

    # Score each category and also collect an overall average
    cat_scores: Dict[str, float] = {}
    all_compounds: List[float] = []
    for key, headlines in all_macro_news.items():
        if not headlines:
            cat_scores[key] = 0.50
            continue
        compounds = [sia.polarity_scores(h)["compound"] for h in headlines]
        avg = sum(compounds) / len(compounds)
        cat_scores[key] = round(0.10 + (avg + 1) / 2 * 0.80, 4)
        all_compounds.extend(compounds)

    if all_compounds:
        _avg = sum(all_compounds) / len(all_compounds)
        overall = round(0.10 + (_avg + 1) / 2 * 0.80, 4)
    else:
        overall = 0.50

    scores: Dict[str, float] = {}
    for ticker in filtered_tickers:
        cat = _TICKER_CATEGORY.get(ticker)
        if cat and cat in cat_scores:
            scores[ticker] = cat_scores[cat]
        elif ticker.endswith(".NS") or ticker.endswith(".BO"):
            scores[ticker] = cat_scores.get("INDIA_EM", overall)
        else:
            scores[ticker] = overall
    return scores


# ── Gemini prompt builder ─────────────────────────────────────────────────────

_SYSTEM_PROMPT = """You are a quantitative macro strategist managing a 10-year horizon systematic portfolio.

Your task: score each ETF from 0.00 to 1.00 to determine monthly capital allocation weights within its bucket.

Scoring scale:
  0.00 – 0.35 : Structural headwinds / risk-off — reduce satellite weight
  0.35 – 0.65 : Neutral — standard weight allocation
  0.65 – 1.00 : Structural tailwinds / risk-on — increase satellite weight

Scoring framework — apply these rules IN ORDER:

  1. STRUCTURAL TAILWINDS OVERRIDE SHORT-TERM DRAWDOWNS.
     A 3-month price dip during a semiconductor inventory correction does NOT lower a score
     if the 10-year AI capex cycle is intact. Do not confuse cyclical noise with secular trend reversal.

  2. FORWARD P/E IS A VALUATION CEILING.
     fwdPE > 40x for tech sectors → cap score at 0.75 regardless of news.
     fwdPE > 30x for broad market → cap score at 0.80.
     Low fwdPE (<15x) with positive news → score can reach 0.90+.

  3. BETA AND RISK-REGIME ADJUSTMENT — SATELLITE VS CORE DISTINCTION.
     For CORE ETFs (VWRA.L — global all-world UCITS; NIFTYBEES.NS — India Nifty 50):
       Beta > 1.3 in a risk-off macro environment → reduce score by 0.05–0.10.
       Beta < 0.8 in a risk-off environment → increase score by 0.05 (defensive benefit confirmed).
     For SATELLITE ETFs (IUIT.L — S&P 500 tech UCITS; WSML.L — World small-cap UCITS; MOM100.NS — India midcap):
       Do NOT penalize high beta during macro panics or risk-off regimes.
       If beta > 1.3 AND sector news confirms the structural tailwind is INTACT
       (AI capex cycle, EM manufacturing rotation, India growth, etc.),
       MAINTAIN the score or INCREASE by +0.05. High-beta Satellite ETFs at discounted
       prices with intact structural catalysts are aggressive accumulation targets on a
       10-year horizon — treat them accordingly.
       Only reduce a Satellite score for beta if sector news is ALSO structurally negative
       (policy reversal, technology obsolescence, or permanent demand destruction).

  4. DIVIDEND YIELD SIGNALS QUALITY.
     Rising dividend yield on VWRA.L is a cash-flow quality signal → +0.05 to score.

  5. SECTOR NEWS IS PAIRED DIRECTLY WITH EACH ETF — use it to assess real capital flows,
     not sentiment. A headline about $40B hyperscaler capex is structural for SOXQ.
     A subsidy announcement is structural for ICLN. Distinguish policy from noise.

Identify boom triggers: macro conditions strongly boosting or suppressing a category.
Examples: "AI_CAPEX_CYCLE", "CLEAN_ENERGY_POLICY_TAILWIND", "INDIA_FII_INFLOWS",
"EM_SUPPLY_CHAIN_ROTATION", "US_BUYBACK_SURGE", "FED_PIVOT", "SEMICONDUCTOR_UPCYCLE",
"ESG_REGULATORY_TIGHTENING", "INDIA_RATE_CUT", "TECH_VALUATION_COMPRESSION".

Return ONLY valid JSON in exactly this format:
{
  "scores": {
    "TICKER1": 0.72,
    "TICKER2": 0.45
  },
  "boom_triggers": ["TRIGGER1", "TRIGGER2"],
  "macro_summary": "2-3 sentence 10-year horizon macro outlook."
}

Rules:
- Every ticker in the input must appear in "scores"
- Scores must be between 0.0 and 1.0
- boom_triggers is an empty list [] if no notable macro conditions
- macro_summary must be 2-3 sentences, 10-year structural perspective only
"""


def _build_user_prompt(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    all_macro_news: Dict[str, List[str]],
) -> str:
    """
    Structured per-ETF prompt block. Each entry contains:
      - Ticker, sector category, region
      - Hard fundamentals: fwdPE, beta, TER, 3m momentum, YTD, dividend yield
      - Sector-specific news headlines paired inline

    Pairing valuation (COST) directly with thematic news (CATALYST) lets
    Gemini weigh them together rather than treating them as separate sections.
    """
    lines = [
        "=== ETF PORTFOLIO — 10-YEAR HORIZON SCORING ===\n",
        "For each ETF: weigh [Fundamentals] (valuation/risk anchor) against "
        "[Sector News] (structural capital flow catalysts).\n",
    ]

    for ticker in filtered_tickers:
        rec      = all_etf_data.get(ticker, {})
        ter      = rec.get("expense_ratio")
        mom3     = rec.get("momentum_3m")
        ytd      = rec.get("ytd_return")
        fpe      = rec.get("forward_pe")
        beta     = rec.get("beta")
        div_yld  = rec.get("dividend_yield")
        category = rec.get("category", "?")
        region   = rec.get("region", "?")

        ter_str  = f"{ter*100:.2f}%"     if ter     is not None else "N/A"
        mom3_str = f"{mom3:+.1f}%"      if mom3    is not None else "N/A"
        ytd_str  = f"{ytd:+.1f}%"       if ytd     is not None else "N/A"
        fpe_str  = f"{fpe:.1f}x"        if fpe     is not None else "N/A"
        beta_str = f"{beta:.2f}"        if beta    is not None else "N/A"
        div_str  = f"{div_yld*100:.2f}%" if div_yld is not None else "N/A"

        # Pair inline sector news for this ticker's thematic category
        theme     = _TICKER_CATEGORY.get(ticker, "QUALITY_CORE")
        headlines = all_macro_news.get(theme, [])

        lines.append(f"[ETF: {ticker}] {category} | region={region}")
        lines.append(
            f"  [Fundamentals] fwdPE={fpe_str}  beta={beta_str}  TER={ter_str}"
            f"  3m={mom3_str}  YTD={ytd_str}  divYield={div_str}"
        )
        lines.append(f"  [Sector News — {theme}]")
        for h in (headlines[:4] if headlines else ["(no headlines available)"]):
            lines.append(f"    • {h}")
        lines.append("")   # blank line between ETFs

    lines.append("=== TICKERS TO SCORE ===")
    lines.append(", ".join(filtered_tickers))

    return "\n".join(lines)


# ── Extracted scorer — callable from both node and backtest ───────────────────

_RETRY_DELAYS = (15, 30, 60)   # seconds between consecutive 429 retries


def score_etfs(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    all_macro_news: Dict[str, List[str]],
    reference_date: Optional[_Date] = None,
    _force_vader: bool = False,
) -> Tuple[Dict[str, float], List[str], str, str]:
    """
    Score ETF sentiment via Gemini (with optional date isolation) or VADER.

    Args:
        filtered_tickers: Tickers to score.
        all_etf_data:     ETF metrics dict (from regional_researcher output).
        all_macro_news:   Headlines per category (from regional_researcher output).
        reference_date:   None → production (no date constraint in prompt).
                          date → backtest (Gemini instructed to evaluate as-of that date).
        _force_vader:     True → skip Gemini entirely; use VADER for all tickers.
                          Useful for backtest speed runs or when GEMINI_API_KEY is absent.

    Returns:
        (sentiment_scores, boom_triggers, macro_summary, scorer_source)
        scorer_source is "gemini" or "vader".
    """
    date_tag = f" [as-of {reference_date}]" if reference_date else ""

    # ── VADER-only mode (backtest escape hatch — zero Gemini calls) ───────────
    if _force_vader:
        sentiment_scores = _vader_fallback(filtered_tickers, all_macro_news)
        macro_summary    = "Macro scores computed via VADER sentiment (LLM disabled)."
        boom_triggers: List[str] = []
        print(f"  [scorer] Scores via VADER (forced){date_tag}")
        return sentiment_scores, boom_triggers, macro_summary, "vader"

    # ── Build system prompt — inject date header for backtest mode ────────────
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

    sentiment_scores_: Dict[str, float] = {}
    boom_triggers_: List[str] = []
    macro_summary_ = ""
    used_fallback = False

    # ── Gemini call with exponential backoff on 429 ───────────────────────────
    for _attempt in range(len(_RETRY_DELAYS) + 1):
        try:
            model        = _get_gemini_model()
            user_content = _build_user_prompt(filtered_tickers, all_etf_data, all_macro_news)
            full_prompt  = system_prompt + "\n\n" + user_content

            response = model.generate_content(full_prompt)
            parsed   = _parse_json(response.text)

            raw_scores = parsed.get("scores", {})
            for ticker in filtered_tickers:
                val = raw_scores.get(ticker)
                sentiment_scores_[ticker] = (
                    max(0.0, min(1.0, float(val))) if val is not None else 0.50
                )

            boom_triggers_ = parsed.get("boom_triggers", [])
            macro_summary_ = parsed.get("macro_summary", "")
            break   # success — exit retry loop

        except EnvironmentError as exc:
            print(f"  [scorer] ⚠  {exc} — falling back to VADER")
            used_fallback = True
            break

        except Exception as exc:
            if _is_rate_limit_error(exc) and _attempt < len(_RETRY_DELAYS):
                delay = _RETRY_DELAYS[_attempt]
                print(
                    f"  [scorer] ⚠  Gemini 429 rate limit — "
                    f"retrying in {delay}s "
                    f"(attempt {_attempt + 1}/{len(_RETRY_DELAYS) + 1}) …"
                )
                time.sleep(delay)
            else:
                msg = "429 max retries exhausted" if _is_rate_limit_error(exc) else str(exc)
                print(f"  [scorer] ⚠  Gemini error: {msg} — falling back to VADER")
                used_fallback = True
                break

    if used_fallback:
        sentiment_scores_ = _vader_fallback(filtered_tickers, all_macro_news)
        macro_summary_     = "Macro scores computed via VADER sentiment (Gemini unavailable)."
        boom_triggers_     = []

    scorer_source = "vader" if used_fallback else "gemini"
    source        = "VADER fallback" if used_fallback else "Gemini"
    print(f"  [scorer] Scores via {source}{date_tag}")
    if boom_triggers_:
        print(f"  [scorer] Boom triggers: {', '.join(boom_triggers_)}")
    print(f"  [scorer] Macro: {macro_summary_[:120]}")

    return sentiment_scores_, boom_triggers_, macro_summary_, scorer_source


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

    sentiment_scores, boom_triggers, macro_summary, scorer_source = score_etfs(
        filtered_tickers = filtered_tickers,
        all_etf_data     = all_etf_data,
        all_macro_news   = all_macro_news,
        reference_date   = reference_date,
    )

    return {
        "sentiment_scores": sentiment_scores,
        "boom_triggers":    boom_triggers,
        "macro_summary":    macro_summary,
        "scorer_source":    scorer_source,
    }
