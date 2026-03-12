"""
Market Research MAS — Programmatic API
=======================================
Import and call run_analysis() to invoke the full pipeline from Python code.

Example
-------
    from api import run_analysis

    result = run_analysis("NVDA", investment_horizon="short", risk_aversity=0.8)

    print(result["recommendation"])   # "HOLD"
    print(result["conviction"])        # "LOW"
    print(result["volatility_warning"])# True
    print(result["final_forecast"])    # full Markdown report
"""

import datetime
import re
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from dotenv import load_dotenv

load_dotenv()

# ── Types ─────────────────────────────────────────────────────────────────────

AnalysisResult = Dict[str, Any]

# ── Helpers ───────────────────────────────────────────────────────────────────

_HORIZON_LABELS = {
    "short": "1–3 months",
    "mid":   "6–12 months",
    "long":  "1–3 years",
}

def _validate_inputs(ticker: str, investment_horizon: str, risk_aversity: float) -> None:
    if not ticker or not ticker.strip():
        raise ValueError("ticker must be a non-empty string.")
    if investment_horizon not in _HORIZON_LABELS:
        raise ValueError(f"investment_horizon must be 'short', 'mid', or 'long'. Got: {investment_horizon!r}")
    if not 0.0 <= risk_aversity <= 1.0:
        raise ValueError(f"risk_aversity must be between 0.0 and 1.0. Got: {risk_aversity}")


def _parse_field(text: str, field: str) -> Optional[str]:
    """Extract a bold markdown field value: **Field:** value"""
    match = re.search(rf"\*\*{re.escape(field)}:\*\*\s*(.+?)(?:\n|$)", text)
    return match.group(1).strip() if match else None


def _detect_volatility_warning(text: str) -> bool:
    """Return True if the Arbiter issued a Volatility Warning."""
    return bool(re.search(r"Volatility Warning\*\*:\s*YES", text, re.IGNORECASE))


# ── Main function ─────────────────────────────────────────────────────────────

def run_analysis(
    ticker: str,
    investment_horizon: str = "mid",
    risk_aversity: float = 0.5,
    save: bool = False,
    verbose: bool = False,
) -> AnalysisResult:
    """
    Run the full Market Research MAS pipeline and return structured results.

    Parameters
    ----------
    ticker : str
        Stock ticker symbol (e.g. "AAPL", "NVDA", "TSLA").

    investment_horizon : str
        Time frame for the analysis:
        - "short" → 1–3 months  (technical signals and near-term catalysts dominate)
        - "mid"   → 6–12 months (balanced technicals + fundamentals)
        - "long"  → 1–3 years   (structural fundamentals dominate)

    risk_aversity : float  [0.0 – 1.0]
        0.0 = risk-seeking  → volatility warnings are softened; Bull-weighted scoring.
        0.5 = balanced      → standard behaviour.
        1.0 = risk-averse   → volatility warnings can downgrade the recommendation;
                              Bear risk warnings receive extra weight.

    save : bool
        If True, write the final forecast to a Markdown file in the current directory.
        Filename format: TICKER_YYYYMMDD_HHMMSS_forecast.md

    verbose : bool
        If True, print the full debate transcript to stdout while running.

    Returns
    -------
    dict with keys:
        ticker              : str   — normalised ticker symbol
        investment_horizon  : str   — horizon used ("short" | "mid" | "long")
        risk_aversity       : float — risk aversity used
        horizon_label       : str   — human label e.g. "6–12 months"
        recommendation      : str   — e.g. "BUY", "HOLD", "SELL"  (None if unparseable)
        conviction          : str   — e.g. "HIGH", "MEDIUM", "LOW" (None if unparseable)
        volatility_warning  : bool  — True if Arbiter issued a volatility warning
        basis_state         : str   — "Contango" | "Backwardation" | None
        volatility_level    : str   — "NORMAL" | "ELEVATED" | "SPIKE" | None
        final_forecast      : str   — full Markdown report
        ground_truth        : dict  — verified quantitative anchor
        fact_check_report   : str   — per-claim verification report
        debate_history      : list  — list of {"role": ..., "content": ...} dicts
        saved_to            : str   — absolute path of saved file, or None
    """
    _validate_inputs(ticker, investment_horizon, risk_aversity)
    ticker = ticker.upper().strip()

    print(f"\n{'█' * 60}")
    print(f"  Market Research MAS  ·  {ticker}")
    print(f"  Horizon : {investment_horizon.upper()} ({_HORIZON_LABELS[investment_horizon]})")
    print(f"  Risk    : {risk_aversity:.2f}/1.00")
    print(f"  LangGraph + Claude Opus 4.6 (Adaptive Thinking)")
    print(f"{'█' * 60}")

    from graph.workflow import build_graph

    graph = build_graph()

    initial_state = {
        "ticker": ticker,
        "investment_horizon": investment_horizon,
        "risk_aversity": risk_aversity,
        "research_data": {},
        "ground_truth": {},
        "bull_case": "",
        "bear_case": "",
        "bull_rebuttal": "",
        "bear_counter": "",
        "debate_history": [],
        "fact_check_report": "",
        "final_forecast": "",
        "debate_round": 0,
    }

    try:
        final_state = graph.invoke(initial_state)
    except Exception as e:
        print(f"\n[ERROR]  Graph execution failed: {e}", file=sys.stderr)
        raise

    forecast = final_state["final_forecast"]
    ground_truth = final_state.get("ground_truth", {})

    if verbose:
        for entry in final_state.get("debate_history", []):
            role = entry["role"].upper().replace("_", " ")
            print(f"\n{'─' * 60}\n  {role}\n{'─' * 60}")
            print(entry["content"])

    print(f"\n{'─' * 60}")
    print(f"  FINAL CONSENSUS FORECAST")
    print(f"{'─' * 60}")
    print(forecast)
    print(f"\n{'█' * 60}\n")

    # ── Save to disk ──────────────────────────────────────────────────────
    saved_to = None
    if save:
        ts = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
        path = Path(f"{ticker}_{ts}_forecast.md")
        path.write_text(forecast, encoding="utf-8")
        saved_to = str(path.resolve())
        print(f"[INFO]  Forecast saved to: {saved_to}")

    return {
        "ticker": ticker,
        "investment_horizon": investment_horizon,
        "risk_aversity": risk_aversity,
        "horizon_label": _HORIZON_LABELS[investment_horizon],
        "recommendation": _parse_field(forecast, "Recommendation"),
        "conviction": _parse_field(forecast, "Conviction"),
        "volatility_warning": _detect_volatility_warning(forecast),
        "basis_state": ground_truth.get("basis_state"),
        "volatility_level": ground_truth.get("volatility_level"),
        "final_forecast": forecast,
        "ground_truth": ground_truth,
        "fact_check_report": final_state.get("fact_check_report", ""),
        "debate_history": final_state.get("debate_history", []),
        "saved_to": saved_to,
    }
