"""
State schema for the Market Research Multi-Agent System.

debate_history uses operator.add so each node's returned list is
appended (not replaced) by LangGraph's state merging logic.
"""

import operator
from typing import Annotated, Any, Dict, List, TypedDict


class MarketState(TypedDict):
    # ── User inputs ───────────────────────────────────────────────────────
    ticker: str
    investment_horizon: str   # "short" (1-3mo) | "mid" (6-12mo) | "long" (1-3yr)
    risk_aversity: float      # 0.0 = risk-seeking  →  1.0 = risk-averse

    # ── Populated by Researcher ───────────────────────────────────────────
    research_data: Dict[str, Any]

    # Quantitative anchor — verified numbers the Arbiter treats as authoritative
    ground_truth: Dict[str, Any]

    # ── Debate rounds ─────────────────────────────────────────────────────
    bull_case: str      # Bull's opening thesis
    bear_case: str      # Bear's direct rebuttal to Bull's points
    bull_rebuttal: str  # Bull defends / escalates against Bear's counter
    bear_counter: str   # Bear's closing counter-argument

    # Full debate transcript (auto-appended via operator.add)
    debate_history: Annotated[List[Dict[str, str]], operator.add]

    # ── Guardrail & output ────────────────────────────────────────────────
    fact_check_report: str
    final_forecast: str

    # Internal round tracker
    debate_round: int
