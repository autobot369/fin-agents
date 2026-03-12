"""
Bull Agent — Two Nodes
----------------------
bull_initial_node  : Opens with an aggressively optimistic thesis grounded in data.
bull_rebuttal_node : Round 2 — directly rebuts Bear's specific counter-arguments.

Both nodes read investment_horizon and risk_aversity from state to frame
the argument at the right time scale and risk sensitivity.
"""

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.researcher import _extract_text
from graph.state import MarketState

_llm = ChatAnthropic(
    model="claude-opus-4-6",
    max_tokens=2500,
    thinking={"type": "adaptive"},
)

_PERSONA = """You are an aggressively optimistic equity analyst and fund manager.
Your mandate: identify asymmetric upside, alpha-generating catalysts, and market
mispricings. You are NOT a mindless permabull — every claim must be backed by a
specific data point or news item from the research brief. Speak with conviction.
Use precise numbers. Structure your argument to be maximally persuasive."""

_HORIZON_GUIDE = {
    "short": (
        "1–3 months",
        "Prioritise near-term catalysts, technical setup (RSI, MACD, futures basis), "
        "and upcoming earnings/events. Fundamentals like PEG are secondary — "
        "emphasise price action, momentum, and imminent catalysts.",
    ),
    "mid": (
        "6–12 months",
        "Balance technicals with fundamentals. Cite both near-term momentum "
        "and medium-term earnings growth / valuation expansion thesis.",
    ),
    "long": (
        "1–3 years",
        "Focus on structural thesis: FCF trajectory, competitive moat, "
        "revenue CAGR, and margin expansion. Technical signals are noise — "
        "build the case on durable fundamental and SEC-disclosed catalysts.",
    ),
}


def _context_block(state: MarketState) -> str:
    horizon = state.get("investment_horizon", "mid")
    risk = state.get("risk_aversity", 0.5)
    label, guidance = _HORIZON_GUIDE.get(horizon, _HORIZON_GUIDE["mid"])

    if risk <= 0.3:
        risk_note = "The investor is risk-seeking — lean into high-conviction, high-upside calls."
    elif risk >= 0.7:
        risk_note = "The investor is risk-averse — your upside case must still acknowledge downside guardrails."
    else:
        risk_note = "The investor has a balanced risk profile."

    return (
        f"INVESTMENT CONTEXT:\n"
        f"  Horizon : {horizon.upper()} ({label})\n"
        f"  Risk    : {risk:.1f}/1.0 — {risk_note}\n"
        f"  Focus   : {guidance}\n"
    )


def bull_initial_node(state: MarketState) -> dict:
    rd = state["research_data"]
    ticker = state["ticker"]
    print(f"\n[BULL - Round 1]  Building initial bull case for {ticker} ...")

    prompt = f"""{_context_block(state)}
RESEARCH BRIEF:
{rd['summary']}

Build your OPENING BULL THESIS for {ticker}. Structure it as:

## Bull Case — Initial Thesis

### 1. Core Value Proposition
(Why is this stock mispriced to the upside right now, given the investment horizon?)

### 2. Technical Setup
(Cite RSI, MACD, MA alignment, futures basis signal — what does the tape say?)

### 3. Fundamental Drivers
(Valuation, growth, margins, FCF — weight these appropriately for the horizon)

### 4. Catalyst Pipeline
(Specific upcoming events or news that will unlock value within the horizon)

### 5. Risk-Adjusted Return Profile
(Bull target price, timeline matching the horizon, key assumption)

Be specific, cite the numbers, and be ruthlessly optimistic."""

    response = _llm.invoke(
        [SystemMessage(content=_PERSONA), HumanMessage(content=prompt)]
    )
    bull_case = _extract_text(response)
    print(f"[BULL - Round 1]  Initial thesis complete.")

    return {
        "bull_case": bull_case,
        "debate_round": 1,
        "debate_history": [{"role": "bull_initial", "content": bull_case}],
    }


def bull_rebuttal_node(state: MarketState) -> dict:
    rd = state["research_data"]
    ticker = state["ticker"]
    print(f"\n[BULL - Round 2]  Formulating rebuttal for {ticker} ...")

    prompt = f"""{_context_block(state)}
RESEARCH BRIEF:
{rd['summary']}

YOUR INITIAL BULL THESIS (Round 1):
{state['bull_case']}

BEAR'S RESPONSE (that you must rebut):
{state['bear_case']}

Now write your ROUND 2 REBUTTAL. You must:
1. Address EACH of Bear's specific objections by number.
2. Flip their bearish evidence into a bullish interpretation wherever possible.
3. Introduce any additional bullish data points not yet cited.
4. Double down on your price target or revise it with justification.

## Bull Rebuttal — Round 2

Structure: For each bear point, quote it briefly then dismantle it with data."""

    response = _llm.invoke(
        [SystemMessage(content=_PERSONA), HumanMessage(content=prompt)]
    )
    rebuttal = _extract_text(response)
    print(f"[BULL - Round 2]  Rebuttal complete.")

    return {
        "bull_rebuttal": rebuttal,
        "debate_round": 2,
        "debate_history": [{"role": "bull_rebuttal", "content": rebuttal}],
    }
