"""
Bear Agent — Two Nodes
-----------------------
bear_response_node : Round 1 — responds SPECIFICALLY to Bull's initial points.
bear_counter_node  : Round 2 — counter-argues Bull's rebuttal with new evidence.

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

_PERSONA = """You are a highly skeptical short-seller and risk manager with a
forensic accounting background. Your mandate: expose overvaluation, macro headwinds,
execution risk, competitive threats, and accounting red flags. You are NOT a
permabear — every claim must be backed by a specific data point from the research.
You MUST directly address the Bull's specific points — vague macro doom is not enough.
Be precise, cite numbers, and be ruthlessly skeptical."""

_HORIZON_GUIDE = {
    "short": (
        "1–3 months",
        "Focus on near-term risks: upcoming earnings misses, technical breakdown signals "
        "(RSI overbought, MACD cross, backwardation), short-squeeze risk, and "
        "imminent macro events. Long-term structural arguments are irrelevant here.",
    ),
    "mid": (
        "6–12 months",
        "Balance near-term execution risk with medium-term headwinds: "
        "valuation compression, guidance risk, and competitive threats that "
        "could materialise within the year.",
    ),
    "long": (
        "1–3 years",
        "Focus on structural risks: secular headwinds, margin erosion, "
        "balance sheet deterioration, SEC-disclosed risk factors, and "
        "competitive moat decay. Near-term technicals are noise — attack the thesis.",
    ),
}


def _context_block(state: MarketState) -> str:
    horizon = state.get("investment_horizon", "mid")
    risk = state.get("risk_aversity", 0.5)
    label, guidance = _HORIZON_GUIDE.get(horizon, _HORIZON_GUIDE["mid"])

    if risk <= 0.3:
        risk_note = "The investor is risk-seeking — make your bear case airtight; they will push back hard."
    elif risk >= 0.7:
        risk_note = "The investor is risk-averse — your downside risks will be weighted heavily. Make them specific."
    else:
        risk_note = "The investor has a balanced risk profile."

    return (
        f"INVESTMENT CONTEXT:\n"
        f"  Horizon : {horizon.upper()} ({label})\n"
        f"  Risk    : {risk:.1f}/1.0 — {risk_note}\n"
        f"  Focus   : {guidance}\n"
    )


def bear_response_node(state: MarketState) -> dict:
    rd = state["research_data"]
    ticker = state["ticker"]
    print(f"\n[BEAR - Round 1]  Formulating bear response for {ticker} ...")

    prompt = f"""{_context_block(state)}
RESEARCH BRIEF:
{rd['summary']}

BULL'S OPENING THESIS (you must dismantle this point by point):
{state['bull_case']}

Write your ROUND 1 BEAR RESPONSE. You must:
1. Quote each of Bull's major claims.
2. Counter each one with specific data or news from the research brief.
3. Add bearish risks that Bull deliberately ignored.
4. Propose a bear-case price target with justification.

## Bear Case — Round 1 Response

Structure: Address Bull's points sequentially, then add 2-3 additional risk factors
they glossed over. End with your downside price target and key risk triggers."""

    response = _llm.invoke(
        [SystemMessage(content=_PERSONA), HumanMessage(content=prompt)]
    )
    bear_case = _extract_text(response)
    print(f"[BEAR - Round 1]  Response complete.")

    return {
        "bear_case": bear_case,
        "debate_history": [{"role": "bear_response", "content": bear_case}],
    }


def bear_counter_node(state: MarketState) -> dict:
    rd = state["research_data"]
    ticker = state["ticker"]
    print(f"\n[BEAR - Round 2]  Formulating closing counter for {ticker} ...")

    prompt = f"""{_context_block(state)}
RESEARCH BRIEF:
{rd['summary']}

YOUR ROUND 1 BEAR RESPONSE:
{state['bear_case']}

BULL'S ROUND 2 REBUTTAL (you must counter this):
{state['bull_rebuttal']}

Write your ROUND 2 CLOSING COUNTER. This is your last word before the Arbiter rules.
Make it count:
1. Identify where Bull's rebuttal is weakest (hand-waving, cherry-picked data, etc.)
2. Reinforce your strongest bear arguments with any remaining data.
3. Highlight the single biggest tail risk the market is not pricing in.
4. Maintain or revise your downside target.

## Bear Closing Counter — Round 2"""

    response = _llm.invoke(
        [SystemMessage(content=_PERSONA), HumanMessage(content=prompt)]
    )
    bear_counter = _extract_text(response)
    print(f"[BEAR - Round 2]  Closing counter complete.")

    return {
        "bear_counter": bear_counter,
        "debate_history": [{"role": "bear_counter", "content": bear_counter}],
    }
