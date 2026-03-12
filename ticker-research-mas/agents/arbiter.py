"""
Arbiter Agent Node (V2)
-----------------------
Scores the Bull and Bear debate anchored to three objective signals:

  A. Futures Basis  — Contango reinforces Bull; Backwardation reinforces Bear.
  B. Volume Volatility — futures_volume_ratio > 1.5x (30d avg) triggers a
                         Volatility Warning even if agents agree on direction.
  C. Contextual Scoring — 10-point Recognition + Interpretation rubric.

The Fact-Check report feeds into Recognition scoring (❌ claims → 0 pts).
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.researcher import _extract_text
from graph.state import MarketState

_llm = ChatAnthropic(
    model="claude-opus-4-6",
    max_tokens=4500,
    thinking={"type": "adaptive"},
)

# Ratio thresholds for volume-induced volatility (today vs 30-day average)
_VOL_ELEVATED = 1.5   # ≥ 1.5x avg  → ELEVATED (note in report)
_VOL_SPIKE    = 2.0   # ≥ 2.0x avg  → SPIKE    (mandatory warning)

_PERSONA = f"""You are an objective, data-driven Portfolio Manager with 20 years running
a long/short equity fund. You arbitrate structured debates and issue final investment
rulings. Your scoring is anchored to three objective signals — ignore rhetoric,
reward evidence, penalize errors.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL A — FUTURES BASIS (Price Discovery)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Compare `futures_price` vs `spot_price` from the Ground Truth.

• CONTANGO (futures > spot, positive basis_pct):
  The market prices in a risk premium for future delivery — a macro tailwind
  for the Bull case. Stronger basis → stronger tailwind.
  → Add up to +1 pt to Bull's Interpretation score.

• BACKWARDATION (spot > futures, negative basis_pct):
  The market demands immediate delivery premium — signals near-term fear or
  supply squeeze. Strong tailwind for Bear's risk warnings.
  → Add up to +1 pt to Bear's Interpretation score.

• NEUTRAL (|basis_pct| < 0.1%): No directional bias from this signal.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL B — VOLUME-INDUCED VOLATILITY
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Use `futures_volume_ratio` (today / 30-day avg) from Ground Truth.

• NORMAL  (ratio < {_VOL_ELEVATED}x):  No special action.
• ELEVATED (ratio {_VOL_ELEVATED}–{_VOL_SPIKE}x): Note increased volatility risk in the report.
• SPIKE   (ratio ≥ {_VOL_SPIKE}x): You MUST issue a ⚠️ VOLATILITY WARNING in the
  final recommendation block, regardless of directional bias.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
SIGNAL C — CONTEXTUAL SCORING (10-point rubric per agent)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
Score each agent out of 10 using two dimensions:

RECOGNITION (0–5 pts) — Factual accuracy and data usage:
  1. Correctly cited price / technical data (RSI, MACD, MAs)
  2. Correctly cited valuation metrics (P/E, PEG, margins)
  3. Referenced SEC risk factors from the filing
  4. Referenced earnings call tone / guidance signal
  5. Referenced the futures basis signal
  → Deduct 1 pt per fact flagged ❌ CONTRADICTED in the Fact-Check report.
  → An agent with ANY ❌ CONTRADICTED claim cannot score 5/5 on Recognition.

INTERPRETATION (0–5 pts) — Insight quality and logical rigor:
  1. Identified a genuine mispricing or asymmetric risk/reward
  2. Drew a non-obvious, data-backed conclusion (not just restating data)
  3. Correctly interpreted the direction implied by technical signals
  4. Correctly framed the futures basis signal's implication
  5. Identified the single most material catalyst or tail risk
  → Add Signal A bonus (+1) to Interpretation if the agent correctly called
    the futures basis implication in their thesis.
  → Deduct 1 pt per ⚠️ LOGICAL MISUSE flagged in the Fact-Check report.

━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━
REQUIRED OUTPUT FORMAT (Markdown — do not deviate)
━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━

# {{TICKER}} — Arbiter's Final Ruling

## 1. Signal Analysis

### A. Futures Basis (Price Discovery)
| Field               | Value |
|---------------------|-------|
| Index               |       |
| Futures Symbol      |       |
| Futures Price       |       |
| Spot Price          |       |
| Basis               |       |
| Basis %             |       |
| Structure           | Contango / Backwardation |
| Directional Bias    | Supports Bull / Bear / Neutral |

*Interpretation:* [1–2 sentences on what this basis implies for the thesis]

### B. Volume-Induced Volatility
| Field                         | Value |
|-------------------------------|-------|
| Futures Volume (today)        |       |
| Futures Volume (30d avg)      |       |
| Volume Ratio                  |       |
| Volatility Level              | NORMAL / ELEVATED / SPIKE |

*Implication:* [1 sentence]

---

## 2. Contextual Scorecard

| Dimension                              | Bull | Bear |
|----------------------------------------|:----:|:----:|
| Recognition — Price/Technical data     |  /1  |  /1  |
| Recognition — Valuation metrics        |  /1  |  /1  |
| Recognition — SEC risk factors         |  /1  |  /1  |
| Recognition — Earnings call signal     |  /1  |  /1  |
| Recognition — Futures basis signal     |  /1  |  /1  |
| **Recognition Subtotal**               | **/5** | **/5** |
| Interpretation — Mispricing/asymmetry  |  /1  |  /1  |
| Interpretation — Non-obvious insight   |  /1  |  /1  |
| Interpretation — Technical direction   |  /1  |  /1  |
| Interpretation — Futures basis framing |  /1  |  /1  |
| Interpretation — Key catalyst/risk     |  /1  |  /1  |
| **Interpretation Subtotal**            | **/5** | **/5** |
| Signal A Bonus (futures basis)         |  +0/+1 | +0/+1 |
| Fact-Check Deductions                  |  −   |  −   |
| **TOTAL**                              | **/10** | **/10** |

**Scoring notes:** [2–3 sentences — cite specific Fact-Check deductions and Signal A bonus allocation]

---

## 3. Final Synthesis

[2–3 paragraphs: who won, why, how the futures basis tilted the ruling, and
what the volatility level means for position sizing]

---

## 4. Recommendation

| Scenario  | Price Target | Probability | Key Trigger |
|-----------|:------------:|:-----------:|-------------|
| Bull Case |              |     %       |             |
| Base Case |              |     %       |             |
| Bear Case |              |     %       |             |

**Recommendation:** [STRONG BUY / BUY / HOLD / SELL / STRONG SELL]
**Conviction:** [HIGH / MEDIUM / LOW]
**Time Horizon:** [e.g., 6–12 months]
⚠️ **Volatility Warning:** [YES — reason / NO]

---

## 5. Risks to Monitor
1. ...
2. ...
3. ...
"""


def arbiter_node(state: MarketState) -> dict:
    ticker = state["ticker"]
    ground_truth = state["ground_truth"]
    fact_check_report = state.get("fact_check_report", "No fact-check report available.")

    # company_name lives in ground_truth (added in _build_ground_truth)
    company_name = ground_truth.get("company_name", ticker)

    print(f"\n{'='*60}")
    print(f"[ARBITER V2]  Scoring {ticker} against anchor signals ...")
    vl = ground_truth.get("volatility_level", "N/A")
    bs = ground_truth.get("basis_state", "N/A")
    vr = ground_truth.get("futures_volume_ratio", "N/A")
    print(f"[ARBITER V2]  Basis: {bs}  |  Vol ratio: {vr}x  |  Level: {vl}")
    print(f"{'='*60}")

    debate_transcript = "\n\n".join(
        f"--- {entry['role'].upper().replace('_', ' ')} ---\n{entry['content']}"
        for entry in state["debate_history"]
        if entry["role"] != "fact_checker"
    )

    # ── Build context-specific scoring rules ─────────────────────────────
    horizon = state.get("investment_horizon", "mid")
    risk = state.get("risk_aversity", 0.5)

    horizon_rules = {
        "short": (
            "SHORT (1–3 months)",
            "Weight technical Recognition rows (Price/Technical, Futures Basis) "
            "at 60% of the Recognition score. Fundamentals (Valuation, SEC) = 40%. "
            "Price targets must reflect a 1–3 month window.",
        ),
        "mid": (
            "MID (6–12 months)",
            "Equal weighting across all Recognition rows. "
            "Price targets must reflect a 6–12 month window.",
        ),
        "long": (
            "LONG (1–3 years)",
            "Weight fundamental Recognition rows (Valuation, SEC, Earnings) "
            "at 70% of the Recognition score. Technical rows = 30%. "
            "Price targets must reflect a 1–3 year window. "
            "Futures basis is relevant only for entry timing, not the core thesis.",
        ),
    }
    horizon_label, horizon_instruction = horizon_rules.get(horizon, horizon_rules["mid"])

    if risk <= 0.3:
        risk_instruction = (
            "RISK-SEEKING (0.0–0.3): The Volatility Warning does NOT change the "
            "directional recommendation — note it but maintain conviction. "
            "Bear's risk warnings receive standard weight."
        )
    elif risk >= 0.7:
        risk_instruction = (
            "RISK-AVERSE (0.7–1.0): Any ELEVATED volatility level drops Conviction "
            "by one step (HIGH→MEDIUM, MEDIUM→LOW). A SPIKE forces the recommendation "
            "down by one step (e.g. BUY→HOLD) and conviction to LOW. "
            "Bear's Risk Acknowledgement row is weighted 1.5× in final tiebreaking."
        )
    else:
        risk_instruction = (
            "BALANCED (0.3–0.7): Standard weighting. SPIKE triggers the Volatility "
            "Warning and reduces conviction by one step."
        )

    prompt = f"""TICKER: {ticker}  |  {company_name}

=== INVESTOR CONTEXT ===
Investment Horizon : {horizon_label}
Risk Aversity      : {risk:.2f}/1.00

Horizon scoring rule : {horizon_instruction}
Risk scoring rule    : {risk_instruction}

=== GROUND TRUTH (verified anchor — do not contradict) ===
{json.dumps(ground_truth, indent=2, default=str)}

=== FACT-CHECK REPORT (drives Recognition deductions) ===
{fact_check_report}

=== FULL DEBATE TRANSCRIPT ===
{debate_transcript}

=== INSTRUCTIONS ===
Apply the three-signal scoring framework from your system instructions,
adjusted by the Investor Context above.

Step 1 — Signal A: determine Contango/Backwardation bias and award the +1 bonus.
Step 2 — Signal B: check futures_volume_ratio; apply the risk-aversity rule
          to determine if a Volatility Warning is required and whether it
          affects the recommendation or conviction.
Step 3 — Signal C: score each agent on the 10-point rubric using the
          horizon-adjusted weighting for Recognition rows.
Step 4 — Write final synthesis and recommendation. Price target timeline
          must match the investment horizon.

Generate the complete report in the required Markdown format."""

    response = _llm.invoke(
        [SystemMessage(content=_PERSONA), HumanMessage(content=prompt)]
    )
    forecast = _extract_text(response)

    print(f"\n[ARBITER V2]  Ruling complete ({len(forecast)} chars)")

    return {"final_forecast": forecast}
