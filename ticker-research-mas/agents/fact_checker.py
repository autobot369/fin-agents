"""
Fact-Checker Node
-----------------
Runs between bear_counter and arbiter.

Scans all four debate turns for specific numerical claims and cross-checks
each against the verified ground_truth JSON produced by the Researcher.
Flags:
  ✅ CONFIRMED   — claim matches ground truth within 5% tolerance
  ❌ CONTRADICTED — claim directly conflicts with ground truth
  ⚠️  LOGICAL MISUSE — correct number, wrong conclusion drawn from it
  🔍 UNVERIFIABLE — cannot be checked from provided ground truth

The resulting report is passed to the Arbiter, which must discount any
agent whose claims were CONTRADICTED or LOGICALLY MISUSED.
"""

import json

from langchain_anthropic import ChatAnthropic
from langchain_core.messages import HumanMessage, SystemMessage

from agents.researcher import _extract_text
from graph.state import MarketState

_llm = ChatAnthropic(
    model="claude-opus-4-6",
    max_tokens=2000,
    thinking={"type": "adaptive"},
)

_SYSTEM = """You are a forensic financial fact-checker with access to verified market data.
Your only job: scan debate arguments for specific numerical claims and cross-check each
against the Ground Truth JSON.

Tolerance rules:
- CONFIRMED ✅: number matches ground truth within 5% (or is directionally correct for booleans)
- CONTRADICTED ❌: number directly conflicts with ground truth values
- LOGICAL MISUSE ⚠️: the number is correct but the conclusion drawn misrepresents it
  (e.g., citing a high RSI as proof of momentum when it actually signals overbought)
- UNVERIFIABLE 🔍: claim references data not present in ground truth

Be terse. No preamble. Output only the structured Markdown report."""


def fact_checker_node(state: MarketState) -> dict:
    ticker = state["ticker"]
    ground_truth = state["ground_truth"]
    print(f"\n{'='*60}")
    print(f"[FACT-CHECKER]  Cross-checking numerical claims for {ticker} ...")
    print(f"{'='*60}")

    debate_block = f"""=== BULL — ROUND 1 (Initial Thesis) ===
{state['bull_case']}

=== BEAR — ROUND 1 (Response) ===
{state['bear_case']}

=== BULL — ROUND 2 (Rebuttal) ===
{state['bull_rebuttal']}

=== BEAR — ROUND 2 (Closing Counter) ===
{state['bear_counter']}"""

    prompt = f"""GROUND TRUTH — verified market data (treat as authoritative):
{json.dumps(ground_truth, indent=2)}

FULL DEBATE TRANSCRIPT:
{debate_block}

Produce the fact-check report in EXACTLY this Markdown structure:

---

## Fact-Check Report — {ticker}

### ✅ Confirmed Claims
| Agent | Claimed | Ground Truth | Note |
|-------|---------|--------------|------|

### ❌ Contradicted Claims
| Agent | Claimed | Ground Truth | Correction |
|-------|---------|--------------|------------|

### ⚠️ Logical Misuse (Correct Number, Wrong Conclusion)
| Agent | Claim | Why the Logic Fails |
|-------|-------|---------------------|

### 🔍 Unverifiable Claims
| Agent | Claim |
|-------|-------|

### Summary
- **Bull reliability score:** X/10
- **Bear reliability score:** X/10
- **Most egregious error:** [one sentence, or "None"]
- **Arbiter guidance:** [one sentence on how the Arbiter should weight each side]

---"""

    response = _llm.invoke(
        [SystemMessage(content=_SYSTEM), HumanMessage(content=prompt)]
    )
    report = _extract_text(response)

    print(f"[FACT-CHECKER]  Report complete ({len(report)} chars)")

    return {
        "fact_check_report": report,
        "debate_history": [{"role": "fact_checker", "content": report}],
    }
