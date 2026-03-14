"""
Node 4 — Risk Auditor (Guardrail + Crash-Accumulator Value-Averaging)
======================================================================
Hard rule-checker. Never approves a portfolio that violates:

  Rule 1  MAX_POSITION  — no single ETF > max_position_pct of total SIP
  Rule 2  MAX_REGION    — no region > max_region_pct of total SIP
  Rule 3  MIN_POSITION  — no ETF allocated < $1.00 USD
  Rule 4  NO_DUPLICATE  — ledger already has an entry for this calendar month
  Rule 5  MIN_ORDERS    — at least 1 valid order must exist

If violations exist:
  - retry_count < MAX_RETRIES  → reject (loops back to Node 3)
  - retry_count >= MAX_RETRIES → hard abort

Crash-Accumulator Value-Averaging (applied only on clean approval):
  Multi-tiered SIP scaling triggered when Node 2 signals macro panic.
  Panic condition : any known risk-off trigger in boom_triggers,
                    OR portfolio-average sentiment < PANIC_SENTIMENT_THRESHOLD.

  Tier 1 — Standard Dip   : panic present AND -15.0% < avg_momentum_3m <= 0.0%
                              → va_multiplier = 1.20  (+20% SIP)
  Tier 2 — Generational Crash: panic present AND avg_momentum_3m <= -15.0%
                              → va_multiplier = 1.50  (+50% SIP)

  When triggered, all proposed_order monthly_usd values are scaled by the
  applicable multiplier and weights are recomputed.  The effective SIP is
  written back to state as sip_amount so Node 5 executes at the correct size.
  va_triggered / va_multiplier are set in state for auditing and Streamlit.

Gemini explains violations and suggests corrective actions in natural language.
Falls back to template explanation if Gemini unavailable.
"""
from __future__ import annotations

import json
import os
import sys
from datetime import date
from pathlib import Path
from typing import Any, Dict, List, Optional

from sip_execution_mas.graph.state import ProposedOrder, SIPExecutionState

_ROOT     = Path(__file__).resolve().parent.parent.parent   # fin-agents/
_SIP_ROOT = Path(__file__).resolve().parent.parent          # sip_execution_mas/
if str(_SIP_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIP_ROOT))

MAX_RETRIES = 2

# ── Value-Averaging constants ─────────────────────────────────────────────────

# Tier 1: standard dip — panic present AND momentum between -15% and 0%
VA_MULTIPLIER_TIER1 = 1.20    # +20% SIP

# Tier 2: generational crash — panic present AND momentum <= -15%
VA_MULTIPLIER_TIER2 = 1.50    # +50% SIP

# Momentum threshold separating Tier 1 from Tier 2
MOMENTUM_CRASH_THRESHOLD = -15.0   # percent

# Sentiment threshold below which the portfolio is considered to be in panic
PANIC_SENTIMENT_THRESHOLD = 0.35

# Boom triggers produced by Node 2 that count as a panic signal
_PANIC_TRIGGER_TOKENS = {
    "GLOBAL_RISK_OFF", "MARKET_CRASH", "SYSTEMIC_RISK", "RECESSION_FEAR",
    "CREDIT_CRUNCH", "BANKING_CRISIS", "LIQUIDITY_CRISIS", "BEAR_MARKET",
    "GEOPOLITICAL_RISK", "RATE_SHOCK", "CHINA_TECH_CRACKDOWN",
    "EMERGING_MARKETS_CRASH", "FLASH_CRASH", "VOLATILITY_SPIKE",
}


# ── Hard rule checks ──────────────────────────────────────────────────────────

def _check_rules(
    proposed_orders: List[ProposedOrder],
    sip_amount: float,
    max_position_pct: float,
    max_region_pct: float,
    ledger_path: str,
) -> List[str]:
    violations: List[str] = []

    # Rule 5: must have orders
    if not proposed_orders:
        violations.append("RULE_5_VIOLATION: No proposed orders generated.")
        return violations   # no point checking further

    # Rule 1: per-position cap
    position_cap = sip_amount * max_position_pct
    for order in proposed_orders:
        if order["monthly_usd"] > position_cap:
            pct = order["monthly_usd"] / sip_amount * 100
            violations.append(
                f"RULE_1_VIOLATION: {order['ticker']} allocated ${order['monthly_usd']:.2f} "
                f"({pct:.1f}% of SIP) exceeds max {max_position_pct*100:.0f}% "
                f"(${position_cap:.2f})."
            )

    # Rule 2: per-region cap
    region_totals: Dict[str, float] = {}
    for order in proposed_orders:
        r = order.get("region", "INTL")
        region_totals[r] = region_totals.get(r, 0) + order["monthly_usd"]

    region_cap = sip_amount * max_region_pct
    for region, total in region_totals.items():
        if total > region_cap + 0.01:   # 1-cent tolerance for rounding
            pct = total / sip_amount * 100
            violations.append(
                f"RULE_2_VIOLATION: Region {region} allocated ${total:.2f} "
                f"({pct:.1f}% of SIP) exceeds max {max_region_pct*100:.0f}% "
                f"(${region_cap:.2f})."
            )

    # Rule 3: minimum position
    for order in proposed_orders:
        if order["monthly_usd"] < 1.0:
            violations.append(
                f"RULE_3_VIOLATION: {order['ticker']} allocated ${order['monthly_usd']:.2f} "
                f"is below minimum $1.00."
            )

    # Rule 4: duplicate month check
    try:
        from simulator.ledger import load_ledger, already_invested_this_month  # type: ignore
        ledger = load_ledger(ledger_path)
        prev = already_invested_this_month(ledger)
        if prev:
            violations.append(
                f"RULE_4_VIOLATION: Ledger already has an investment this month ({prev}). "
                f"Use --force to override."
            )
    except Exception:
        pass   # ledger not found → not a violation

    return violations


# ── Gemini explanation ────────────────────────────────────────────────────────

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


_AUDIT_PROMPT = """You are a risk compliance officer reviewing a proposed ETF portfolio allocation.

Given the following rule violations, provide:
1. A concise explanation of each violation
2. Specific corrective actions the portfolio optimizer should take
3. An overall risk assessment (MINOR / MODERATE / SEVERE)

Return ONLY valid JSON:
{
  "explanations": ["...", "..."],
  "corrections": ["...", "..."],
  "risk_level": "MINOR" | "MODERATE" | "SEVERE",
  "audit_summary": "2-sentence overall assessment."
}

Violations to assess:
"""


def _gemini_audit_notes(
    violations: List[str],
    proposed_orders: List[ProposedOrder],
    sip_amount: float,
) -> str:
    if not violations:
        return "All risk checks passed. Portfolio approved."

    try:
        model = _get_gemini_model()
        context_lines = [
            "PORTFOLIO SUMMARY:",
            f"  Total SIP: ${sip_amount:.2f}",
            f"  Positions: {len(proposed_orders)}",
        ]
        for o in proposed_orders:
            pct = o["monthly_usd"] / sip_amount * 100 if sip_amount else 0
            context_lines.append(
                f"  {o['ticker']:<20} {o['region']:<5} ${o['monthly_usd']:>7.2f} ({pct:.1f}%)"
            )
        context_lines.append("\nVIOLATIONS:")
        for v in violations:
            context_lines.append(f"  - {v}")

        prompt = _AUDIT_PROMPT + "\n".join(context_lines)
        response = model.generate_content(prompt)
        parsed = _parse_json(response.text)

        lines = [f"Risk Level: {parsed.get('risk_level', 'UNKNOWN')}"]
        lines.append(parsed.get("audit_summary", ""))
        lines.append("\nExplanations:")
        for exp in parsed.get("explanations", []):
            lines.append(f"  • {exp}")
        lines.append("\nCorrections:")
        for cor in parsed.get("corrections", []):
            lines.append(f"  → {cor}")
        return "\n".join(lines)

    except Exception as exc:
        # Template fallback
        lines = [f"AUDIT FAILED — {len(violations)} violation(s) detected."]
        for v in violations:
            lines.append(f"  • {v}")
        lines.append("\nRecommendation: Reduce overweight positions and re-run optimizer.")
        return "\n".join(lines)


# ── Value-Averaging helpers ───────────────────────────────────────────────────

def _detect_va_condition(
    tickers: List[str],
    sentiment_scores: Dict[str, float],
    all_etf_data: Dict[str, Any],
    boom_triggers: List[str],
) -> tuple:
    """
    Return (multiplier: float, reason: str).

    multiplier = 1.0  → no Value-Averaging
    multiplier = 1.20 → Tier 1 Standard Dip
    multiplier = 1.50 → Tier 2 Generational Crash

    Panic condition : any _PANIC_TRIGGER_TOKENS token in boom_triggers,
                      OR portfolio-average sentiment < PANIC_SENTIMENT_THRESHOLD.

    Tier 1 : panic present AND MOMENTUM_CRASH_THRESHOLD < avg_momentum_3m <= 0.0%
    Tier 2 : panic present AND avg_momentum_3m <= MOMENTUM_CRASH_THRESHOLD

    tickers: list of ticker strings (e.g. filtered_tickers or [o["ticker"] for o in orders])
    """
    if not tickers:
        return 1.0, ""

    # ── Panic check ───────────────────────────────────────────────────────────
    trigger_tokens   = {t.upper() for t in boom_triggers}
    panic_from_token = bool(trigger_tokens & _PANIC_TRIGGER_TOKENS)

    sentiments = [sentiment_scores.get(t, 0.50) for t in tickers]
    avg_sentiment = sum(sentiments) / len(sentiments)
    panic_from_sentiment = avg_sentiment < PANIC_SENTIMENT_THRESHOLD

    is_panic = panic_from_token or panic_from_sentiment

    if not is_panic:
        return 1.0, ""

    # ── Momentum tier check ───────────────────────────────────────────────────
    mom_values = [
        all_etf_data.get(t, {}).get("momentum_3m")
        for t in tickers
    ]
    valid_moms = [m for m in mom_values if m is not None]
    if not valid_moms:
        return 1.0, ""

    avg_momentum = sum(valid_moms) / len(valid_moms)

    panic_detail = (
        f"trigger(s)={sorted(trigger_tokens & _PANIC_TRIGGER_TOKENS)}"
        if panic_from_token
        else f"avg_sentiment={avg_sentiment:.3f} < {PANIC_SENTIMENT_THRESHOLD}"
    )

    if avg_momentum <= MOMENTUM_CRASH_THRESHOLD:
        # Tier 2 — Generational Crash: avg momentum below -15%
        reason = (
            f"VA TIER 2 (Generational Crash) — panic: {panic_detail}; "
            f"avg_mom3m={avg_momentum:.1f}% ≤ {MOMENTUM_CRASH_THRESHOLD}%"
        )
        return VA_MULTIPLIER_TIER2, reason
    elif avg_momentum <= 0.0:
        # Tier 1 — Standard Dip: avg momentum between -15% and 0%
        reason = (
            f"VA TIER 1 (Standard Dip) — panic: {panic_detail}; "
            f"avg_mom3m={avg_momentum:.1f}% in ({MOMENTUM_CRASH_THRESHOLD}%, 0%]"
        )
        return VA_MULTIPLIER_TIER1, reason
    else:
        # Positive momentum with panic signal — no VA (market not yet discounted)
        return 1.0, ""


def _apply_va_multiplier(
    proposed_orders: List[ProposedOrder],
    base_sip: float,
    multiplier: float,
) -> tuple:
    """
    Scale every order's monthly_usd by multiplier, recompute weights.
    Returns (scaled_orders, effective_sip).
    """
    effective_sip = round(base_sip * multiplier, 2)
    scaled: List[ProposedOrder] = []
    for order in proposed_orders:
        o = dict(order)
        o["monthly_usd"] = round(o["monthly_usd"] * multiplier, 2)
        o["weight"]      = round(o["monthly_usd"] / effective_sip, 5)
        scaled.append(o)  # type: ignore[arg-type]
    return scaled, effective_sip


# ── Node function ─────────────────────────────────────────────────────────────

def risk_auditor_node(state: SIPExecutionState) -> dict:
    """
    Node 4 — Risk Auditor

    Checks proposed_orders against hard limits.
    - approved=True  → (optionally applies Value-Averaging) → Node 5
    - approved=False, retry_count < MAX_RETRIES → back to Node 3
    - approved=False, retry_count >= MAX_RETRIES → Node 6 (abort)

    Value-Averaging: on clean approval, checks for panic + momentum-tier condition.
    If detected, scales all order amounts by the tier multiplier (1.20 or 1.50) and
    updates sip_amount in state so Node 5 executes at the correct size.
    """
    proposed_orders  = state["proposed_orders"]
    sip_amount       = state["sip_amount"]
    max_position_pct = state["max_position_pct"]
    max_region_pct   = state["max_region_pct"]
    retry_count      = state.get("audit_retry_count", 0)
    ledger_path      = state.get("ledger_path",
                           str(_SIP_ROOT / "simulator" / "outputs" / "portfolio_ledger.json"))
    dry_run          = state.get("dry_run", True)
    force            = state.get("force", False)
    # Rule 4 (no duplicate month) is bypassed in paper mode or when --force is set
    skip_rule4       = dry_run or force

    # Node 2 outputs needed for Value-Averaging detection
    sentiment_scores = state.get("sentiment_scores", {})
    all_etf_data     = state.get("all_etf_data", {})
    boom_triggers    = state.get("boom_triggers", [])

    print(f"\n[Node 4] Risk Auditor — auditing {len(proposed_orders)} orders …")
    if force and not dry_run:
        print(f"  [Node 4] --force active: Rule 4 (duplicate month) bypassed")

    effective_ledger = "__skip_rule4__" if skip_rule4 else ledger_path

    violations = _check_rules(
        proposed_orders, sip_amount, max_position_pct, max_region_pct,
        effective_ledger,
    )

    # Generate audit notes via Gemini
    audit_notes = _gemini_audit_notes(violations, proposed_orders, sip_amount)

    approved  = len(violations) == 0
    new_retry = retry_count + (0 if approved else 1)

    if approved:
        print(f"  [Node 4] ✓ APPROVED — all risk checks passed")
    else:
        print(f"  [Node 4] ✗ REJECTED — {len(violations)} violation(s)")
        for v in violations:
            print(f"    • {v}")
        if new_retry >= MAX_RETRIES:
            print(f"  [Node 4] Max retries ({MAX_RETRIES}) reached → ABORT")
        else:
            print(f"  [Node 4] Sending back to Optimizer (attempt {new_retry}/{MAX_RETRIES})")

    # ── Value-Averaging (only on clean approval) ──────────────────────────────
    va_triggered  = False
    va_mult       = 1.0
    effective_sip = sip_amount

    if approved:
        va_mult, va_reason = _detect_va_condition(
            [o["ticker"] for o in proposed_orders],
            sentiment_scores, all_etf_data, boom_triggers,
        )
        if va_mult > 1.0:
            proposed_orders, effective_sip = _apply_va_multiplier(
                proposed_orders, sip_amount, va_mult,
            )
            va_triggered = True
            tier_label = "TIER 2 — Generational Crash" if va_mult >= VA_MULTIPLIER_TIER2 else "TIER 1 — Standard Dip"
            print(
                f"  [Node 4] ⚡ VALUE-AVERAGING {tier_label} — "
                f"SIP ${sip_amount:.2f} → ${effective_sip:.2f} "
                f"(×{va_mult:.2f})  |  {va_reason}"
            )
        else:
            va_mult = 1.0   # normalise in case _detect returned 1.0
            print(f"  [Node 4] Value-Averaging: not triggered "
                  f"(panic={bool(set(t.upper() for t in boom_triggers) & _PANIC_TRIGGER_TOKENS)}, "
                  f"momentum=positive or no data)")

    return {
        "risk_approved":     approved,
        "risk_violations":   violations,
        "risk_audit_notes":  audit_notes,
        "audit_retry_count": new_retry,
        # VA fields — always written so state is consistent
        "va_triggered":      va_triggered,
        "va_multiplier":     va_mult,
        # Overwrite proposed_orders + sip_amount when VA fired
        "proposed_orders":   proposed_orders,
        "sip_amount":        effective_sip,
    }
