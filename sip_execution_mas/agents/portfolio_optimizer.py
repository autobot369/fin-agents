"""
Node 3 — Portfolio Optimizer (Gemini + allocator)
===================================================
Strategist node:

1. Computes expense scores + consensus scores from sentiment (Node 2)
2. Enforces region caps / position caps from previous audit violations
3. Calls simulator.allocator.compute_allocation() for exact USD amounts
4. Calls Gemini to generate one-sentence rationales per ETF
5. Builds the ProposedOrder list for the Risk Auditor

Formula:
  expense_score_i  = max(0, 1 - TER_i / ter_threshold)
  consensus_score_i = 0.60 × sentiment_score_i + 0.40 × expense_score_i
"""
from __future__ import annotations

import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from sip_execution_mas.graph.state import ProposedOrder, SIPExecutionState

# ── Resolve paths so we can import simulator.allocator ───────────────────────
_ROOT     = Path(__file__).resolve().parent.parent.parent   # fin-agents/
_SIP_ROOT = Path(__file__).resolve().parent.parent          # sip_execution_mas/
if str(_SIP_ROOT) not in sys.path:
    sys.path.insert(0, str(_SIP_ROOT))

from simulator.allocator import compute_allocation  # type: ignore


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_scores(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    sentiment_scores: Dict[str, float],
    ter_threshold: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    expense_scores: Dict[str, float] = {}
    consensus_scores: Dict[str, float] = {}

    for ticker in filtered_tickers:
        rec = all_etf_data.get(ticker, {})
        ter = rec.get("expense_ratio")
        exp_score = max(0.0, 1.0 - ter / ter_threshold) if ter else 0.50
        expense_scores[ticker] = round(min(1.0, exp_score), 4)

        sent = sentiment_scores.get(ticker, 0.50)
        consensus_scores[ticker] = round(0.60 * sent + 0.40 * expense_scores[ticker], 4)

    return expense_scores, consensus_scores


def _build_rankings_list(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    sentiment_scores: Dict[str, float],
    expense_scores: Dict[str, float],
    consensus_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """Build a rankings list compatible with simulator.allocator.compute_allocation()."""
    entries = []
    for ticker in filtered_tickers:
        rec = all_etf_data.get(ticker, {})
        entries.append({
            "rank":            0,         # placeholder; sorted below
            "ticker":          ticker,
            "name":            rec.get("name", ticker),
            "region":          rec.get("region", "INTL"),
            "market_label":    rec.get("market", "NASDAQ"),
            "category":        rec.get("category", "Unknown"),
            "trade_on":        rec.get("market", "NASDAQ"),
            "expense_ratio":   rec.get("expense_ratio"),
            "aum_b":           rec.get("aum_b"),
            "ytd_return":      rec.get("ytd_return"),
            "momentum_3m":     rec.get("momentum_3m"),
            "momentum_1m":     rec.get("momentum_1m"),
            "current_price":   rec.get("current_price"),
            "sentiment_score": sentiment_scores.get(ticker, 0.50),
            "expense_score":   expense_scores.get(ticker, 0.50),
            "consensus_score": consensus_scores.get(ticker, 0.50),
            "sentiment_rationale": "",  # filled by Gemini below
        })

    entries.sort(key=lambda x: (-x["consensus_score"], -(x["momentum_3m"] or 0)))
    for i, e in enumerate(entries):
        e["rank"] = i + 1
    return entries


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
            temperature=0.2,
        ),
    )


def _parse_json(text: str) -> dict:
    text = text.strip()
    if text.startswith("```"):
        parts = text.split("```")
        text = parts[1].lstrip("json").strip() if len(parts) > 1 else text
    return json.loads(text)


_RATIONALE_PROMPT = """You are a portfolio strategist. Write a ONE-sentence investment rationale
for each ETF in the selected portfolio. Focus on macro tailwinds, momentum, or value.

Return ONLY valid JSON:
{
  "rationales": {
    "TICKER1": "One sentence rationale.",
    "TICKER2": "One sentence rationale."
  }
}

ETF DATA:
"""


def _fetch_rationales(
    portfolio_etfs: List[Dict[str, Any]],
) -> Dict[str, str]:
    """Call Gemini to generate rationales for the selected ETFs."""
    if not portfolio_etfs:
        return {}

    lines = []
    for etf in portfolio_etfs:
        ter = etf.get("expense_ratio")
        ter_str = f"{ter*100:.2f}%" if ter else "N/A"
        mom3 = etf.get("momentum_3m")
        mom3_str = f"{mom3:+.1f}%" if mom3 is not None else "N/A"
        lines.append(
            f"{etf['ticker']}: {etf['name']} | region={etf['region']} | "
            f"consensus={etf['consensus_score']:.3f} | TER={ter_str} | 3m={mom3_str}"
        )

    prompt = _RATIONALE_PROMPT + "\n".join(lines)

    try:
        model = _get_gemini_model()
        response = model.generate_content(prompt)
        parsed = _parse_json(response.text)
        return {k: v for k, v in parsed.get("rationales", {}).items()}
    except Exception as exc:
        print(f"  [Node 3] ⚠  Gemini rationale error: {exc} — using template rationales")
        # Template fallback
        result = {}
        for etf in portfolio_etfs:
            region = etf.get("region", "global")
            score = etf.get("consensus_score", 0.5)
            sentiment = "bullish" if score > 0.65 else ("bearish" if score < 0.40 else "neutral")
            result[etf["ticker"]] = (
                f"{etf['name']} shows {sentiment} signals with consensus score "
                f"{score:.2f} in the {region} region."
            )
        return result


def _apply_violation_caps(
    plan: Dict[str, Any],
    violations: List[str],
    max_position_pct: float,
    max_region_pct: float,
    sip_amount: float,
) -> Dict[str, Any]:
    """
    Post-process the allocation plan to enforce caps that were flagged
    by the Risk Auditor in a previous attempt.
    """
    all_positions = plan.get("all", [])
    if not all_positions:
        return plan

    # Enforce per-position cap
    position_cap_usd = sip_amount * max_position_pct
    for pos in all_positions:
        if pos["monthly_usd"] > position_cap_usd:
            pos["monthly_usd"] = round(position_cap_usd, 2)

    # Enforce per-region cap
    region_totals: Dict[str, float] = {}
    for pos in all_positions:
        r = pos.get("region", "INTL")
        region_totals[r] = region_totals.get(r, 0) + pos["monthly_usd"]

    region_cap_usd = sip_amount * max_region_pct
    for region, total in region_totals.items():
        if total > region_cap_usd:
            scale = region_cap_usd / total
            for pos in all_positions:
                if pos.get("region") == region:
                    pos["monthly_usd"] = round(pos["monthly_usd"] * scale, 2)

    # Recompute weights
    total_invested = sum(p["monthly_usd"] for p in all_positions)
    for pos in all_positions:
        pos["weight"] = round(pos["monthly_usd"] / sip_amount, 4) if sip_amount else 0

    plan["all"] = all_positions
    return plan


# ── Node function ─────────────────────────────────────────────────────────────

def portfolio_optimizer_node(state: SIPExecutionState) -> dict:
    """
    Node 3 — Portfolio Optimizer

    Computes scores, allocates SIP, generates Gemini rationales,
    and builds the ProposedOrder list.
    If called after a Risk Auditor rejection, applies hard caps.
    """
    filtered_tickers  = state["filtered_tickers"]
    all_etf_data      = state["all_etf_data"]
    sentiment_scores  = state["sentiment_scores"]
    ter_threshold     = state["ter_threshold"]
    sip_amount        = state["sip_amount"]
    top_n             = state["top_n"]
    core_count        = state["core_count"]
    core_pct          = state["core_pct"]
    max_position_pct  = state["max_position_pct"]
    max_region_pct    = state["max_region_pct"]
    audit_retry       = state.get("audit_retry_count", 0)
    violations        = state.get("risk_violations", [])

    retry_msg = f" [retry #{audit_retry}]" if audit_retry > 0 else ""
    print(f"\n[Node 3] Portfolio Optimizer — building allocation{retry_msg} …")

    # 1. Compute scores
    expense_scores, consensus_scores = _compute_scores(
        filtered_tickers, all_etf_data, sentiment_scores, ter_threshold
    )

    # 2. Build ranked list for allocator
    rankings = _build_rankings_list(
        filtered_tickers, all_etf_data,
        sentiment_scores, expense_scores, consensus_scores,
    )

    # 3. Compute allocation via shared allocator
    top_n_actual   = min(top_n, len(rankings))
    core_actual    = min(core_count, top_n_actual - 1)
    plan = compute_allocation(
        rankings,
        sip_amount  = sip_amount,
        top_n       = top_n_actual,
        core_count  = core_actual,
        core_pct    = core_pct,
    )

    # 4. If this is a retry, apply hard caps to fix previous violations
    if audit_retry > 0 and violations:
        print(f"  [Node 3] Applying hard caps to fix {len(violations)} violation(s) …")
        plan = _apply_violation_caps(
            plan, violations, max_position_pct, max_region_pct, sip_amount
        )

    # 5. Fetch Gemini rationales for selected ETFs
    selected_etfs = [
        e for e in rankings
        if e["ticker"] in {p["ticker"] for p in plan["all"]}
    ]
    rationales = _fetch_rationales(selected_etfs)

    # 6. Build ProposedOrder list
    proposed_orders: List[ProposedOrder] = []
    for alloc in plan["all"]:
        ticker = alloc["ticker"]
        rec = all_etf_data.get(ticker, {})
        order: ProposedOrder = {
            "ticker":              ticker,
            "name":                alloc.get("name", ticker),
            "region":              alloc.get("region", rec.get("region", "INTL")),
            "market":              alloc.get("market", rec.get("market", "NASDAQ")),
            "bucket":              alloc.get("bucket", "core"),
            "monthly_usd":         alloc["monthly_usd"],
            "weight":              alloc["weight"],
            "consensus_score":     alloc["consensus_score"],
            "sentiment_score":     alloc["sentiment_score"],
            "expense_score":       alloc["expense_score"],
            "sentiment_rationale": rationales.get(ticker, ""),
            "currency":            rec.get("currency", "USD"),
        }
        proposed_orders.append(order)

    print(f"  [Node 3] Proposed {len(proposed_orders)} orders")
    print(f"  [Node 3] Core {core_actual} (${plan['core_budget']:.2f}) + "
          f"Satellite {top_n_actual - core_actual} (${plan['satellite_budget']:.2f})")

    # Build notes for auditor
    region_totals: Dict[str, float] = {}
    for o in proposed_orders:
        region_totals[o["region"]] = region_totals.get(o["region"], 0) + o["monthly_usd"]
    region_breakdown = ", ".join(f"{r} ${v:.0f}" for r, v in sorted(region_totals.items()))
    optimizer_notes = (
        f"Top-{top_n_actual} ETFs | Core {core_actual} × {core_pct*100:.0f}% | "
        f"Satellite {top_n_actual - core_actual} × {(1-core_pct)*100:.0f}% | "
        f"Region: {region_breakdown}"
    )

    return {
        "expense_scores":   expense_scores,
        "consensus_scores": consensus_scores,
        "allocation_plan":  plan,
        "proposed_orders":  proposed_orders,
        "optimizer_notes":  optimizer_notes,
    }
