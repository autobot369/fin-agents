"""
Sentiment-Weighted Dynamic SIP Allocator
=========================================
Implements the Core / Satellite allocation strategy:

  Core (default 70% of SIP):
    Top `core_count` ETFs from the MAS ranking.
    Each ETF's share within the core bucket is proportional to its
    consensus_score relative to all other core ETFs.

  Satellite (remaining 30% of SIP):
    ETFs ranked core_count+1 … top_n from the MAS ranking.
    Same score-weighted distribution within the satellite bucket.

Weight formula (per the Sentiment-Weighted Dynamic SIP spec):
    Weight_i = ConsensusScore_i / Σ ConsensusScore_bucket

This "doubles down" on boom signals: higher-scored ETFs get
disproportionately more capital, while all selected ETFs stay
in the portfolio (no selling).
"""

from typing import Any, Dict, List

from simulator.state import AllocationPlan, ETFAllocation


def compute_allocation(
    rankings: List[Dict[str, Any]],
    sip_amount: float = 500.0,
    top_n: int = 10,
    core_count: int = 5,
    core_pct: float = 0.70,
    min_alloc: float = 1.0,
) -> AllocationPlan:
    """
    Build a Core/Satellite Sentiment-Weighted Dynamic SIP allocation plan.

    Args:
        rankings:    Ranked ETF list from etf-selection-mas (rankings.json).
        sip_amount:  Total monthly SIP in USD.
        top_n:       How many ETFs to invest in (core + satellite combined).
        core_count:  How many top ETFs form the Core bucket.
        core_pct:    Fraction of SIP allocated to the Core bucket (0–1).
        min_alloc:   Minimum USD per ETF (prevents negligible positions).

    Returns:
        AllocationPlan with core, satellite, and combined lists.
    """
    selected = rankings[:top_n]
    if not selected:
        return AllocationPlan(
            sip_amount=sip_amount, core_budget=0, satellite_budget=0,
            core_pct=core_pct, core=[], satellite=[], all=[],
        )

    core_etfs      = selected[:core_count]
    satellite_etfs = selected[core_count:]

    core_budget      = round(sip_amount * core_pct, 2)
    satellite_budget = round(sip_amount - core_budget, 2)

    core_allocs      = _allocate_bucket(core_etfs,      core_budget,      "core",      min_alloc)
    satellite_allocs = _allocate_bucket(satellite_etfs, satellite_budget, "satellite", min_alloc)

    # Correct any rounding drift so buckets sum exactly to their budget
    _snap_to_budget(core_allocs,      core_budget)
    _snap_to_budget(satellite_allocs, satellite_budget)

    # Recompute global weight (fraction of total SIP)
    all_allocs = core_allocs + satellite_allocs
    for a in all_allocs:
        a["weight"] = round(a["monthly_usd"] / sip_amount, 5)

    return AllocationPlan(
        sip_amount=sip_amount,
        core_budget=core_budget,
        satellite_budget=satellite_budget,
        core_pct=core_pct,
        core=core_allocs,
        satellite=satellite_allocs,
        all=all_allocs,
    )


# ── helpers ──────────────────────────────────────────────────────────────────

def _allocate_bucket(
    etfs: List[Dict[str, Any]],
    budget: float,
    bucket: str,
    min_alloc: float,
) -> List[ETFAllocation]:
    """Distribute `budget` across `etfs` proportional to consensus_score."""
    if not etfs or budget <= 0:
        return []

    total_score = sum(e.get("consensus_score", 0.5) for e in etfs) or len(etfs) * 0.5

    allocs: List[ETFAllocation] = []
    for e in etfs:
        score      = e.get("consensus_score", 0.5)
        raw_weight = score / total_score
        monthly    = max(min_alloc, round(raw_weight * budget, 2))
        region     = e.get("region", "INTL")
        allocs.append(ETFAllocation(
            ticker            = e["ticker"],
            name              = e.get("name", e["ticker"]),
            region            = region,
            market            = e.get("market_label", "NSE" if region == "BSE" else "NASDAQ"),
            category          = e.get("category", "—"),
            trade_on          = e.get("trade_on", "—"),
            bucket            = bucket,
            consensus_score   = score,
            sentiment_score   = e.get("sentiment_score", 0.5),
            expense_score     = e.get("expense_score", 0.5),
            weight            = raw_weight,          # overwritten later to be global weight
            monthly_usd       = monthly,
            currency          = "INR" if region == "BSE" else "USD",
            expense_ratio     = e.get("expense_ratio"),
            momentum_3m       = e.get("momentum_3m"),
            sentiment_rationale = e.get("sentiment_rationale", ""),
        ))
    return allocs


def _snap_to_budget(allocs: List[ETFAllocation], budget: float) -> None:
    """Adjust the highest-scored ETF so the bucket sums exactly to budget."""
    if not allocs:
        return
    total = sum(a["monthly_usd"] for a in allocs)
    diff  = round(budget - total, 2)
    if diff != 0:
        # Apply adjustment to the top-scored ETF (index 0, already sorted by rank)
        allocs[0]["monthly_usd"] = round(allocs[0]["monthly_usd"] + diff, 2)
