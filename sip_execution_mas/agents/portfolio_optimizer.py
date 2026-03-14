"""
Node 3 — Portfolio Optimizer (Gemini + allocator)
===================================================
Strategist node:

1. Computes expense scores + consensus scores from sentiment (Node 2)
2. Applies sticky-holdings policy (see EVICTION_THRESHOLD below)
3. Enforces region caps / position caps from previous audit violations
4. Calls simulator.allocator.compute_allocation() for exact USD amounts
5. Calls Gemini to generate one-sentence rationales per ETF
6. Builds the ProposedOrder list for the Risk Auditor

Formula (pseudo-Sharpe rank):
  expense_score_i   = max(0, 1 - TER_i / ter_threshold)
  raw_i             = 0.60 × sentiment_score_i + 0.40 × expense_score_i
  consensus_score_i = (raw_i / volatility_i) / max_j(raw_j / volatility_j)

  volatility_i = trailing_volatility_3m from Node 1 (annualised decimal),
                 floored at 0.05 to prevent blow-up for low-vol ETFs.
  Division by max_j normalises the batch to [0, 1] so EVICTION_THRESHOLD
  retains its calibration across months.

Sticky-holdings policy:
  A currently-held ETF keeps its slot in the portfolio unless its
  consensus_score drops below EVICTION_THRESHOLD.  This prevents the
  algorithm from cycling ETFs purely because a new ticker scored a few
  hundredths of a point higher this month.
  New tickers can only enter when a slot is genuinely free (i.e. a held
  ticker was evicted or the portfolio still has room).
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


# ── Constants ─────────────────────────────────────────────────────────────────

# Held ETFs are evicted only when their consensus_score falls below this.
# Above this floor they always keep their slot regardless of new-entrant scores.
EVICTION_THRESHOLD = 0.30

# Minimum annualised volatility used as the divisor (prevents near-zero blow-up
# for money-market / ultra-low-vol ETFs such as LIQUIDBEES.NS).
_VOL_FLOOR = 0.05   # 5% annualised

# ── Locked production universe — 12 ETFs in two permanent buckets ─────────────
# When filtered_tickers ⊆ _LOCKED_TICKERS the optimizer uses a fixed core/sat
# bucket split instead of sticky-select + rank-based compute_allocation.
_CORE_TICKERS      = frozenset([
    "USCA", "SPDW", "SPEM", "FLIN", "NIFTYBEES.NS",
])
_SATELLITE_TICKERS = frozenset([
    "SOXQ", "XLK", "AVUV", "URNM", "CIBR", "SMIN", "XBI",
])
_LOCKED_TICKERS = _CORE_TICKERS | _SATELLITE_TICKERS
_BUCKET_SPLIT   = 0.70   # core fraction of total SIP


# ── Helpers ───────────────────────────────────────────────────────────────────

def _compute_scores(
    filtered_tickers: List[str],
    all_etf_data: Dict[str, Any],
    sentiment_scores: Dict[str, float],
    ter_threshold: float,
) -> tuple[Dict[str, float], Dict[str, float]]:
    """
    Compute expense and consensus scores.

    consensus_score = ((0.60 × sentiment) + (0.40 × expense)) / volatility

    This is a pseudo-Sharpe rank: the numerator rewards high sentiment + low cost;
    the denominator penalises high trailing volatility.  Raw scores are normalised
    to [0, 1] (divided by the batch maximum) so that EVICTION_THRESHOLD = 0.30
    retains its meaning across months.

    Volatility source: trailing_volatility_3m from Node 1 (annualised decimal,
    e.g. 0.15 = 15%).  Falls back to _VOL_FLOOR = 0.05 when unavailable.
    """
    expense_scores: Dict[str, float] = {}
    raw_consensus:  Dict[str, float] = {}

    for ticker in filtered_tickers:
        rec = all_etf_data.get(ticker, {})

        # Expense score
        ter = rec.get("expense_ratio")
        exp_score = max(0.0, 1.0 - ter / ter_threshold) if ter else 0.50
        expense_scores[ticker] = round(min(1.0, exp_score), 4)

        # Pseudo-Sharpe numerator
        sent = sentiment_scores.get(ticker, 0.50)
        numerator = 0.60 * sent + 0.40 * expense_scores[ticker]

        # Volatility divisor (with floor to avoid blow-up for low-vol ETFs)
        vol = rec.get("trailing_volatility_3m")
        vol_dec = max(_VOL_FLOOR, float(vol)) if vol else _VOL_FLOOR

        raw_consensus[ticker] = numerator / vol_dec

    # Normalise to [0, 1] so the eviction threshold remains calibrated
    max_raw = max(raw_consensus.values()) if raw_consensus else 1.0
    if max_raw <= 0:
        max_raw = 1.0
    consensus_scores: Dict[str, float] = {
        t: round(v / max_raw, 4) for t, v in raw_consensus.items()
    }

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
        model_name="gemini-2.5-flash",
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


def _load_held_tickers(ledger_path: str) -> set:
    """
    Return the set of tickers that appear in any ledger entry (i.e. currently held).
    Returns an empty set if the ledger is missing or unreadable.
    """
    import json
    from pathlib import Path
    p = Path(ledger_path)
    if not p.exists():
        return set()
    try:
        with open(p, encoding="utf-8") as f:
            ledger = json.load(f)
    except Exception:
        return set()
    held: set = set()
    for entry in ledger.get("entries", []):
        for pos in entry.get("positions", []):
            held.add(pos["ticker"])
    return held


def _sticky_select(
    filtered_tickers: List[str],
    consensus_scores: Dict[str, float],
    held_tickers: set,
    top_n: int,
) -> tuple:
    """
    Select up to top_n tickers applying the sticky-holdings policy.

    Returns (selected, protected, evicted):
      selected  – ordered list of tickers to pass to the allocator
      protected – held tickers that scored >= EVICTION_THRESHOLD (kept)
      evicted   – held tickers that scored <  EVICTION_THRESHOLD (dropped)

    Algorithm:
      1. Protected = held ∩ filtered  where score >= EVICTION_THRESHOLD
         These always occupy a slot (up to top_n total).
      2. Candidates = everything else in filtered, sorted by score desc.
         They fill the remaining (top_n - len(protected)) slots.
      3. Both groups are merged and re-sorted by score for the allocator
         (so the core/satellite split stays score-driven).
    """
    protected = [
        t for t in filtered_tickers
        if t in held_tickers and consensus_scores.get(t, 0.0) >= EVICTION_THRESHOLD
    ]
    evicted = [
        t for t in filtered_tickers
        if t in held_tickers and consensus_scores.get(t, 0.0) < EVICTION_THRESHOLD
    ]
    candidates = [
        t for t in filtered_tickers
        if t not in held_tickers or consensus_scores.get(t, 0.0) < EVICTION_THRESHOLD
    ]
    candidates.sort(key=lambda t: -consensus_scores.get(t, 0.0))

    remaining = max(0, top_n - len(protected))
    selected = protected + candidates[:remaining]
    # Re-sort by score so core/satellite split is score-driven
    selected.sort(key=lambda t: -consensus_scores.get(t, 0.0))
    return selected, protected, evicted


def _allocate_fixed_bucket(
    tickers: List[str],
    budget: float,
    bucket: str,
    all_etf_data: Dict[str, Any],
    sentiment_scores: Dict[str, float],
    expense_scores: Dict[str, float],
    consensus_scores: Dict[str, float],
) -> List[Dict[str, Any]]:
    """
    Allocate budget across tickers proportionally to consensus_score.
    All tickers always get a position (no rank-based culling).
    Returns allocation dicts compatible with plan["all"].
    """
    total_score = sum(max(consensus_scores.get(t, 0.01), 0.01) for t in tickers)
    allocs = []
    for ticker in tickers:
        rec   = all_etf_data.get(ticker, {})
        score = max(consensus_scores.get(ticker, 0.01), 0.01)
        region = rec.get("region", "US")
        allocs.append({
            "ticker":          ticker,
            "name":            rec.get("name", ticker),
            "region":          region,
            "market":          rec.get("market", "NASDAQ"),
            "category":        rec.get("category", "Unknown"),
            "trade_on":        rec.get("market", "NASDAQ"),
            "currency":        "INR" if region == "BSE" else "USD",
            "bucket":          bucket,
            "monthly_usd":     round(budget * score / total_score, 2),
            "weight":          0.0,   # recomputed by caller after combining buckets
            "consensus_score": consensus_scores.get(ticker, 0.50),
            "sentiment_score": sentiment_scores.get(ticker, 0.50),
            "expense_score":   expense_scores.get(ticker, 0.50),
        })
    return allocs


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
            region_positions = [p for p in all_positions if p.get("region") == region]
            for pos in region_positions:
                pos["monthly_usd"] = round(pos["monthly_usd"] * scale, 2)
            # Snap rounding drift so the region total never exceeds the cap
            drift = round(sum(p["monthly_usd"] for p in region_positions) - region_cap_usd, 2)
            if drift > 0 and region_positions:
                largest = max(region_positions, key=lambda p: p["monthly_usd"])
                largest["monthly_usd"] = round(largest["monthly_usd"] - drift, 2)

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

    Locked-universe path (production):
      filtered_tickers ⊆ _LOCKED_TICKERS → fixed 70/30 bucket split.
      Core (70%): VTI, SPLG, SPDW, SPEM, FLIN, NIFTYBEES.NS, QUAL — always all 7.
      Satellite (30%): XLK, QQQM, SOXQ, ICLN, USCA, ESGV, XLY — always all 7,
        sentiment-weighted so macro cycle rotates capital between them.

    Legacy path (backtest):
      Applies sticky-holdings + compute_allocation as before.

    Both paths: apply hard caps on retry, then fetch Gemini rationales.
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

    # 1. Compute scores (pseudo-Sharpe, same for both paths)
    expense_scores, consensus_scores = _compute_scores(
        filtered_tickers, all_etf_data, sentiment_scores, ter_threshold
    )

    # ── Detect which allocation path to use ───────────────────────────────────
    is_locked = set(filtered_tickers) <= _LOCKED_TICKERS

    if is_locked:
        # ── Locked-universe: fixed 70 / 30 bucket split ───────────────────────
        core_tickers      = [t for t in filtered_tickers if t in _CORE_TICKERS]
        satellite_tickers = [t for t in filtered_tickers if t in _SATELLITE_TICKERS]
        core_budget       = round(sip_amount * _BUCKET_SPLIT, 2)
        satellite_budget  = round(sip_amount * (1.0 - _BUCKET_SPLIT), 2)

        core_allocs      = _allocate_fixed_bucket(
            core_tickers, core_budget, "core",
            all_etf_data, sentiment_scores, expense_scores, consensus_scores,
        )
        satellite_allocs = _allocate_fixed_bucket(
            satellite_tickers, satellite_budget, "satellite",
            all_etf_data, sentiment_scores, expense_scores, consensus_scores,
        )

        all_allocs = core_allocs + satellite_allocs
        for a in all_allocs:
            a["weight"] = round(a["monthly_usd"] / sip_amount, 5) if sip_amount else 0

        plan = {
            "sip_amount":       sip_amount,
            "core_pct":         _BUCKET_SPLIT,
            "all":              all_allocs,
            "core":             core_allocs,
            "satellite":        satellite_allocs,
            "core_budget":      core_budget,
            "satellite_budget": satellite_budget,
        }
        top_n_actual = len(all_allocs)
        core_actual  = len(core_allocs)
        print(f"  [Node 3] Locked 70/30 — core {len(core_tickers)} (${core_budget:.2f}) + "
              f"satellite {len(satellite_tickers)} (${satellite_budget:.2f})")

    else:
        # ── Legacy path (backtest): sticky-select + compute_allocation ─────────
        held_tickers = _load_held_tickers(state["ledger_path"])
        if held_tickers:
            selected_tickers, protected, evicted = _sticky_select(
                filtered_tickers, consensus_scores, held_tickers, top_n
            )
            if protected:
                print(f"  [Node 3] Sticky: {len(protected)} held ETF(s) protected "
                      f"(score ≥ {EVICTION_THRESHOLD}) → {sorted(protected)}")
            if evicted:
                print(f"  [Node 3] Sticky: {len(evicted)} held ETF(s) evicted "
                      f"(score < {EVICTION_THRESHOLD}) → {sorted(evicted)}")
            new_entries = [t for t in selected_tickers if t not in held_tickers]
            if new_entries:
                print(f"  [Node 3] Sticky: {len(new_entries)} new ETF(s) filling open slot(s) "
                      f"→ {sorted(new_entries)}")
        else:
            selected_tickers = filtered_tickers

        rankings = _build_rankings_list(
            selected_tickers, all_etf_data,
            sentiment_scores, expense_scores, consensus_scores,
        )
        top_n_actual = min(top_n, len(rankings))
        core_actual  = min(core_count, top_n_actual - 1)
        plan = compute_allocation(
            rankings,
            sip_amount  = sip_amount,
            top_n       = top_n_actual,
            core_count  = core_actual,
            core_pct    = core_pct,
        )

    # 2. If retry, apply hard caps to fix previous violations
    if audit_retry > 0 and violations:
        print(f"  [Node 3] Applying hard caps to fix {len(violations)} violation(s) …")
        plan = _apply_violation_caps(
            plan, violations, max_position_pct, max_region_pct, sip_amount
        )

    # 3. Fetch Gemini rationales
    if is_locked:
        selected_etfs = [
            {
                "ticker":          a["ticker"],
                "name":            a.get("name", a["ticker"]),
                "region":          a.get("region", "US"),
                "consensus_score": a["consensus_score"],
                "expense_ratio":   all_etf_data.get(a["ticker"], {}).get("expense_ratio"),
                "momentum_3m":     all_etf_data.get(a["ticker"], {}).get("momentum_3m"),
            }
            for a in plan["all"]
        ]
    else:
        plan_tickers  = {p["ticker"] for p in plan["all"]}
        selected_etfs = [e for e in rankings if e["ticker"] in plan_tickers]

    rationales = _fetch_rationales(selected_etfs)

    # 4. Build ProposedOrder list
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

    region_totals: Dict[str, float] = {}
    for o in proposed_orders:
        region_totals[o["region"]] = region_totals.get(o["region"], 0) + o["monthly_usd"]
    region_breakdown = ", ".join(f"{r} ${v:.0f}" for r, v in sorted(region_totals.items()))

    if is_locked:
        optimizer_notes = (
            f"Locked universe | Core {len(_CORE_TICKERS)} × {_BUCKET_SPLIT*100:.0f}% | "
            f"Satellite {len(_SATELLITE_TICKERS)} × {(1-_BUCKET_SPLIT)*100:.0f}% | "
            f"Region: {region_breakdown}"
        )
    else:
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
