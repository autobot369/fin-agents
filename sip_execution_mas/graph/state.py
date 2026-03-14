"""
SIP Execution MAS — Shared LangGraph State
===========================================
All 6 nodes read/write from this single TypedDict.
"""
from __future__ import annotations

from typing import Any, Dict, List, Optional
from typing_extensions import TypedDict


# ── Per-node sub-structs ──────────────────────────────────────────────────────

class ETFRecord(TypedDict):
    """Populated by Node 1 (Regional Researcher)."""
    ticker: str
    name: str
    region: str           # "US" | "HKCN" | "BSE"
    market: str           # "NASDAQ" | "NYSE" | "NSE" | "HKEX"
    category: str         # e.g. "core" | "thematic" | "India Large-Cap" | rationale
    expense_ratio: Optional[float]   # decimal (0.0007 = 0.07%)
    aum_b: Optional[float]
    ytd_return: Optional[float]
    momentum_3m: Optional[float]
    momentum_1m: Optional[float]
    trailing_volatility_3m: Optional[float]  # annualised daily vol, decimal (e.g. 0.15 = 15%)
    forward_pe: Optional[float]              # forward P/E ratio (from yfinance)
    beta: Optional[float]                    # 1-year beta vs S&P 500
    dividend_yield: Optional[float]          # trailing annual dividend yield (decimal)
    current_price: Optional[float]
    currency: str         # "USD" | "INR" | "HKD"
    data_source: str      # "yfinance" | "gemini_estimate" | "fallback"
    fetch_error: Optional[str]
    # ── Cost & routing fields (added by Node 1) ───────────────────────
    recommended_broker: str          # "Dhan" | "Alpaca" | "Tiger" | "IBKR"
    adv_usd: Optional[float]         # average daily volume in USD equivalent
    liquidity_ok: bool               # ADV >= $1M USD equivalent
    est_entry_cost_pct: Optional[float]   # (brokerage_min/500) + stamp_duty + TER/12
    is_proxy: bool                   # True = US-listed proxy for an HKEX ticker
    proxy_for: Optional[str]         # HKEX ticker this proxies (e.g. "2800.HK")


class ProposedOrder(TypedDict):
    """One ETF position in the proposed allocation (Node 3 output)."""
    ticker: str
    name: str
    region: str
    market: str
    bucket: str           # "core" | "satellite"
    monthly_usd: float
    weight: float         # fraction of total SIP
    consensus_score: float
    sentiment_score: float
    expense_score: float
    sentiment_rationale: str
    currency: str


class ExecutionResult(TypedDict):
    """What actually happened at the broker (Node 5 output)."""
    ticker: str
    status: str           # "filled" | "dry_run" | "skipped" | "error"
    requested_usd: float
    filled_usd: float
    units: float
    price_native: float   # price in native currency (USD or INR)
    currency: str
    broker: str           # "paper" | "alpaca" | "dhan"
    order_id: Optional[str]
    error: Optional[str]


# ── Master state ──────────────────────────────────────────────────────────────

class SIPExecutionState(TypedDict):
    """
    Shared state passed through all 6 nodes.

    ┌─ Node 1: Regional Researcher ─────────────────────────────────┐
    │  Populates: all_etf_data, all_macro_news                      │
    └───────────────────────────────────────────────────────────────┘
    ┌─ Node 2: Signal Scorer (Gemini) ──────────────────────────────┐
    │  Populates: sentiment_scores, macro_summary, boom_triggers,   │
    │             scorer_source                                     │
    └───────────────────────────────────────────────────────────────┘
    ┌─ Node 3: Portfolio Optimizer (Gemini + allocator) ────────────┐
    │  Populates: expense_scores, consensus_scores,                 │
    │             allocation_plan, proposed_orders, optimizer_notes │
    └───────────────────────────────────────────────────────────────┘
    ┌─ Node 4: Risk Auditor (hard rules + Gemini) ──────────────────┐
    │  Populates: risk_approved, risk_violations,                   │
    │             risk_audit_notes, audit_retry_count               │
    │  Routes: approved → Node5 | retry → Node3 | abort → Node6    │
    └───────────────────────────────────────────────────────────────┘
    ┌─ Node 5: Broker Connector ────────────────────────────────────┐
    │  Populates: execution_results, execution_status, usd_inr_rate │
    └───────────────────────────────────────────────────────────────┘
    ┌─ Node 6: Execution Logger ────────────────────────────────────┐
    │  Populates: log_path, run_id (terminal state)                 │
    └───────────────────────────────────────────────────────────────┘
    """

    # ── Run config ────────────────────────────────────────────────
    run_id: str
    ter_threshold: float
    sip_amount: float
    top_n: int
    core_count: int
    core_pct: float
    max_position_pct: float   # hard limit: max fraction of SIP per ETF  (default 0.15)
    max_region_pct: float     # hard limit: max fraction of SIP per region (default 0.50)
    dry_run: bool             # True = paper mode, False = live execution
    force: bool               # True = bypass Rule 4 (allow re-investment same month)
    ledger_path: str          # path to portfolio_ledger.json

    # ── Node 1 outputs ────────────────────────────────────────────
    all_etf_data: Dict[str, ETFRecord]          # ticker → metrics
    all_macro_news: Dict[str, List[str]]        # region_query → headlines
    filtered_tickers: List[str]                 # tickers that passed TER filter

    # ── Node 2 outputs ────────────────────────────────────────────
    sentiment_scores: Dict[str, float]          # ticker → 0.0–1.0
    macro_summary: str
    boom_triggers: List[str]
    scorer_source: str                          # "gemini" | "vader"

    # ── Node 3 outputs ────────────────────────────────────────────
    expense_scores: Dict[str, float]            # ticker → 0.0–1.0
    consensus_scores: Dict[str, float]          # ticker → 0.0–1.0
    allocation_plan: Optional[Dict[str, Any]]   # AllocationPlan dict
    proposed_orders: List[ProposedOrder]
    optimizer_notes: str

    # ── Node 4 outputs ────────────────────────────────────────────
    risk_approved: bool
    risk_violations: List[str]
    risk_audit_notes: str
    audit_retry_count: int
    # Value-Averaging fields (set by Node 4 when panic+floor condition fires)
    va_triggered: bool    # True when the 20% top-up was applied this month
    va_multiplier: float  # Actual multiplier used (1.0 = no adjustment; 1.20 = top-up)

    # ── Node 5 outputs ────────────────────────────────────────────
    execution_results: List[ExecutionResult]
    execution_status: str     # "success" | "partial" | "dry_run" | "failed" | "aborted"
    usd_inr_rate: float

    # ── Node 6 outputs ────────────────────────────────────────────
    log_path: str
    run_id_out: str
