"""
Simulator State Types
=====================
TypedDicts for the SIP Simulator data pipeline.
"""

from typing import Any, Dict, List, Optional, TypedDict


class ETFAllocation(TypedDict):
    """Single ETF's monthly SIP allocation."""
    ticker: str
    name: str
    region: str           # "INTL" | "HKCN" | "BSE"
    market: str           # "NASDAQ" | "NSE"
    category: str
    trade_on: str
    bucket: str           # "core" | "satellite"
    consensus_score: float
    sentiment_score: float
    expense_score: float
    weight: float         # fraction of total SIP (0–1)
    monthly_usd: float    # USD amount invested per month
    currency: str         # "USD" or "INR"
    expense_ratio: Optional[float]
    momentum_3m: Optional[float]
    sentiment_rationale: str


class AllocationPlan(TypedDict):
    """Full Core + Satellite allocation plan for one SIP cycle."""
    sip_amount: float
    core_budget: float          # e.g. 350 (70%)
    satellite_budget: float     # e.g. 150 (30%)
    core_pct: float             # e.g. 0.70
    core: List[ETFAllocation]
    satellite: List[ETFAllocation]
    all: List[ETFAllocation]    # core + satellite, ordered by rank


class MonthlySnapshot(TypedDict):
    """State of one SIP installment."""
    date: str                        # "YYYY-MM-DD" (first trading day of month)
    units_bought: Dict[str, float]   # ticker → units purchased this month
    price_paid: Dict[str, float]     # ticker → native-currency price
    usd_invested: float              # total USD invested this month


class ETFHolding(TypedDict):
    """Current holding position for one ETF."""
    ticker: str
    bucket: str
    region: str
    total_units: float
    current_price_native: float   # price in native currency (USD or INR)
    value_usd: float
    invested_usd: float
    pnl_usd: float
    return_pct: float


class SimulationResult(TypedDict):
    """Full result of the historical SIP backtest."""
    total_invested_usd: float
    current_value_usd: float
    total_pnl_usd: float
    total_return_pct: float
    cagr: float
    months_simulated: int
    usd_inr_rate: float
    monthly_snapshots: List[MonthlySnapshot]
    holdings: Dict[str, ETFHolding]   # ticker → holding
    skipped_tickers: List[str]        # tickers with no price data


# ── Month-by-month backtest types ─────────────────────────────────────────────

class BacktestPosition(TypedDict):
    """One ETF position within a single backtest month."""
    ticker: str
    name: str
    bucket: str           # "core" | "satellite"
    region: str           # "BSE" | "US" | "HKCN"
    category: str
    consensus_score: float
    sentiment_score: float
    expense_score: float
    weight: float         # fraction of total SIP
    monthly_usd: float    # USD allocated this month
    price_native: float   # buy price in native currency (USD or INR)
    units_bought: float
    currency: str         # "USD" | "INR"


class BacktestMonthEntry(TypedDict):
    """Research + investment record for a single calendar month."""
    month: str                     # "YYYY-MM"
    date: str                      # "YYYY-MM-DD" (first trading day used as buy date)
    sip_amount: float
    core_budget: float
    satellite_budget: float
    total_invested_usd: float
    positions: List[BacktestPosition]
    boom_triggers: List[str]
    macro_summary: str
    scorer: str                    # "gemini" | "vader"
    usd_inr_rate: float


class HistoricalBacktestResult(TypedDict):
    """Full result of the month-by-month historical backtest."""
    start_date: str                        # "YYYY-MM-DD"
    end_date: str                          # "YYYY-MM-DD"
    months_run: int
    sip_amount: float
    total_invested_usd: float
    current_value_usd: float
    total_pnl_usd: float
    total_return_pct: float
    cagr: float
    usd_inr_rate: float
    monthly_entries: List[BacktestMonthEntry]
    holdings: Dict[str, Any]              # ticker → holding info dict
    skipped_months: List[str]             # months where research failed entirely
