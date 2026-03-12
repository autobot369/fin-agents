"""
ETF Selection MAS — Global State Schema (v2)
--------------------------------------------
Redesigned for a single cross-market pipeline that processes INTL + HKCN + BSE
simultaneously and produces one global Top-20 Boom List.

Key differences from v1:
  • `market` field removed — the pipeline always runs all three segments.
  • All per-ticker data lives in `all_etf_data`, `all_filtered_tickers`, etc.
  • `sentiment_rationales` added — LLM-generated one-sentence boom rationale
    per ETF, produced by the PortfolioArbiter after the top-20 is selected.
  • boom_triggers_fired uses operator.add so each market's triggers accumulate.

All monetary values are in USD. Expense ratios are decimal fractions (0.0007 = 0.07%).
"""

import operator
from typing import Annotated, Any, Dict, List, Optional, TypedDict


class ETFRecord(TypedDict):
    """Per-ETF quantitative data populated by the MultiMarketResearcher."""
    ticker: str
    name: str
    market: str                       # "NASDAQ" | "NSE"
    region: str                       # "INTL" | "HKCN" | "BSE"
    expense_ratio: Optional[float]    # decimal, e.g. 0.0007 = 0.07%
    aum_b: Optional[float]            # AUM in USD billions
    ytd_return: Optional[float]       # YTD return as decimal
    momentum_3m: Optional[float]      # 3-month price return
    momentum_1m: Optional[float]      # 1-month price return
    current_price: Optional[float]
    volume_avg_30d: Optional[int]
    data_source: str                  # "yfinance" | "fallback"
    fetch_error: Optional[str]


class ETFSelectionState(TypedDict):
    # ── Pipeline config ───────────────────────────────────────────────────
    ter_threshold: float               # TER ceiling (decimal), default 0.007 = 0.70%
    top_n: int                         # Final list size, default 20

    # ── Populated by MultiMarketResearcher ────────────────────────────────
    all_etf_data: Dict[str, ETFRecord]          # ticker → ETFRecord (all markets)
    all_macro_news: Dict[str, List[Dict]]        # query → articles (all markets)
    researcher_notes: str                         # LLM macro brief (all markets)

    # ── Populated by ExpenseGuard ─────────────────────────────────────────
    all_filtered_tickers: List[str]              # Passed TER filter (all markets)
    all_pruned_tickers: List[str]                # Failed TER filter (all markets)

    # ── Populated by SentimentScorer ─────────────────────────────────────
    boom_triggers_fired: Annotated[List[str], operator.add]
    raw_article_scores: Dict[str, List[Dict]]    # query → scored article list
    all_sentiment_scores: Dict[str, float]       # ticker → score [0.0, 1.0]
    sentiment_narrative: str

    # ── Populated by GlobalRankingNode (ConsensusNode) ────────────────────
    all_expense_scores: Dict[str, float]         # ticker → expense score [0.0, 1.0]
    all_consensus_scores: Dict[str, float]       # ticker → 0.60*sentiment + 0.40*expense

    # ── Populated by PortfolioArbiter ─────────────────────────────────────
    sentiment_rationales: Dict[str, str]         # ticker → one-sentence boom rationale
    final_ranking: List[Dict[str, Any]]          # Top-20 sorted list
    output_markdown: str                          # Written to TOP_20_BOOM_REPORT.md
