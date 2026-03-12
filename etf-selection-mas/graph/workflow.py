"""
LangGraph Workflow — Global ETF Top-20 Boom List
=================================================

Graph topology (linear, all-market pipeline):

  MultiMarketResearcher      ← fetches INTL + HKCN + BSE simultaneously
         ↓
  ExpenseGuard               ← prunes TER > threshold across all markets
         ↓
  SentimentScorer            ← scores news, fires boom triggers, normalises
         ↓
  GlobalRankingNode          ← pure function: 60% sentiment + 40% expense score
         ↓                      → produces all_consensus_scores
  PortfolioArbiter           ← selects top-20, generates rationales, writes MD
         ↓
        END

GlobalRankingNode (pure function — no LLM, no I/O)
---------------------------------------------------
  expense_score(t) = max(0.0, 1.0 − TER(t) / ter_threshold)
      → 1.0 when TER = 0%    (no cost)
      → 0.0 when TER = threshold  (exactly at ceiling)
      → Unknown TER → 0.50  (neutral penalty)

  consensus_score(t) = SENTIMENT_W × sentiment_score(t)
                     + EXPENSE_W   × expense_score(t)
                     = 0.60 × s + 0.40 × e
"""

from typing import Any, Dict, List

from langgraph.graph import END, START, StateGraph

from agents.expense_guard import expense_guard_node
from agents.portfolio_arbiter import portfolio_arbiter_node
from agents.regional_researcher import multi_market_researcher_node
from agents.sentiment_scorer import sentiment_scorer_node
from graph.state import ETFSelectionState

SENTIMENT_WEIGHT = 0.60
EXPENSE_WEIGHT   = 0.40


# ---------------------------------------------------------------------------
# GlobalRankingNode — pure consensus scorer (all markets combined)
# ---------------------------------------------------------------------------

def global_ranking_node(state: ETFSelectionState) -> Dict[str, Any]:
    """
    Pure-function node: Global Ranking.

    Computes expense_score and consensus_score for every ticker that passed
    the ExpenseGuard, across all three market segments simultaneously.

    Reads  : state["all_sentiment_scores"], state["all_filtered_tickers"],
             state["all_etf_data"], state["ter_threshold"]
    Writes : state["all_expense_scores"], state["all_consensus_scores"]
    """
    sentiment_scores: Dict[str, float] = state.get("all_sentiment_scores", {})
    filtered_tickers: List[str]        = state.get("all_filtered_tickers", [])
    etf_data: Dict[str, Any]           = state.get("all_etf_data", {})
    ter_threshold: float               = state.get("ter_threshold", 0.007)

    print(f"\n{'='*60}")
    print(
        f"[GLOBAL_RANKING]  Scoring {len(filtered_tickers)} ETFs  "
        f"(weights: {SENTIMENT_WEIGHT*100:.0f}% sentiment + "
        f"{EXPENSE_WEIGHT*100:.0f}% expense)"
    )
    print(f"{'='*60}")

    expense_scores:   Dict[str, float] = {}
    consensus_scores: Dict[str, float] = {}

    for ticker in filtered_tickers:
        rec = etf_data.get(ticker, {})
        ter = rec.get("expense_ratio")

        # ── Expense score ─────────────────────────────────────────────────
        if ter is None:
            e_score = 0.50   # neutral — unknown TER
        else:
            e_score = max(0.0, round(1.0 - ter / ter_threshold, 4))

        expense_scores[ticker] = e_score

        # ── Sentiment score (default 0.5 if not scored) ───────────────────
        s_score = sentiment_scores.get(ticker, 0.50)

        # ── Consensus ─────────────────────────────────────────────────────
        c_score = round(SENTIMENT_WEIGHT * s_score + EXPENSE_WEIGHT * e_score, 4)
        consensus_scores[ticker] = c_score

        region = rec.get("region", "?")
        print(
            f"  [{region:<4}] {ticker:<22}  s={s_score:.4f}  e={e_score:.4f}  "
            f"→ consensus={c_score:.4f}"
        )

    top5 = sorted(consensus_scores.items(), key=lambda x: -x[1])[:5]
    print(f"\n  Preview top-5: {[f'{t}({s:.4f})' for t, s in top5]}")

    return {
        "all_expense_scores":   expense_scores,
        "all_consensus_scores": consensus_scores,
    }


# ---------------------------------------------------------------------------
# Build & compile the LangGraph StateGraph
# ---------------------------------------------------------------------------

def build_etf_graph() -> Any:
    """
    Assemble and compile the Global ETF Selection graph.

    Returns a compiled LangGraph object ready for `.invoke(initial_state)`.
    """
    g = StateGraph(ETFSelectionState)

    g.add_node("multi_market_researcher", multi_market_researcher_node)
    g.add_node("expense_guard",           expense_guard_node)
    g.add_node("sentiment_scorer",        sentiment_scorer_node)
    g.add_node("global_ranking_node",     global_ranking_node)
    g.add_node("portfolio_arbiter",       portfolio_arbiter_node)

    g.add_edge(START,                      "multi_market_researcher")
    g.add_edge("multi_market_researcher",  "expense_guard")
    g.add_edge("expense_guard",            "sentiment_scorer")
    g.add_edge("sentiment_scorer",         "global_ranking_node")
    g.add_edge("global_ranking_node",      "portfolio_arbiter")
    g.add_edge("portfolio_arbiter",        END)

    return g.compile()
