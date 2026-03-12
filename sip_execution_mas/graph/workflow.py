"""
LangGraph Workflow — 6-Node SIP Execution Loop
================================================

Topology:

  researcher → scorer → optimizer → auditor
                                       │
                          ┌────────────┼────────────┐
                       approved      retry        abort
                          │            │             │
                        broker      optimizer     logger
                          │                         │
                        logger ──────────────────────┘

The audit loop retries at most MAX_RETRIES=2 times before aborting.
"""
from __future__ import annotations

from langgraph.graph import StateGraph, END

from sip_execution_mas.graph.state import SIPExecutionState
from sip_execution_mas.agents.regional_researcher import regional_researcher_node
from sip_execution_mas.agents.signal_scorer import signal_scorer_node
from sip_execution_mas.agents.portfolio_optimizer import portfolio_optimizer_node
from sip_execution_mas.agents.risk_auditor import risk_auditor_node, MAX_RETRIES
from sip_execution_mas.agents.broker_connector import broker_connector_node
from sip_execution_mas.agents.execution_logger import execution_logger_node


# ── Conditional router ────────────────────────────────────────────────────────

def _route_after_audit(state: SIPExecutionState) -> str:
    """
    After Risk Auditor runs:
    - Approved                     → broker
    - Rejected + retries left      → optimizer  (with violations in state)
    - Rejected + max retries hit   → logger     (abort path)
    """
    if state["risk_approved"]:
        return "broker"
    if state["audit_retry_count"] < MAX_RETRIES:
        return "optimizer"
    # Set abort status before going to logger
    return "logger"


# ── Graph builder ─────────────────────────────────────────────────────────────

def build_graph() -> StateGraph:
    graph = StateGraph(SIPExecutionState)

    # Register nodes
    graph.add_node("researcher", regional_researcher_node)
    graph.add_node("scorer",     signal_scorer_node)
    graph.add_node("optimizer",  portfolio_optimizer_node)
    graph.add_node("auditor",    risk_auditor_node)
    graph.add_node("broker",     broker_connector_node)
    graph.add_node("logger",     execution_logger_node)

    # Linear flow: researcher → scorer → optimizer → auditor
    graph.set_entry_point("researcher")
    graph.add_edge("researcher", "scorer")
    graph.add_edge("scorer",     "optimizer")
    graph.add_edge("optimizer",  "auditor")

    # Conditional audit routing
    graph.add_conditional_edges(
        "auditor",
        _route_after_audit,
        {
            "broker":    "broker",
            "optimizer": "optimizer",   # loop back on retry
            "logger":    "logger",      # abort path
        },
    )

    # After broker → logger → END
    graph.add_edge("broker", "logger")
    graph.add_edge("logger", END)

    return graph.compile()


# ── Convenience runner ────────────────────────────────────────────────────────

def run_sip_execution(
    sip_amount:       float = 500.0,
    top_n:            int   = 10,
    core_count:       int   = 5,
    core_pct:         float = 0.70,
    ter_threshold:    float = 0.007,
    max_position_pct: float = 0.15,
    max_region_pct:   float = 0.50,
    dry_run:          bool  = True,
    force:            bool  = False,
    ledger_path:      str   = "",
    run_id:           str   = "",
) -> dict:
    """
    Build and invoke the 6-node graph with the given parameters.
    Returns the final state dict.
    """
    import uuid
    from pathlib import Path

    if not run_id:
        run_id = str(uuid.uuid4())[:8].upper()

    if not ledger_path:
        _sip_root = Path(__file__).resolve().parent.parent   # sip_execution_mas/
        ledger_path = str(_sip_root / "simulator" / "outputs" / "portfolio_ledger.json")

    initial_state: SIPExecutionState = {
        # Config
        "run_id":           run_id,
        "ter_threshold":    ter_threshold,
        "sip_amount":       sip_amount,
        "top_n":            top_n,
        "core_count":       core_count,
        "core_pct":         core_pct,
        "max_position_pct": max_position_pct,
        "max_region_pct":   max_region_pct,
        "dry_run":          dry_run,
        "force":            force,
        "ledger_path":      ledger_path,
        # Node outputs (initialised empty)
        "all_etf_data":      {},
        "all_macro_news":    {},
        "filtered_tickers":  [],
        "sentiment_scores":  {},
        "macro_summary":     "",
        "boom_triggers":     [],
        "expense_scores":    {},
        "consensus_scores":  {},
        "allocation_plan":   None,
        "proposed_orders":   [],
        "optimizer_notes":   "",
        "risk_approved":     False,
        "risk_violations":   [],
        "risk_audit_notes":  "",
        "audit_retry_count": 0,
        "execution_results": [],
        "execution_status":  "pending",
        "usd_inr_rate":      84.0,
        "log_path":          "",
        "run_id_out":        "",
    }

    app = build_graph()
    final_state = app.invoke(initial_state)
    return final_state
