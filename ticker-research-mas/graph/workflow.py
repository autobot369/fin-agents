"""
LangGraph StateGraph Assembly
------------------------------
Debate flow (linear — no conditional branching needed for a fixed 2-round debate):

  researcher
      │
  bull_initial          ← Round 1: Bull opens
      │
  bear_response         ← Round 1: Bear responds specifically to Bull's points
      │
  bull_rebuttal         ← Round 2: Bull dismantles Bear's counter-arguments
      │
  bear_counter          ← Round 2: Bear's closing counter
      │
  fact_checker          ← Guardrail: cross-checks all numerical claims vs ground_truth
      │
  arbiter               ← Synthesises, scores, and issues consensus forecast
      │
     END
"""

from langgraph.graph import END, StateGraph

from agents.arbiter import arbiter_node
from agents.bear import bear_counter_node, bear_response_node
from agents.bull import bull_initial_node, bull_rebuttal_node
from agents.fact_checker import fact_checker_node
from agents.researcher import researcher_node
from graph.state import MarketState


def build_graph():
    builder = StateGraph(MarketState)

    # Register nodes
    builder.add_node("researcher", researcher_node)
    builder.add_node("bull_initial", bull_initial_node)
    builder.add_node("bear_response", bear_response_node)
    builder.add_node("bull_rebuttal", bull_rebuttal_node)
    builder.add_node("bear_counter", bear_counter_node)
    builder.add_node("fact_checker", fact_checker_node)
    builder.add_node("arbiter", arbiter_node)

    # Wire the debate chain
    builder.set_entry_point("researcher")
    builder.add_edge("researcher", "bull_initial")
    builder.add_edge("bull_initial", "bear_response")
    builder.add_edge("bear_response", "bull_rebuttal")
    builder.add_edge("bull_rebuttal", "bear_counter")
    builder.add_edge("bear_counter", "fact_checker")
    builder.add_edge("fact_checker", "arbiter")
    builder.add_edge("arbiter", END)

    return builder.compile()
