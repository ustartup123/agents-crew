"""
graph/project_graph.py — Main LangGraph StateGraph for the idea-to-completion workflow.

Graph topology:
  START → ceo_router
  ceo_router → product (always)
  product → dev  (normal flow or after answering Dev's question)
  dev → product  (if dev has a clarification question)
  dev → qa       (if no clarification needed and code is written)
  qa → dev       (if tests fail and iterations < max)
  qa → business_and_summary (if approved or max iterations)
  business_and_summary → END

Note: Business agents (CFO, Marketing, Sales) are invoked inside the
business_and_summary node based on agents_needed, keeping the graph simple.
"""

from langgraph.graph import StateGraph, START, END
from graph.state import ProjectState
from graph.nodes import (
    ceo_router_node, product_node, dev_node, qa_node,
    cfo_node, marketing_node, sales_node, ceo_summary_node,
)

def wait_for_founder_input(state: ProjectState) -> dict:
    """Dummy node that gets interrupted, waiting for founder to reply."""
    return {}

def _route_after_ceo(state: ProjectState) -> str:
    """After CEO evaluation, either wait for founder or go to product."""
    if state.get("waiting_on_founder"):
        return "wait_for_founder"
    return "product"


def _route_after_product(state: ProjectState) -> str:
    """After Product runs, go to Dev."""
    return "dev"


def _route_after_dev(state: ProjectState) -> str:
    """After Dev runs, either ask Product for clarification or go to QA."""
    pending = state.get("pending_clarification") or {}
    if pending and not pending.get("answered"):
        return "product"
    return "qa"


def _route_after_qa(state: ProjectState) -> str:
    """After QA, approve or send back to Dev. If too many iterations, escalate."""
    if state.get("qa_approved"):
        return "business_and_summary"
    iterations = state.get("code_iterations", 0)
    max_iter = state.get("max_code_iterations", 3)
    if iterations >= max_iter:
        return "business_and_summary"
    return "dev"


def _business_and_summary_node(state: ProjectState) -> dict:
    """Run business agents (if needed) then CEO summary — all in one node."""
    needed = state.get("agents_needed", [])
    combined_state: dict = {}

    # Run business agents sequentially if they're in the needed list
    if "cfo" in needed:
        result = cfo_node(state)
        combined_state.update(result)

    if "marketing" in needed:
        result = marketing_node(state)
        combined_state.update(result)

    if "sales" in needed:
        result = sales_node(state)
        combined_state.update(result)

    # Merge business state into a temporary state for CEO summary
    merged_state = {**state, **combined_state}
    summary_result = ceo_summary_node(merged_state)

    # Combine all messages from business agents + summary
    all_messages = combined_state.get("messages", []) + summary_result.get("messages", [])
    combined_state.update(summary_result)
    combined_state["messages"] = all_messages

    return combined_state


def build_project_graph(checkpointer):
    """Build and compile the project graph with checkpointing."""
    g = StateGraph(ProjectState)

    # Add nodes
    g.add_node("ceo_router", ceo_router_node)
    g.add_node("wait_for_founder", wait_for_founder_input)
    g.add_node("product", product_node)
    g.add_node("dev", dev_node)
    g.add_node("qa", qa_node)
    g.add_node("business_and_summary", _business_and_summary_node)

    # Edges
    g.add_edge(START, "ceo_router")
    g.add_conditional_edges("ceo_router", _route_after_ceo, {
        "wait_for_founder": "wait_for_founder",
        "product": "product",
    })
    g.add_edge("wait_for_founder", "ceo_router")
    g.add_conditional_edges("product", _route_after_product, {"dev": "dev"})
    g.add_conditional_edges("dev", _route_after_dev, {"product": "product", "qa": "qa"})
    g.add_conditional_edges("qa", _route_after_qa, {
        "dev": "dev",
        "business_and_summary": "business_and_summary",
    })
    g.add_edge("business_and_summary", END)

    return g.compile(checkpointer=checkpointer, interrupt_before=["wait_for_founder"])
