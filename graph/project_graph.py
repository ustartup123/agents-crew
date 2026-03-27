"""
graph/project_graph.py — Main LangGraph StateGraph for the idea-to-completion workflow.

Graph topology:
  START → ceo_router
  ceo_router → wait_for_founder | product
  wait_for_founder → ceo_router
  product → wait_for_prd_review             (founder reviews PRD)
  wait_for_prd_review → product | dev       (reject → redo, approve → dev)
  dev → product | qa                        (clarification loop or proceed)
  qa → dev | wait_for_code_review           (fail → redo, pass → founder reviews code)
  wait_for_code_review → dev | marketing    (reject → redo, approve → marketing)
  marketing → wait_for_marketing_review     (founder reviews GTM plan)
  wait_for_marketing_review → marketing | finance_sales_summary
  finance_sales_summary → END
"""

from langgraph.graph import StateGraph, START, END
from graph.state import ProjectState
from graph.nodes import (
    ceo_router_node, product_node, dev_node, qa_node,
    cfo_node, marketing_node, sales_node, ceo_summary_node,
)


# ── Interrupt (wait) nodes ───────────────────────────────────────────────────

def wait_for_founder_input(state: ProjectState) -> dict:
    """Interrupted node — waiting for founder to clarify the idea."""
    return {}


def wait_for_prd_review(state: ProjectState) -> dict:
    """Interrupted node — waiting for founder to approve/reject the PRD."""
    return {}


def wait_for_code_review(state: ProjectState) -> dict:
    """Interrupted node — waiting for founder to approve/reject the code."""
    return {}


def wait_for_marketing_review(state: ProjectState) -> dict:
    """Interrupted node — waiting for founder to approve/reject the GTM plan."""
    return {}


# ── Routing functions ────────────────────────────────────────────────────────

def _route_after_ceo(state: ProjectState) -> str:
    if state.get("waiting_on_founder"):
        return "wait_for_founder"
    return "product"


def _route_after_prd_review(state: ProjectState) -> str:
    review = state.get("pending_review") or {}
    if review.get("approved") is False:
        return "product"
    return "dev"


def _route_after_dev(state: ProjectState) -> str:
    pending = state.get("pending_clarification") or {}
    if pending and not pending.get("answered"):
        return "product"
    return "qa"


def _route_after_qa(state: ProjectState) -> str:
    if state.get("qa_approved"):
        return "wait_for_code_review"
    iterations = state.get("code_iterations", 0)
    max_iter = state.get("max_code_iterations", 3)
    if iterations >= max_iter:
        return "wait_for_code_review"
    return "dev"


def _route_after_code_review(state: ProjectState) -> str:
    review = state.get("pending_review") or {}
    if review.get("approved") is False:
        return "dev"
    return "marketing"


def _route_after_marketing_review(state: ProjectState) -> str:
    review = state.get("pending_review") or {}
    if review.get("approved") is False:
        return "marketing"
    return "finance_sales_summary"


# ── Finance + Sales + CEO summary combo node ─────────────────────────────────

def _finance_sales_summary_node(state: ProjectState) -> dict:
    """Run CFO, Sales (if needed) then CEO summary — all in one node.

    Marketing now runs as its own node with a review gate, so it's
    no longer included here.
    """
    needed = state.get("agents_needed", [])
    combined_state: dict = {}
    all_messages: list = []

    if "cfo" in needed:
        result = cfo_node(state)
        all_messages.extend(result.pop("messages", []))
        combined_state.update(result)

    if "sales" in needed:
        result = sales_node(state)
        all_messages.extend(result.pop("messages", []))
        combined_state.update(result)

    merged_state = {**state, **combined_state}
    summary_result = ceo_summary_node(merged_state)
    all_messages.extend(summary_result.pop("messages", []))
    combined_state.update(summary_result)

    combined_state["messages"] = all_messages
    return combined_state


# ── Graph builder ────────────────────────────────────────────────────────────

def build_project_graph(checkpointer):
    """Build and compile the project graph with checkpointing."""
    g = StateGraph(ProjectState)

    # Nodes
    g.add_node("ceo_router", ceo_router_node)
    g.add_node("wait_for_founder", wait_for_founder_input)
    g.add_node("product", product_node)
    g.add_node("wait_for_prd_review", wait_for_prd_review)
    g.add_node("dev", dev_node)
    g.add_node("qa", qa_node)
    g.add_node("wait_for_code_review", wait_for_code_review)
    g.add_node("marketing", marketing_node)
    g.add_node("wait_for_marketing_review", wait_for_marketing_review)
    g.add_node("finance_sales_summary", _finance_sales_summary_node)

    # Edges
    g.add_edge(START, "ceo_router")

    g.add_conditional_edges("ceo_router", _route_after_ceo, {
        "wait_for_founder": "wait_for_founder",
        "product": "product",
    })
    g.add_edge("wait_for_founder", "ceo_router")

    g.add_edge("product", "wait_for_prd_review")

    g.add_conditional_edges("wait_for_prd_review", _route_after_prd_review, {
        "product": "product",
        "dev": "dev",
    })

    g.add_conditional_edges("dev", _route_after_dev, {
        "product": "product",
        "qa": "qa",
    })

    g.add_conditional_edges("qa", _route_after_qa, {
        "dev": "dev",
        "wait_for_code_review": "wait_for_code_review",
    })

    g.add_conditional_edges("wait_for_code_review", _route_after_code_review, {
        "dev": "dev",
        "marketing": "marketing",
    })

    g.add_edge("marketing", "wait_for_marketing_review")

    g.add_conditional_edges("wait_for_marketing_review", _route_after_marketing_review, {
        "marketing": "marketing",
        "finance_sales_summary": "finance_sales_summary",
    })

    g.add_edge("finance_sales_summary", END)

    return g.compile(
        checkpointer=checkpointer,
        interrupt_before=[
            "wait_for_founder",
            "wait_for_prd_review",
            "wait_for_code_review",
            "wait_for_marketing_review",
        ],
    )
