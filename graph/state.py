"""
graph/state.py — Shared state passed between all LangGraph nodes.

Every project has a unique thread_id (= project_id) stored in the SQLite
checkpointer. Loading state:
  graph.invoke(input, config={"configurable": {"thread_id": "project-xyz"}})
"""

from __future__ import annotations

from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator


class Message(TypedDict):
    """A message between two agents, also posted to Slack."""
    from_agent: str       # e.g. "dev"
    to_agent: str         # e.g. "product"
    content: str
    message_type: str     # "question", "answer", "update", "bug_report", "approval"
    timestamp: str


class ProjectState(TypedDict):
    # ── Identity ─────────────────────────────────────────────────────────────
    project_id: str               # = LangGraph thread_id
    project_name: str             # short slug, also GitHub repo name
    idea: str                     # the original idea from the founder
    idea_refinement_history: Annotated[list[str], operator.add]
    waiting_on_founder: bool

    # ── Routing ──────────────────────────────────────────────────────────────
    phase: str                    # "routing" | "planning" | "development" | "qa" | "business" | "summary" | "done"
    agents_needed: list[str]      # CEO decides, e.g. ["product", "dev", "qa"] or all 7

    # ── Communication (visible in Slack) ─────────────────────────────────────
    # Use Annotated with operator.add so messages are appended, not overwritten
    messages: Annotated[list[Message], operator.add]
    slack_channel: str            # channel where this project is discussed
    slack_thread_ts: str          # thread timestamp — all messages go in one thread

    # ── Pending clarification (feedback loops) ────────────────────────────────
    pending_clarification: Optional[dict]  # {"from": "dev", "to": "product", "question": "...", "answered": False}

    # ── Documents produced (Notion pages) ────────────────────────────────────
    notion_vision_url: str
    notion_prd_url: str
    notion_arch_url: str
    notion_test_strategy_url: str
    notion_financial_url: str
    notion_gtm_url: str
    notion_sales_url: str
    notion_task_db_id: str        # ID of the master task board database

    # ── Document content (passed as context between agents) ───────────────────
    vision_content: str
    prd_content: str
    arch_content: str
    gtm_content: str              # Marketing GTM strategy content

    # ── Development ───────────────────────────────────────────────────────────
    github_repo_url: str
    github_repo_name: str         # owner/repo format
    code_iterations: int          # how many dev→qa→dev cycles completed
    max_code_iterations: int      # default 3; CEO can escalate if exceeded
    qa_approved: bool
    qa_feedback: str              # bug report from QA to Dev (empty if approved)

    # ── Founder control ───────────────────────────────────────────────────────
    paused: bool                      # True when founder has paused the project
    founder_feedback: Annotated[list[dict], operator.add]  # {"agent": "dev", "message": "...", "ts": "..."}

    # ── Review gates ──────────────────────────────────────────────────────────
    pending_review: Optional[dict]    # {"gate": "prd_review"|"code_review", "content_summary": "...", "approved": None|True|False}
    review_rejection_reason: str      # Founder's rejection message, passed back to the agent

    # ── Status tracking ───────────────────────────────────────────────────────
    current_node: str                 # Name of the currently-executing node
    node_started_at: str              # ISO timestamp when current node began
    status_summary: str               # Human-readable one-liner set by each node

    # ── Error tracking ────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]


def make_initial_state(
    project_id: str,
    idea: str,
    slack_channel: str = "",
    slack_thread_ts: str = "",
) -> dict:
    """Create a complete initial state dict with all ProjectState fields populated.

    Using a single factory avoids the risk of main.py and slack_bot.py
    drifting out of sync when new fields are added to ProjectState.
    """
    return {
        "project_id": project_id,
        "project_name": "",
        "idea": idea,
        "idea_refinement_history": [],
        "waiting_on_founder": False,
        "phase": "routing",
        "agents_needed": [],
        "messages": [],
        "slack_channel": slack_channel,
        "slack_thread_ts": slack_thread_ts,
        "pending_clarification": None,
        "code_iterations": 0,
        "max_code_iterations": 3,
        "qa_approved": False,
        "qa_feedback": "",
        "github_repo_url": "",
        "github_repo_name": "",
        "paused": False,
        "founder_feedback": [],
        "pending_review": None,
        "review_rejection_reason": "",
        "current_node": "",
        "node_started_at": "",
        "status_summary": "Project created, awaiting kickoff.",
        "errors": [],
        "notion_vision_url": "",
        "notion_prd_url": "",
        "notion_arch_url": "",
        "notion_test_strategy_url": "",
        "notion_financial_url": "",
        "notion_gtm_url": "",
        "notion_sales_url": "",
        "notion_task_db_id": "",
        "vision_content": "",
        "prd_content": "",
        "arch_content": "",
        "gtm_content": "",
    }
