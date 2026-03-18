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

    # ── Development ───────────────────────────────────────────────────────────
    github_repo_url: str
    github_repo_name: str         # owner/repo format
    code_iterations: int          # how many dev→qa→dev cycles completed
    max_code_iterations: int      # default 3; CEO can escalate if exceeded
    qa_approved: bool
    qa_feedback: str              # bug report from QA to Dev (empty if approved)

    # ── Error tracking ────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
