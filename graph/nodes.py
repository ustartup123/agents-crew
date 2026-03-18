"""
graph/nodes.py — All agent node functions for the LangGraph project graph.

Each agent is a LangGraph node — a function that receives ProjectState,
invokes its respective Agent class, posts to Slack, and returns updated state.
"""

from __future__ import annotations

import logging
from datetime import datetime, timezone

from config.settings import slack_cfg
from agents.ceo_agent import CEOAgent
from agents.product_agent import ProductAgent
from agents.dev_agent import DevAgent
from agents.qa_agent import QAAgent
from agents.business_agents import CFOAgent, MarketingAgent, SalesAgent
from tools.slack_tools import slack_reply_thread, slack_send_message
from tools.code_exec_tools import close_sandbox
from graph.state import ProjectState, Message

logger = logging.getLogger(__name__)

# ── Shared helpers ───────────────────────────────────────────────────────────

def _post_to_slack(state: ProjectState, agent_role: str, message: str) -> None:
    """Post a message to the project's Slack thread."""
    channel = state.get("slack_channel") or slack_cfg.channel_general
    thread_ts = state.get("slack_thread_ts", "")

    try:
        if thread_ts:
            slack_reply_thread.invoke({
                "channel": channel,
                "thread_ts": thread_ts,
                "message": message,
                "agent_name": agent_role,
            })
        else:
            slack_send_message.invoke({
                "channel": channel,
                "message": message,
                "agent_name": agent_role,
            })
    except Exception as e:
        logger.error(f"Slack post error ({agent_role}): {e}")


def _make_message(from_agent: str, to_agent: str, content: str, msg_type: str) -> Message:
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        message_type=msg_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


# ── Nodes ─────────────────────────────────────────────────────────────────────

def ceo_router_node(state: ProjectState) -> dict:
    idea = state["idea"]
    project_id = state["project_id"]
    
    agent = CEOAgent()
    parsed = agent.execute_kickoff(idea, project_id, state.get("slack_channel", slack_cfg.channel_general))

    kickoff_summary = parsed.get("kickoff_summary")
    if not kickoff_summary:
        parsed_str = str(parsed)
        kickoff_summary = parsed_str[:500]

    _post_to_slack(state, agent.role, kickoff_summary)

    return {
        "phase": "planning",
        "project_name": parsed.get("project_name", project_id),
        "agents_needed": parsed.get("agents_needed", ["product", "dev", "qa"]),
        "github_repo_url": parsed.get("github_repo_url", ""),
        "github_repo_name": parsed.get("github_repo_name", ""),
        "notion_task_db_id": parsed.get("notion_task_db_id", ""),
        "messages": [_make_message("ceo", "team", kickoff_summary or "", "update")],
    }


def product_node(state: ProjectState) -> dict:
    agent = ProductAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, agent.role, f"PRD complete. See Notion: {parsed.get('notion_prd_url', '')}")

    new_state: dict = {
        "messages": [_make_message("product", "team", "PRD complete.", "update")],
        "notion_prd_url": parsed.get("notion_prd_url", ""),
        "prd_content": parsed.get("prd_content", str(parsed)),
    }

    if parsed.get("clarification_answer"):
        new_state["pending_clarification"] = {
            **state["pending_clarification"],
            "answer": parsed["clarification_answer"],
            "answered": True,
        }
        _post_to_slack(state, agent.role, f"Answering Dev's question: {parsed['clarification_answer']}")

    return new_state


def dev_node(state: ProjectState) -> dict:
    agent = DevAgent()
    parsed = agent.execute(state)
    close_sandbox()

    new_state: dict = {
        "messages": [_make_message("dev", "team", parsed.get("summary", "Dev work complete."), "update")],
        "arch_content": parsed.get("summary", state.get("arch_content", "")),
        "qa_feedback": "",
    }

    if parsed.get("needs_clarification"):
        question = parsed.get("question", "")
        _post_to_slack(state, agent.role, f"Need clarification from Product before coding:\n{question}")
        new_state["pending_clarification"] = {
            "from": "dev",
            "to": "product",
            "question": question,
            "answered": False,
        }
    else:
        _post_to_slack(state, agent.role, f"Code committed to GitHub. Summary: {parsed.get('summary', '')}")
        new_state["pending_clarification"] = None

    return new_state


def qa_node(state: ProjectState) -> dict:
    agent = QAAgent()
    parsed = agent.execute(state)
    close_sandbox()

    approved = parsed.get("qa_approved", False)
    feedback = parsed.get("qa_feedback", "")
    summary = parsed.get("test_summary", "")

    if approved:
        _post_to_slack(state, agent.role, f"QA APPROVED. {summary}")
    else:
        _post_to_slack(state, agent.role, f"QA FAILED — sending back to Dev.\n{feedback}")

    return {
        "qa_approved": approved,
        "qa_feedback": feedback,
        "code_iterations": state.get("code_iterations", 0) + 1,
        "messages": [_make_message("qa", "dev" if not approved else "team",
                                   feedback if not approved else summary,
                                   "bug_report" if not approved else "approval")],
    }


def cfo_node(state: ProjectState) -> dict:
    agent = CFOAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, agent.role, f"Financial plan complete. {parsed.get('summary', '')}")

    return {
        "notion_financial_url": parsed.get("notion_financial_url", ""),
        "messages": [_make_message("cfo", "team", "Financial plan complete.", "update")],
    }


def marketing_node(state: ProjectState) -> dict:
    agent = MarketingAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, agent.role, f"GTM strategy complete. {parsed.get('summary', '')}")

    return {
        "notion_gtm_url": parsed.get("notion_gtm_url", ""),
        "messages": [_make_message("marketing", "team", "GTM strategy complete.", "update")],
    }


def sales_node(state: ProjectState) -> dict:
    agent = SalesAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, agent.role, f"Sales playbook complete. {parsed.get('summary', '')}")

    return {
        "notion_sales_url": parsed.get("notion_sales_url", ""),
        "messages": [_make_message("sales", "team", "Sales playbook complete.", "update")],
    }


def ceo_summary_node(state: ProjectState) -> dict:
    agent = CEOAgent()
    parsed = agent.execute_summary(state)

    _post_to_slack(state, agent.role, "Project complete! See Notion for full launch plan.")

    return {
        "phase": "done",
        "messages": [_make_message("ceo", "founder", "Project complete.", "update")],
    }
