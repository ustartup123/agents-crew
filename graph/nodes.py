"""
graph/nodes.py — All agent node functions for the LangGraph project graph.

Each agent is a LangGraph node — a function that receives ProjectState,
invokes its respective Agent class, posts to Slack, and returns updated state.

Nodes now also:
- Post "starting" / "finished" messages to Slack for founder visibility
- Track current_node / status_summary for the status command
- Consume founder_feedback if directed at this agent
- Set up pending_review for review-gate interrupt nodes
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
from graph.project_registry import update_project

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


def _now_iso() -> str:
    return datetime.now(timezone.utc).isoformat()


def _get_founder_feedback(state: ProjectState, agent_key: str) -> str:
    """Extract any founder feedback directed at this agent."""
    feedback_items = [
        f for f in state.get("founder_feedback", [])
        if f.get("agent", "").lower() == agent_key.lower()
    ]
    if not feedback_items:
        return ""
    return "\n".join(f"Founder feedback: {f['message']}" for f in feedback_items)


def _track_node(state: ProjectState, node_name: str, summary: str) -> None:
    """Update the project registry with current node info."""
    thread_ts = state.get("slack_thread_ts") or state.get("project_id", "")
    if thread_ts:
        update_project(
            thread_ts,
            current_node=node_name,
            status_summary=summary,
        )


# ── Nodes ─────────────────────────────────────────────────────────────────────

def ceo_router_node(state: ProjectState) -> dict:
    _post_to_slack(state, "CEO", ":brain: CEO is evaluating the idea...")
    _track_node(state, "ceo_router", "CEO evaluating idea")

    idea = state["idea"]
    project_id = state["project_id"]

    agent = CEOAgent()
    parsed = agent.execute_kickoff(
        idea,
        project_id,
        state.get("slack_channel", slack_cfg.channel_general),
        history=state.get("idea_refinement_history", [])
    )

    action = parsed.get("action")
    if action == "ask":
        question = parsed.get("question")
        _post_to_slack(state, agent.role, f"Founder, I need more info: {question}")
        _track_node(state, "wait_for_founder", "Waiting for founder clarification")
        return {
            "waiting_on_founder": True,
            "idea_refinement_history": [f"CEO: {question}"],
            "current_node": "wait_for_founder",
            "status_summary": f"Waiting for founder: {question[:100]}",
        }

    # Kickoff path
    kickoff_summary = parsed.get("kickoff_summary")
    if not kickoff_summary:
        parsed_str = str(parsed)
        kickoff_summary = parsed_str[:500]

    _post_to_slack(state, agent.role, kickoff_summary)
    _track_node(state, "ceo_router", "CEO kickoff complete, moving to Product")

    return {
        "waiting_on_founder": False,
        "phase": "planning",
        "project_name": parsed.get("project_name", project_id),
        "agents_needed": parsed.get("agents_needed", ["product", "dev", "qa"]),
        "github_repo_url": parsed.get("github_repo_url", ""),
        "github_repo_name": parsed.get("github_repo_name", ""),
        "notion_task_db_id": parsed.get("notion_task_db_id", ""),
        "messages": [_make_message("ceo", "team", kickoff_summary or "", "update")],
        "current_node": "product",
        "node_started_at": _now_iso(),
        "status_summary": "CEO kickoff complete, Product next.",
    }


def product_node(state: ProjectState) -> dict:
    _post_to_slack(state, "Product Manager", ":memo: Starting PRD creation...")
    _track_node(state, "product", "Product Manager writing PRD")

    # Consume founder feedback if any
    founder_fb = _get_founder_feedback(state, "product")

    # Check if this is a redo after rejection
    rejection = state.get("review_rejection_reason", "")

    agent = ProductAgent()
    # Pass feedback/rejection context via state (agent reads from state)
    if rejection or founder_fb:
        extra = ""
        if rejection:
            extra += f"\n\nThe founder rejected the previous PRD with this feedback: {rejection}"
        if founder_fb:
            extra += f"\n\n{founder_fb}"
        # Temporarily augment the idea with extra context
        augmented_state = {**state, "idea": state["idea"] + extra}
        parsed = agent.execute(augmented_state)
    else:
        parsed = agent.execute(state)

    prd_url = parsed.get("notion_prd_url", "")
    _post_to_slack(state, "Product Manager", f":white_check_mark: PRD complete. See Notion: {prd_url}\n\n:eyes: *Waiting for founder review.* Reply `approve` or `reject <reason>`.")

    new_state: dict = {
        "messages": [_make_message("product", "team", "PRD complete.", "update")],
        "notion_prd_url": prd_url,
        "prd_content": parsed.get("prd_content", str(parsed)),
        "current_node": "wait_for_prd_review",
        "node_started_at": _now_iso(),
        "status_summary": "PRD complete, awaiting founder review.",
        "review_rejection_reason": "",  # Clear previous rejection
        "pending_review": {
            "gate": "prd_review",
            "content_summary": parsed.get("prd_content", str(parsed))[:500],
            "approved": None,
        },
    }

    if parsed.get("clarification_answer"):
        new_state["pending_clarification"] = {
            **state["pending_clarification"],
            "answer": parsed["clarification_answer"],
            "answered": True,
        }
        _post_to_slack(state, "Product Manager", f"Answering Dev's question: {parsed['clarification_answer']}")

    _track_node(state, "wait_for_prd_review", "PRD awaiting founder review")
    return new_state


def dev_node(state: ProjectState) -> dict:
    _post_to_slack(state, "Lead Developer", ":computer: Starting development...")
    _track_node(state, "dev", "Lead Developer writing code")

    # Consume founder feedback and rejection reason
    founder_fb = _get_founder_feedback(state, "dev")
    rejection = state.get("review_rejection_reason", "")

    agent = DevAgent()
    if rejection or founder_fb:
        extra = ""
        if rejection:
            extra += f"\n\nThe founder rejected the previous code with this feedback: {rejection}"
        if founder_fb:
            extra += f"\n\n{founder_fb}"
        augmented_state = {**state, "idea": state["idea"] + extra}
        parsed = agent.execute(augmented_state)
    else:
        parsed = agent.execute(state)

    close_sandbox()

    new_state: dict = {
        "messages": [_make_message("dev", "team", parsed.get("summary", "Dev work complete."), "update")],
        "arch_content": parsed.get("summary", state.get("arch_content", "")),
        "qa_feedback": "",
        "current_node": "dev",
        "node_started_at": _now_iso(),
        "review_rejection_reason": "",  # Clear previous rejection
    }

    if parsed.get("needs_clarification"):
        question = parsed.get("question", "")
        _post_to_slack(state, "Lead Developer", f"Need clarification from Product before coding:\n{question}")
        new_state["pending_clarification"] = {
            "from": "dev",
            "to": "product",
            "question": question,
            "answered": False,
        }
        new_state["status_summary"] = f"Dev waiting on Product clarification"
    else:
        _post_to_slack(state, "Lead Developer", f":white_check_mark: Code committed to GitHub. Summary: {parsed.get('summary', '')}")
        new_state["pending_clarification"] = None
        new_state["status_summary"] = "Code complete, moving to QA."

    _track_node(state, new_state.get("current_node", "dev"), new_state.get("status_summary", ""))
    return new_state


def qa_node(state: ProjectState) -> dict:
    _post_to_slack(state, "QA Lead", ":mag: Running QA tests...")
    _track_node(state, "qa", "QA Lead testing code")

    agent = QAAgent()
    parsed = agent.execute(state)
    close_sandbox()

    approved = parsed.get("qa_approved", False)
    feedback = parsed.get("qa_feedback", "")
    summary = parsed.get("test_summary", "")

    iterations = state.get("code_iterations", 0) + 1

    if approved:
        _post_to_slack(state, "QA Lead", f":white_check_mark: QA APPROVED. {summary}\n\n:eyes: *Waiting for founder code review.* Reply `approve` or `reject <reason>`.")
        status = "QA approved, awaiting founder code review."
    else:
        if iterations >= state.get("max_code_iterations", 3):
            _post_to_slack(state, "QA Lead", f":warning: QA FAILED after {iterations} iterations. Moving to code review.\n{feedback}\n\n:eyes: *Waiting for founder code review.* Reply `approve` or `reject <reason>`.")
            status = f"QA failed after {iterations} iterations, awaiting founder code review."
        else:
            _post_to_slack(state, "QA Lead", f":x: QA FAILED — sending back to Dev (iteration {iterations}).\n{feedback}")
            status = f"QA failed, sending back to Dev (iteration {iterations})."

    new_state = {
        "qa_approved": approved,
        "qa_feedback": feedback,
        "code_iterations": iterations,
        "messages": [_make_message("qa", "dev" if not approved else "team",
                                   feedback if not approved else summary,
                                   "bug_report" if not approved else "approval")],
        "current_node": "qa",
        "node_started_at": _now_iso(),
        "status_summary": status,
    }

    # Set up code review gate if QA approved or max iterations reached
    if approved or iterations >= state.get("max_code_iterations", 3):
        new_state["pending_review"] = {
            "gate": "code_review",
            "content_summary": summary[:500] if summary else feedback[:500],
            "approved": None,
        }
        _track_node(state, "wait_for_code_review", "Code awaiting founder review")

    _track_node(state, "qa", status)
    return new_state


def cfo_node(state: ProjectState) -> dict:
    _post_to_slack(state, "CFO", ":chart_with_upwards_trend: Working on financial plan...")
    _track_node(state, "cfo", "CFO creating financial plan")

    agent = CFOAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, "CFO", f":white_check_mark: Financial plan complete. {parsed.get('summary', '')}")

    return {
        "notion_financial_url": parsed.get("notion_financial_url", ""),
        "messages": [_make_message("cfo", "team", "Financial plan complete.", "update")],
        "current_node": "cfo",
        "status_summary": "Financial plan complete.",
    }


def marketing_node(state: ProjectState) -> dict:
    _post_to_slack(state, "Marketing", ":mega: Working on GTM strategy...")
    _track_node(state, "marketing", "Marketing creating GTM strategy")

    # Consume founder feedback and rejection reason
    founder_fb = _get_founder_feedback(state, "marketing")
    rejection = state.get("review_rejection_reason", "")

    agent = MarketingAgent()
    if rejection or founder_fb:
        extra = ""
        if rejection:
            extra += f"\n\nThe founder rejected the previous GTM plan with this feedback: {rejection}"
        if founder_fb:
            extra += f"\n\n{founder_fb}"
        augmented_state = {**state, "idea": state["idea"] + extra}
        parsed = agent.execute(augmented_state)
    else:
        parsed = agent.execute(state)

    gtm_url = parsed.get("notion_gtm_url", "")
    gtm_content = parsed.get("gtm_content", parsed.get("summary", str(parsed)))

    _post_to_slack(
        state, "Marketing",
        f":white_check_mark: GTM strategy complete. See Notion: {gtm_url}\n\n"
        f":eyes: *Waiting for founder review.* Reply `approve` or `reject <reason>`."
    )

    _track_node(state, "wait_for_marketing_review", "GTM plan awaiting founder review")
    return {
        "notion_gtm_url": gtm_url,
        "gtm_content": gtm_content,
        "messages": [_make_message("marketing", "team", "GTM strategy complete.", "update")],
        "current_node": "wait_for_marketing_review",
        "node_started_at": _now_iso(),
        "status_summary": "GTM plan complete, awaiting founder review.",
        "review_rejection_reason": "",
        "pending_review": {
            "gate": "marketing_review",
            "content_summary": gtm_content[:500],
            "approved": None,
        },
    }


def sales_node(state: ProjectState) -> dict:
    _post_to_slack(state, "Sales", ":handshake: Working on sales playbook...")
    _track_node(state, "sales", "Sales creating playbook")

    agent = SalesAgent()
    parsed = agent.execute(state)

    _post_to_slack(state, "Sales", f":white_check_mark: Sales playbook complete. {parsed.get('summary', '')}")

    return {
        "notion_sales_url": parsed.get("notion_sales_url", ""),
        "messages": [_make_message("sales", "team", "Sales playbook complete.", "update")],
        "current_node": "sales",
        "status_summary": "Sales playbook complete.",
    }


def ceo_summary_node(state: ProjectState) -> dict:
    _post_to_slack(state, "CEO", ":star: Preparing final project summary...")
    _track_node(state, "ceo_summary", "CEO writing final summary")

    agent = CEOAgent()
    parsed = agent.execute_summary(state)

    _post_to_slack(state, "CEO", ":tada: Project complete! See Notion for full launch plan.")

    _track_node(state, "done", "Project complete.")
    return {
        "phase": "done",
        "messages": [_make_message("ceo", "founder", "Project complete.", "update")],
        "current_node": "done",
        "status_summary": "Project complete.",
    }
