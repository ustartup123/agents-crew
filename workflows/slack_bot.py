"""
workflows/slack_bot.py — Real-time Slack bot that routes messages to the right agent.

Listens for @mentions and DMs, determines which agent should respond,
and kicks off a LangGraph task to handle the request.

Founder commands (in any channel or DM):
  status              — Overview of all active projects
  status <name>       — Detailed status of one project
  products            — List all products (active + completed)
  demo                — Run a demo of the project code (in-thread)
  demo <name>         — Run a demo of a specific project
  feedback <agent>: … — Inject feedback for an agent mid-workflow
  pause <name>        — Pause a running project
  resume <name>       — Resume a paused project
  approve             — Approve the current review gate (in-thread)
  reject <reason>     — Reject and send back for rework (in-thread)
"""

from __future__ import annotations

import json
import logging
import re
import threading
from datetime import datetime, timezone
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents import AGENT_PERSONAS
from config.settings import slack_cfg, gemini_cfg, app_cfg
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
from graph.state import make_initial_state
from graph.project_registry import (
    register_project, update_project, get_project,
    get_all_projects, get_all_products, get_project_by_name,
    get_projects_context_summary,
)
from tools.slack_tools import SLACK_TOOLS, slack_reply_thread
from tools.notion_tools import NOTION_TOOLS
from tools.github_tools import github_read_file, github_list_files
from tools.code_exec_tools import get_sandbox, close_sandbox

logger = logging.getLogger(__name__)

# All review gate node names
_REVIEW_GATES = ("wait_for_prd_review", "wait_for_code_review", "wait_for_marketing_review")

_GATE_LABELS = {
    "wait_for_prd_review": ("PRD", "Product Manager"),
    "wait_for_code_review": ("Code", "Lead Developer"),
    "wait_for_marketing_review": ("Marketing Plan", "Marketing"),
}

# ── Founder command parsing ──────────────────────────────────────────────────

_COMMAND_PATTERNS = [
    ("status",   re.compile(r"^status(?:\s+(.+))?$", re.IGNORECASE)),
    ("products", re.compile(r"^products?(?:\s+list)?$", re.IGNORECASE)),
    ("demo",     re.compile(r"^demo(?:\s+(.+))?$", re.IGNORECASE)),
    ("feedback", re.compile(r"^feedback\s+(\w+):\s*(.+)$", re.IGNORECASE | re.DOTALL)),
    ("pause",    re.compile(r"^pause(?:\s+(.+))?$", re.IGNORECASE)),
    ("resume",   re.compile(r"^resume(?:\s+(.+))?$", re.IGNORECASE)),
    ("approve",  re.compile(r"^approve$", re.IGNORECASE)),
    ("reject",   re.compile(r"^reject(?:\s+(.+))?$", re.IGNORECASE | re.DOTALL)),
]


def _parse_founder_command(text: str) -> Optional[tuple[str, tuple]]:
    """Return (command_name, regex_groups) or None if not a command."""
    text = text.strip()
    for cmd_name, pattern in _COMMAND_PATTERNS:
        m = pattern.match(text)
        if m:
            return (cmd_name, m.groups())
    return None


# ── Agent routing keywords ───────────────────────────────────────────────────

_ROUTE_KEYWORDS = {
    "ceo": ["ceo", "strategy", "vision", "decision", "priority", "roadmap", "okr", "team"],
    "cfo": ["cfo", "finance", "budget", "runway", "burn", "revenue", "pricing", "fundrais", "investor", "cost"],
    "product": ["product", "feature", "prd", "requirement", "backlog", "user story", "spec"],
    "dev": ["dev", "code", "engineer", "architect", "api", "database", "deploy", "ci/cd", "tech", "bug fix", "implement"],
    "qa": ["qa", "test", "quality", "bug", "regression", "edge case", "coverage"],
    "marketing": ["marketing", "brand", "content", "seo", "launch", "campaign", "social media", "growth", "awareness"],
    "sales": ["sales", "lead", "prospect", "deal", "pipeline", "outreach", "pitch", "close", "crm", "customer"],
}

PROJECT_TRIGGER_KEYWORDS = ["build", "create", "make", "develop", "idea", "startup", "launch", "new project"]


def _route_message(text: str) -> str:
    """Determine which agent key should handle a message based on keyword scoring."""
    text_lower = text.lower()

    ROLE_WEIGHT = 5
    KEYWORD_WEIGHT = 1

    scores = {key: 0 for key in _ROUTE_KEYWORDS}
    for key, keywords in _ROUTE_KEYWORDS.items():
        if keywords[0] in text_lower:
            scores[key] += ROLE_WEIGHT
        for kw in keywords[1:]:
            if kw in text_lower:
                scores[key] += KEYWORD_WEIGHT

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "ceo"


def _is_new_project(text: str) -> bool:
    return any(kw in text.lower() for kw in PROJECT_TRIGGER_KEYWORDS)


# ── Founder command handlers ─────────────────────────────────────────────────

def _enrich_project_from_checkpointer(proj: dict) -> dict:
    """Enrich a registry project dict with Notion URLs and other state from the checkpointer."""
    try:
        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": proj["project_id"]}}
        snapshot = graph.get_state(config)
        if snapshot and snapshot.values:
            vals = snapshot.values
            for key in (
                "notion_prd_url", "notion_financial_url", "notion_gtm_url",
                "notion_sales_url", "notion_arch_url", "notion_vision_url",
                "notion_task_db_id", "github_repo_url", "github_repo_name",
                "code_iterations", "qa_approved",
            ):
                if vals.get(key) and not proj.get(key):
                    proj[key] = vals[key]
            # Update status fields if checkpointer has fresher data
            if vals.get("status_summary"):
                proj["status_summary"] = vals["status_summary"]
            if vals.get("current_node"):
                proj["current_node"] = vals["current_node"]
            if vals.get("phase"):
                proj["phase"] = vals["phase"]
    except Exception:
        pass  # Graceful fallback — show what we have
    return proj


def _format_project_detail(proj: dict) -> str:
    """Format a single project into a rich Slack status block."""
    proj = _enrich_project_from_checkpointer(proj)
    name = proj.get("project_name") or proj["project_id"][:12]
    paused_tag = " :double_vertical_bar: *PAUSED*" if proj.get("paused") else ""

    lines = [f"*Project: {name}*{paused_tag}"]
    lines.append(f"> *Phase:* `{proj.get('current_node', '?')}`")
    lines.append(f"> *Status:* {proj.get('status_summary', 'Unknown')}")
    lines.append(f"> *Started:* {proj.get('started_at', '?')[:19]}")
    lines.append(f"> *Idea:* {proj.get('idea', '')[:200]}")

    # Links section
    links = []
    if proj.get("github_repo_url"):
        links.append(f"<{proj['github_repo_url']}|GitHub>")
    if proj.get("notion_prd_url"):
        links.append(f"<{proj['notion_prd_url']}|PRD>")
    if proj.get("notion_financial_url"):
        links.append(f"<{proj['notion_financial_url']}|Financial Plan>")
    if proj.get("notion_gtm_url"):
        links.append(f"<{proj['notion_gtm_url']}|GTM Strategy>")
    if proj.get("notion_sales_url"):
        links.append(f"<{proj['notion_sales_url']}|Sales Playbook>")
    if proj.get("notion_arch_url"):
        links.append(f"<{proj['notion_arch_url']}|Architecture>")
    if links:
        lines.append(f"> *Links:* {' · '.join(links)}")

    if proj.get("code_iterations"):
        qa_status = ":white_check_mark: Approved" if proj.get("qa_approved") else ":arrows_counterclockwise: In progress"
        lines.append(f"> *Dev iterations:* {proj['code_iterations']} — QA: {qa_status}")

    return "\n".join(lines)


def _handle_status(say, thread_ts: str, project_name: Optional[str]):
    """Show status of all projects, or one specific project."""
    if project_name:
        proj = get_project_by_name(project_name)
        if not proj:
            say(text=f"No active project matching *{project_name}*.", thread_ts=thread_ts)
            return
        say(text=_format_project_detail(proj), thread_ts=thread_ts)
        return

    # All active projects
    projects = get_all_projects()
    if not projects:
        say(text="No active projects right now.", thread_ts=thread_ts)
        return

    lines = [f"*Active Projects ({len(projects)}):*\n"]
    for p in projects:
        p = _enrich_project_from_checkpointer(p)
        paused = " :double_vertical_bar: PAUSED" if p.get("paused") else ""
        name = p.get("project_name") or p["project_id"][:12]
        github = f" · <{p['github_repo_url']}|GitHub>" if p.get("github_repo_url") else ""
        prd = f" · <{p['notion_prd_url']}|PRD>" if p.get("notion_prd_url") else ""
        lines.append(f"- *{name}*{paused} — `{p.get('current_node', '?')}` — {p.get('status_summary', '')}{github}{prd}")
    say(text="\n".join(lines), thread_ts=thread_ts)


def _handle_products(say, thread_ts: str):
    """List all products (active + completed)."""
    all_prods = get_all_products()
    if not all_prods:
        say(text="No products yet. Start a project to create your first product!", thread_ts=thread_ts)
        return

    lines = [f"*All Products ({len(all_prods)}):*\n"]
    for p in all_prods:
        p = _enrich_project_from_checkpointer(p)
        name = p.get("project_name") or p["project_id"][:12]
        phase = p.get("phase", p.get("current_node", "?"))
        idea = p.get("idea", "")[:100]

        if phase == "done" or p.get("current_node") == "done":
            status_icon = ":white_check_mark:"
            status_text = f"Completed {p.get('completed_at', '?')[:10]}"
        elif p.get("paused"):
            status_icon = ":double_vertical_bar:"
            status_text = "Paused"
        else:
            status_icon = ":arrows_counterclockwise:"
            status_text = f"In progress — `{p.get('current_node', '?')}`"

        line = f"{status_icon} *{name}* — {status_text}"
        if idea:
            line += f"\n    _{idea}_"

        # Collect links
        links = []
        if p.get("github_repo_url"):
            links.append(f"<{p['github_repo_url']}|GitHub>")
        if p.get("notion_prd_url"):
            links.append(f"<{p['notion_prd_url']}|PRD>")
        if p.get("notion_financial_url"):
            links.append(f"<{p['notion_financial_url']}|Financials>")
        if p.get("notion_gtm_url"):
            links.append(f"<{p['notion_gtm_url']}|GTM>")
        if links:
            line += f"\n    {' · '.join(links)}"
        lines.append(line)

    say(text="\n".join(lines), thread_ts=thread_ts)


def _handle_demo(say, channel: str, thread_ts: str, project_name: Optional[str]):
    """Run a demo of a project's code in the E2B sandbox."""
    # Find the project
    proj = None
    if project_name:
        proj = get_project_by_name(project_name)
    else:
        proj = get_project(thread_ts)

    if not proj:
        say(text="No matching project found. Use `demo <project-name>` or send this in a project thread.", thread_ts=thread_ts)
        return

    # Get the project state from checkpointer to find GitHub repo
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": proj["project_id"]}}

    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        say(text=f"Could not load project state: {e}", thread_ts=thread_ts)
        return

    if not snapshot or not snapshot.values:
        say(text="Could not find project state.", thread_ts=thread_ts)
        return

    vals = snapshot.values
    repo_name = vals.get("github_repo_name", "")
    if not repo_name:
        say(text="This project doesn't have a GitHub repo yet. The Dev agent hasn't run.", thread_ts=thread_ts)
        return

    name = proj.get("project_name") or proj["project_id"][:12]
    say(text=f":rocket: Starting demo for *{name}*...\nFetching code from `{repo_name}`...", thread_ts=thread_ts)

    # Run the demo in a background thread
    t = threading.Thread(
        target=_run_demo,
        args=(repo_name, channel, thread_ts, name),
        daemon=True,
    )
    t.start()


def _run_demo(repo_name: str, channel: str, thread_ts: str, project_name: str):
    """Fetch code from GitHub, run it in sandbox, post results to Slack."""
    try:
        # List files in the repo
        files_json = github_list_files.invoke({"repo_name": repo_name})
        files = json.loads(files_json) if isinstance(files_json, str) else files_json

        if isinstance(files, dict) and not files.get("ok", True):
            slack_reply_thread.invoke({
                "channel": channel, "thread_ts": thread_ts,
                "message": f"Could not list repo files: {files.get('error', 'unknown')}",
                "agent_name": "Demo Runner",
            })
            return

        # Find Python entry points
        py_files = [f for f in files if isinstance(f, dict) and f.get("name", "").endswith(".py")]
        entry_candidates = [f for f in py_files if f["name"] in ("main.py", "app.py", "server.py", "run.py")]
        if not entry_candidates:
            entry_candidates = py_files[:3]  # Just try the first few .py files

        if not entry_candidates:
            slack_reply_thread.invoke({
                "channel": channel, "thread_ts": thread_ts,
                "message": "No Python files found in the repo. Cannot run demo.",
                "agent_name": "Demo Runner",
            })
            return

        # Read and run the entry point
        entry_file = entry_candidates[0]
        code_content = github_read_file.invoke({
            "repo_name": repo_name,
            "file_path": entry_file["path"],
        })

        if isinstance(code_content, str) and code_content.startswith("{"):
            try:
                err = json.loads(code_content)
                if not err.get("ok", True):
                    slack_reply_thread.invoke({
                        "channel": channel, "thread_ts": thread_ts,
                        "message": f"Could not read `{entry_file['path']}`: {err.get('error')}",
                        "agent_name": "Demo Runner",
                    })
                    return
            except json.JSONDecodeError:
                pass

        # Post the code being run
        code_preview = code_content[:1500] if isinstance(code_content, str) else str(code_content)[:1500]
        slack_reply_thread.invoke({
            "channel": channel, "thread_ts": thread_ts,
            "message": f":page_facing_up: Running `{entry_file['path']}`:\n```\n{code_preview}\n```",
            "agent_name": "Demo Runner",
        })

        # Execute in sandbox
        sandbox = get_sandbox()
        sandbox.files.write(entry_file["path"], code_content)
        execution = sandbox.run_code(code_content)

        stdout = "\n".join(execution.logs.stdout) if execution.logs.stdout else "(no output)"
        stderr = "\n".join(execution.logs.stderr) if execution.logs.stderr else ""
        error = str(execution.error) if execution.error else ""

        result_parts = [f":computer: *Demo Output for {project_name}:*\n```\n{stdout[:2000]}\n```"]
        if stderr:
            result_parts.append(f":warning: *Stderr:*\n```\n{stderr[:1000]}\n```")
        if error:
            result_parts.append(f":x: *Error:*\n```\n{error[:1000]}\n```")

        slack_reply_thread.invoke({
            "channel": channel, "thread_ts": thread_ts,
            "message": "\n".join(result_parts),
            "agent_name": "Demo Runner",
        })

        close_sandbox()

    except Exception as e:
        logger.error(f"Demo error: {e}")
        try:
            slack_reply_thread.invoke({
                "channel": channel, "thread_ts": thread_ts,
                "message": f":x: Demo failed: {str(e)[:300]}",
                "agent_name": "Demo Runner",
            })
        except Exception:
            pass
        close_sandbox()


def _handle_feedback(say, thread_ts: str, agent_key: str, message: str):
    """Inject founder feedback for a specific agent into the project state."""
    agent_key = agent_key.lower().strip()
    valid_agents = {"ceo", "cfo", "product", "dev", "qa", "marketing", "sales"}
    if agent_key not in valid_agents:
        say(text=f"Unknown agent `{agent_key}`. Valid: {', '.join(sorted(valid_agents))}", thread_ts=thread_ts)
        return

    proj = get_project(thread_ts)
    if not proj:
        say(text="No active project in this thread. Start a project first or use the project thread.", thread_ts=thread_ts)
        return

    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_ts}}

    feedback_entry = {
        "agent": agent_key,
        "message": message,
        "ts": datetime.now(timezone.utc).isoformat(),
    }

    try:
        graph.update_state(config, {"founder_feedback": [feedback_entry]})
        persona_name = AGENT_PERSONAS.get(agent_key, {}).get("role", agent_key)
        say(text=f":speech_balloon: Feedback recorded for *{persona_name}*. They'll see it in their next run.", thread_ts=thread_ts)
    except Exception as e:
        logger.error(f"Feedback update error: {e}")
        say(text=f"Failed to record feedback: {e}", thread_ts=thread_ts)


def _handle_pause(say, thread_ts: str, project_name: Optional[str]):
    """Pause a project by name or the current thread's project."""
    proj = None
    if project_name:
        proj = get_project_by_name(project_name)
    else:
        proj = get_project(thread_ts)

    if not proj:
        say(text="No matching active project found.", thread_ts=thread_ts)
        return

    pid = proj["project_id"]
    update_project(pid, paused=True)

    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": pid}}
    try:
        graph.update_state(config, {"paused": True, "status_summary": "Paused by founder."})
    except Exception as e:
        logger.warning(f"Could not update graph state for pause: {e}")

    name = proj.get("project_name") or pid[:12]
    say(text=f":double_vertical_bar: Project *{name}* is now paused. Use `resume` to continue.", thread_ts=thread_ts)


def _handle_resume(say, thread_ts: str, project_name: Optional[str]):
    """Resume a paused project."""
    proj = None
    if project_name:
        proj = get_project_by_name(project_name)
    else:
        proj = get_project(thread_ts)

    if not proj:
        say(text="No matching active project found.", thread_ts=thread_ts)
        return

    if not proj.get("paused"):
        say(text="That project isn't paused.", thread_ts=thread_ts)
        return

    pid = proj["project_id"]
    update_project(pid, paused=False)

    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": pid}}
    try:
        graph.update_state(config, {"paused": False, "status_summary": "Resumed by founder."})
    except Exception as e:
        logger.warning(f"Could not update graph state for resume: {e}")

    name = proj.get("project_name") or pid[:12]
    say(text=f":arrow_forward: Project *{name}* resumed.", thread_ts=thread_ts)

    # If the graph is at an interrupt, re-invoke to continue
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next:
            say(text=f"Continuing from `{snapshot.next[0]}`...", thread_ts=thread_ts)
            channel = proj.get("slack_channel", "")
            t = threading.Thread(
                target=_resume_graph,
                args=(graph, config, channel, pid),
                daemon=True,
            )
            t.start()
    except Exception as e:
        logger.error(f"Resume re-invoke error: {e}")


def _handle_approve(say, channel: str, thread_ts: str):
    """Approve the current review gate (PRD, code, or marketing review)."""
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_ts}}

    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        say(text=f"Could not load project state: {e}", thread_ts=thread_ts)
        return

    if not snapshot or not snapshot.next:
        say(text="No pending review in this thread.", thread_ts=thread_ts)
        return

    next_node = snapshot.next[0]
    if next_node not in _REVIEW_GATES:
        say(text=f"This thread is waiting at `{next_node}`, not a review gate.", thread_ts=thread_ts)
        return

    current_review = snapshot.values.get("pending_review") or {}
    current_review["approved"] = True

    try:
        graph.update_state(config, {
            "pending_review": current_review,
            "review_rejection_reason": "",
        })
    except Exception as e:
        say(text=f"Failed to update state: {e}", thread_ts=thread_ts)
        return

    gate_label = _GATE_LABELS.get(next_node, ("Review", "Agent"))[0]
    say(text=f":white_check_mark: *{gate_label} approved!* Continuing the workflow...", thread_ts=thread_ts)

    t = threading.Thread(
        target=_resume_graph,
        args=(graph, config, channel, thread_ts),
        daemon=True,
    )
    t.start()


def _handle_reject(say, channel: str, thread_ts: str, reason: Optional[str]):
    """Reject the current review gate and send back for rework."""
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_ts}}

    try:
        snapshot = graph.get_state(config)
    except Exception as e:
        say(text=f"Could not load project state: {e}", thread_ts=thread_ts)
        return

    if not snapshot or not snapshot.next:
        say(text="No pending review in this thread.", thread_ts=thread_ts)
        return

    next_node = snapshot.next[0]
    if next_node not in _REVIEW_GATES:
        say(text=f"This thread is waiting at `{next_node}`, not a review gate.", thread_ts=thread_ts)
        return

    current_review = snapshot.values.get("pending_review") or {}
    current_review["approved"] = False
    rejection_reason = reason or "No reason given."

    try:
        graph.update_state(config, {
            "pending_review": current_review,
            "review_rejection_reason": rejection_reason,
        })
    except Exception as e:
        say(text=f"Failed to update state: {e}", thread_ts=thread_ts)
        return

    gate_label, agent_name = _GATE_LABELS.get(next_node, ("Review", "Agent"))
    say(
        text=f":x: *{gate_label} rejected.* Sending back to {agent_name} for rework.\n> Reason: {rejection_reason}",
        thread_ts=thread_ts,
    )

    t = threading.Thread(
        target=_resume_graph,
        args=(graph, config, channel, thread_ts),
        daemon=True,
    )
    t.start()


def _resume_graph(graph, config: dict, channel: str, thread_ts: str):
    """Resume a graph from an interrupt point. Runs in a daemon thread."""
    try:
        graph.invoke(None, config=config)
    except Exception as e:
        logger.error(f"Graph resume error: {e}")
        try:
            slack_reply_thread.invoke({
                "channel": channel,
                "thread_ts": thread_ts,
                "message": f"Error resuming workflow: {str(e)[:200]}",
                "agent_name": "System",
            })
        except Exception:
            pass


# ── Command dispatcher ───────────────────────────────────────────────────────

def _dispatch_command(cmd_name: str, groups: tuple, say, channel: str, thread_ts: str):
    """Central dispatcher for all founder commands."""
    if cmd_name == "status":
        _handle_status(say, thread_ts, groups[0])
    elif cmd_name == "products":
        _handle_products(say, thread_ts)
    elif cmd_name == "demo":
        _handle_demo(say, channel, thread_ts, groups[0] if groups else None)
    elif cmd_name == "feedback":
        _handle_feedback(say, thread_ts, groups[0], groups[1])
    elif cmd_name == "pause":
        _handle_pause(say, thread_ts, groups[0])
    elif cmd_name == "resume":
        _handle_resume(say, thread_ts, groups[0])
    elif cmd_name == "approve":
        _handle_approve(say, channel, thread_ts)
    elif cmd_name == "reject":
        _handle_reject(say, channel, thread_ts, groups[0])


# ── Agent task handler ───────────────────────────────────────────────────────

def _handle_agent_task(agent_key: str, user_msg: str, channel: str, thread_ts: str):
    """Route to either a full project graph or a one-off agent response."""
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_ts}}

    # Check if there is already an interrupted graph in this thread
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next:
            next_node = snapshot.next[0]

            # Handle wait_for_founder resume
            if next_node == "wait_for_founder":
                logger.info(f"Resuming project graph in thread {thread_ts}")
                current_history = snapshot.values.get("idea_refinement_history", [])
                new_history = current_history + [f"Founder: {user_msg}"]
                graph.update_state(config, {"idea_refinement_history": new_history})
                try:
                    graph.invoke(None, config=config)
                    return
                except Exception as e:
                    logger.error(f"Project graph resume error: {e}")
                    try:
                        slack_reply_thread.invoke({"channel": channel, "thread_ts": thread_ts, "message": f"Resume error: {e}", "agent_name": "System"})
                    except Exception:
                        pass
                    return

            # If at a review gate, hint to use approve/reject
            if next_node in _REVIEW_GATES:
                gate_label = _GATE_LABELS.get(next_node, ("Review", "Agent"))[0]
                try:
                    slack_reply_thread.invoke({
                        "channel": channel,
                        "thread_ts": thread_ts,
                        "message": f"This project is waiting for your *{gate_label} review*. Reply `approve` or `reject <reason>`.",
                        "agent_name": "System",
                    })
                except Exception:
                    pass
                return

    except Exception as e:
        logger.error(f"Error checking project state: {e}")

    if agent_key == "ceo" and _is_new_project(user_msg):
        project_id = thread_ts

        initial_state = make_initial_state(
            project_id=project_id,
            idea=user_msg,
            slack_channel=channel,
            slack_thread_ts=thread_ts,
        )

        register_project(
            thread_ts=thread_ts,
            project_name="",
            slack_channel=channel,
            idea=user_msg,
        )

        try:
            logger.info(f"Starting project graph: {project_id}")
            graph.invoke(initial_state, config=config)
            logger.info(f"Project graph complete: {project_id}")
        except Exception as e:
            logger.error(f"Project graph error: {e}")
            try:
                slack_reply_thread.invoke({
                    "channel": channel, "thread_ts": thread_ts,
                    "message": f"Project error: {str(e)[:200]}", "agent_name": "System",
                })
            except Exception:
                pass
    else:
        # One-off agent response (not a full project)
        persona = AGENT_PERSONAS.get(agent_key, AGENT_PERSONAS["ceo"])
        llm = ChatGoogleGenerativeAI(
            model=gemini_cfg.model,
            google_api_key=gemini_cfg.api_key,
            temperature=0.7,
            include_thoughts=True,
        )
        tools = SLACK_TOOLS + NOTION_TOOLS

        # Ground the agent with real project data to prevent hallucinations
        projects_context = get_projects_context_summary()

        system_prompt = (
            f"You are the {persona['role']}. {persona['backstory']}\n\n"
            f"CRITICAL RULES — read before every response:\n"
            f"1. NEVER invent, fabricate, or hallucinate projects, products, metrics, "
            f"or data that do not exist in the real project registry below.\n"
            f"2. If no projects exist, say so honestly. Do NOT make up examples.\n"
            f"3. Only reference projects, repos, and documents that appear in the registry.\n"
            f"4. Your backstory is fictional context for your persona — do NOT treat it as "
            f"real company history. The ONLY real projects are listed below.\n"
            f"5. When creating Notion pages, NEVER fabricate product names or portfolio data. "
            f"Only document real projects from the registry.\n\n"
            f"--- REAL PROJECT REGISTRY (source of truth) ---\n"
            f"{projects_context}\n"
            f"--- END REGISTRY ---"
        )
        prompt = (
            f"You received this message from the founder: \"{user_msg}\"\n"
            f"Respond as the {persona['role']}. Take action if needed. "
            f"Post your response to Slack channel {channel} in thread {thread_ts}.\n"
            f"Base your answer ONLY on real data from the project registry above. "
            f"If you don't have the information, say so — never guess or fabricate."
        )

        agent = create_react_agent(llm, tools, prompt=system_prompt)
        try:
            agent.invoke({"messages": [("user", prompt)]})
        except Exception as e:
            logger.error(f"Agent task error ({agent_key}): {e}")
            try:
                slack_reply_thread.invoke({
                    "channel": channel, "thread_ts": thread_ts,
                    "message": f"Error: {str(e)[:200]}", "agent_name": persona["role"],
                })
            except Exception:
                pass


# ── Slack bot class ──────────────────────────────────────────────────────────

class SlackBot:
    """Manages the Slack bot lifecycle — listening, routing, responding."""

    def __init__(self):
        self.app = App(token=slack_cfg.bot_token, signing_secret=slack_cfg.signing_secret)
        self._register_handlers()
        self._handler: Optional[SocketModeHandler] = None

    def _register_handlers(self):
        """Set up Slack event listeners."""

        @self.app.event("app_mention")
        def handle_mention(event, say):
            """Respond when the bot is @mentioned in a channel."""
            text = event.get("text", "")
            channel = event.get("channel", "")
            thread_ts = event.get("thread_ts") or event.get("ts", "")

            clean_text = re.sub(r"<@[A-Z0-9]+>", "", text).strip()

            if not clean_text:
                say(
                    text="Hey! I'm your AI startup team. Mention me with a request and "
                         "I'll route it to the right team member.\n\n"
                         "*Commands:* `status`, `products`, `demo`, "
                         "`feedback <agent>: <msg>`, `pause`, `resume`, "
                         "`approve`, `reject <reason>`",
                    thread_ts=thread_ts,
                )
                return

            cmd = _parse_founder_command(clean_text)
            if cmd:
                _dispatch_command(cmd[0], cmd[1], say, channel, thread_ts)
                return

            agent_key = _route_message(clean_text)
            persona = AGENT_PERSONAS[agent_key]

            say(
                text=f"Routing to *{persona['role']}*... one moment.",
                thread_ts=thread_ts,
            )

            t = threading.Thread(
                target=_handle_agent_task,
                args=(agent_key, clean_text, channel, thread_ts),
                daemon=True,
            )
            t.start()

        @self.app.event("message")
        def handle_dm(event, say):
            """Handle direct messages to the bot."""
            if event.get("channel_type") != "im":
                return
            if event.get("bot_id"):
                return

            text = event.get("text", "")
            channel = event.get("channel", "")
            thread_ts = event.get("ts", "")

            if not text.strip():
                return

            cmd = _parse_founder_command(text.strip())
            if cmd:
                _dispatch_command(cmd[0], cmd[1], say, channel, thread_ts)
                return

            agent_key = _route_message(text)
            persona = AGENT_PERSONAS[agent_key]

            say(
                text=f"Routing to *{persona['role']}*...",
                thread_ts=thread_ts,
            )

            t = threading.Thread(
                target=_handle_agent_task,
                args=(agent_key, text, channel, thread_ts),
                daemon=True,
            )
            t.start()

        @self.app.command("/team")
        def handle_team_command(ack, respond, command):
            """Handle /team slash command for direct agent requests."""
            ack()
            text = command.get("text", "").strip()
            channel = command.get("channel_id", "")

            if not text:
                respond(
                    "Usage: `/team <message>` -- Route a request to the right team member.\n"
                    "Examples:\n"
                    "- `/team What's our burn rate?` -> CFO\n"
                    "- `/team Write a PRD for the new feature` -> Product\n"
                    "- `/team Set up the CI/CD pipeline` -> Dev\n"
                    "- `/team Create a launch plan` -> Marketing\n\n"
                    "Founder commands:\n"
                    "- `/team status` — All active projects\n"
                    "- `/team products` — All products (active + completed)\n"
                    "- `/team demo <project>` — Run a demo\n"
                    "- `/team feedback dev: use FastAPI not Flask` — Give feedback\n"
                    "- `/team pause <project>` / `resume <project>`"
                )
                return

            cmd = _parse_founder_command(text)
            if cmd:
                cmd_name, groups = cmd
                if cmd_name in ("status", "products"):
                    # These work without thread context
                    if cmd_name == "status":
                        projects = get_all_projects()
                        if not projects:
                            respond("No active projects right now.")
                        else:
                            lines = [f"*Active Projects ({len(projects)}):*\n"]
                            for p in projects:
                                p = _enrich_project_from_checkpointer(p)
                                paused = " :double_vertical_bar: PAUSED" if p.get("paused") else ""
                                name = p.get("project_name") or p["project_id"][:12]
                                github = f" · <{p['github_repo_url']}|GitHub>" if p.get("github_repo_url") else ""
                                prd = f" · <{p['notion_prd_url']}|PRD>" if p.get("notion_prd_url") else ""
                                lines.append(f"- *{name}*{paused} — `{p.get('current_node', '?')}` — {p.get('status_summary', '')}{github}{prd}")
                            respond("\n".join(lines))
                    elif cmd_name == "products":
                        all_prods = get_all_products()
                        if not all_prods:
                            respond("No products yet.")
                        else:
                            lines = [f"*All Products ({len(all_prods)}):*\n"]
                            for p in all_prods:
                                p = _enrich_project_from_checkpointer(p)
                                name = p.get("project_name") or p["project_id"][:12]
                                done = p.get("current_node") == "done"
                                icon = ":white_check_mark:" if done else ":arrows_counterclockwise:"
                                links = []
                                if p.get("github_repo_url"):
                                    links.append(f"<{p['github_repo_url']}|GitHub>")
                                if p.get("notion_prd_url"):
                                    links.append(f"<{p['notion_prd_url']}|PRD>")
                                link_text = f" · {' · '.join(links)}" if links else ""
                                lines.append(f"{icon} *{name}* — {p.get('idea', '')[:80]}{link_text}")
                            respond("\n".join(lines))
                    return

            agent_key = _route_message(text)
            persona = AGENT_PERSONAS[agent_key]
            respond(f"Routing to *{persona['role']}*... I'll post the response here shortly.")

            t = threading.Thread(
                target=_handle_agent_task,
                args=(agent_key, text, channel, ""),
                daemon=True,
            )
            t.start()

    def start(self):
        """Start listening for Slack events via Socket Mode."""
        logger.info("Starting Slack bot (Socket Mode)...")
        self._handler = SocketModeHandler(self.app, slack_cfg.app_token)
        self._handler.start()

    def start_async(self):
        """Start the bot in a background thread."""
        t = threading.Thread(target=self.start, daemon=True)
        t.start()
        logger.info("Slack bot started in background thread")

    def stop(self):
        if self._handler:
            self._handler.close()
