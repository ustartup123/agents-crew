"""
workflows/slack_bot.py — Real-time Slack bot that routes messages to the right agent.

Listens for @mentions and DMs, determines which agent should respond,
and kicks off a LangGraph task to handle the request.
"""

from __future__ import annotations

import logging
import re
import threading
import uuid
from typing import Optional

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from slack_bolt import App
from slack_bolt.adapter.socket_mode import SocketModeHandler

from agents import AGENT_PERSONAS
from config.settings import slack_cfg, gemini_cfg, app_cfg
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
from tools.slack_tools import SLACK_TOOLS, slack_reply_thread
from tools.notion_tools import NOTION_TOOLS

logger = logging.getLogger(__name__)

# ── Agent routing keywords ───────────────────────────────────────────────────

_ROUTE_KEYWORDS = {
    "ceo": ["ceo", "strategy", "vision", "decision", "priority", "roadmap", "okr", "team"],
    "cfo": ["cfo", "finance", "budget", "runway", "burn", "revenue", "pricing", "fundrais", "investor", "cost"],
    "product": ["product", "feature", "prd", "requirement", "backlog", "user story", "spec", "prioriti"],
    "dev": ["dev", "code", "engineer", "architect", "api", "database", "deploy", "ci/cd", "tech", "bug fix", "implement"],
    "qa": ["qa", "test", "quality", "bug", "regression", "edge case", "coverage"],
    "marketing": ["marketing", "brand", "content", "seo", "launch", "campaign", "social media", "growth", "awareness"],
    "sales": ["sales", "lead", "prospect", "deal", "pipeline", "outreach", "pitch", "close", "crm", "customer"],
}

PROJECT_TRIGGER_KEYWORDS = ["build", "create", "make", "develop", "idea", "startup", "launch", "new project"]


def _route_message(text: str) -> str:
    """Determine which agent key should handle a message based on keywords."""
    text_lower = text.lower()

    # Check for explicit agent mentions first
    for key, keywords in _ROUTE_KEYWORDS.items():
        for kw in keywords[:2]:
            if kw in text_lower:
                return key

    # Score-based fallback
    scores = {key: 0 for key in _ROUTE_KEYWORDS}
    for key, keywords in _ROUTE_KEYWORDS.items():
        for kw in keywords:
            if kw in text_lower:
                scores[key] += 1

    best = max(scores, key=scores.get)
    if scores[best] > 0:
        return best

    return "ceo"


def _is_new_project(text: str) -> bool:
    return any(kw in text.lower() for kw in PROJECT_TRIGGER_KEYWORDS)


def _handle_agent_task(agent_key: str, user_msg: str, channel: str, thread_ts: str):
    """Route to either a full project graph or a one-off agent response."""
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": thread_ts}}

    # Check if there is already an interrupted graph in this thread
    try:
        snapshot = graph.get_state(config)
        if snapshot and snapshot.next and "wait_for_founder" in snapshot.next:
            logger.info(f"Resuming project graph in thread {thread_ts}")
            
            # Update history
            current_history = snapshot.values.get("idea_refinement_history", [])
            new_history = current_history + [f"Founder: {user_msg}"]
            graph.update_state(config, {"idea_refinement_history": new_history})
            
            # Resume Graph
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
    except Exception as e:
        logger.error(f"Error checking project state: {e}")

    if agent_key == "ceo" and _is_new_project(user_msg):
        # Kick off a full project
        project_id = thread_ts  # Use thread_ts as the project ID to keep continuity in the same thread
        
        initial_state = {
            "project_id": project_id,
            "project_name": "",
            "idea": user_msg,
            "phase": "routing",
            "agents_needed": [],
            "messages": [],
            "slack_channel": channel,
            "slack_thread_ts": thread_ts,
            "pending_clarification": None,
            "code_iterations": 0,
            "max_code_iterations": 3,
            "qa_approved": False,
            "qa_feedback": "",
            "github_repo_url": "",
            "github_repo_name": "",
            "errors": [],
            "notion_vision_url": "", "notion_prd_url": "", "notion_arch_url": "",
            "notion_test_strategy_url": "", "notion_financial_url": "",
            "notion_gtm_url": "", "notion_sales_url": "", "notion_task_db_id": "",
            "vision_content": "", "prd_content": "", "arch_content": "",
        }

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
        )
        tools = SLACK_TOOLS + NOTION_TOOLS

        system_prompt = f"You are the {persona['role']}. {persona['backstory']}"
        prompt = (
            f"You received this message from the founder: \"{user_msg}\"\n"
            f"Respond as the {persona['role']}. Take action if needed. "
            f"Post your response to Slack channel {channel} in thread {thread_ts}."
        )

        agent = create_react_agent(llm, tools, state_modifier=system_prompt)
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
                         "I'll route it to the right team member (CEO, CFO, Dev, QA, "
                         "Product, Marketing, or Sales).",
                    thread_ts=thread_ts,
                )
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
                    "- `/team Create a launch plan` -> Marketing"
                )
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
