"""
tools/slack_tools.py — LangChain-compatible tools for Slack communication.

Each tool wraps the Slack Web API so agents can send messages,
read channels, and reply to threads.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

from config.settings import slack_cfg

logger = logging.getLogger(__name__)

# ── Shared Slack client ──────────────────────────────────────────────────────

_client: Optional[WebClient] = None


def _get_client() -> WebClient:
    global _client
    if _client is None:
        _client = WebClient(token=slack_cfg.bot_token)
    return _client


# ── Tool: Send a message to a channel ────────────────────────────────────────

@tool
def slack_send_message(channel: str, message: str, agent_name: str = "Agent") -> str:
    """Send a message to a Slack channel.
    Args: channel (Slack channel ID e.g. C0123456789), message (text to send, supports mrkdwn), agent_name (name to prefix the message with)."""
    try:
        formatted = f"*[{agent_name}]*\n{message}"
        resp = _get_client().chat_postMessage(channel=channel, text=formatted)
        ts = resp["ts"]
        logger.info(f"Slack message sent to {channel} (ts={ts})")
        return json.dumps({"ok": True, "ts": ts, "channel": channel})
    except SlackApiError as e:
        logger.error(f"Slack send error: {e.response['error']}")
        return json.dumps({"ok": False, "error": str(e.response["error"])})


# ── Tool: Read recent messages from a channel ────────────────────────────────

@tool
def slack_read_channel(channel: str, limit: int = 10) -> str:
    """Read recent messages from a Slack channel. Use this to catch up on team conversations or gather context.
    Args: channel (Slack channel ID), limit (max messages to fetch, default 10, max 50)."""
    try:
        limit = min(limit, 50)
        resp = _get_client().conversations_history(channel=channel, limit=limit)
        messages = []
        for msg in resp.get("messages", []):
            messages.append({
                "user": msg.get("user", "unknown"),
                "text": msg.get("text", ""),
                "ts": msg.get("ts", ""),
                "thread_ts": msg.get("thread_ts"),
            })
        return json.dumps({"ok": True, "messages": messages})
    except SlackApiError as e:
        logger.error(f"Slack read error: {e.response['error']}")
        return json.dumps({"ok": False, "error": str(e.response["error"])})


# ── Tool: Reply to a thread ──────────────────────────────────────────────────

@tool
def slack_reply_thread(channel: str, thread_ts: str, message: str, agent_name: str = "Agent") -> str:
    """Reply to a specific Slack thread. Use this to continue a discussion or respond to a question.
    Args: channel (Slack channel ID), thread_ts (timestamp of the parent message), message (reply text), agent_name (name to prefix the reply with)."""
    try:
        formatted = f"*[{agent_name}]*\n{message}"
        resp = _get_client().chat_postMessage(
            channel=channel,
            text=formatted,
            thread_ts=thread_ts,
        )
        return json.dumps({"ok": True, "ts": resp["ts"]})
    except SlackApiError as e:
        logger.error(f"Slack reply error: {e.response['error']}")
        return json.dumps({"ok": False, "error": str(e.response["error"])})


# ── Convenience list for imports ─────────────────────────────────────────────

SLACK_TOOLS = [slack_send_message, slack_read_channel, slack_reply_thread]
