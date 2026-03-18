#!/usr/bin/env python3
"""
setup_slack_channel.py
───────────────────────────────────────────────────────────────────────────────
One-time setup script that:

  1. Creates the  #ai-team-standup  channel in your Slack workspace
  2. Adds YOU (looked up by your email) to the channel
  3. Joins the bot to the channel
  4. Writes the channel ID into your .env file automatically
  5. Posts a welcome message from the CEO to kick things off

Run once:
    python setup_slack_channel.py

After running, your daily standup will automatically post to #ai-team-standup.
───────────────────────────────────────────────────────────────────────────────
"""

import os
import re
import sys
import time

from dotenv import load_dotenv, set_key
from rich.console import Console
from rich.panel import Panel
from rich.prompt import Prompt
from slack_sdk import WebClient
from slack_sdk.errors import SlackApiError

load_dotenv()
console = Console()

# ── Config ───────────────────────────────────────────────────────────────────

CHANNEL_NAME = "ai-team-standup"
BOT_TOKEN    = os.getenv("SLACK_BOT_TOKEN", "")
ENV_FILE     = os.path.join(os.path.dirname(__file__), ".env")

AGENT_NAMES = [
    "CEO / Chief Executive Officer",
    "CFO / Chief Financial Officer",
    "VP of Product",
    "Lead Software Developer",
    "QA Lead",
    "Head of Marketing",
    "Head of Sales",
]


# ── Helpers ──────────────────────────────────────────────────────────────────

def _check_token():
    if not BOT_TOKEN or BOT_TOKEN.startswith("xoxb-your"):
        console.print("[bold red]Error:[/] SLACK_BOT_TOKEN is not set in your .env file.")
        console.print("Add your bot token and re-run this script.")
        sys.exit(1)


def _find_or_create_channel(client: WebClient) -> tuple[str, bool]:
    """Return (channel_id, was_created)."""
    # Check if channel already exists
    cursor = None
    while True:
        resp = client.conversations_list(
            types="public_channel,private_channel",
            limit=200,
            cursor=cursor,
        )
        for ch in resp.get("channels", []):
            if ch["name"] == CHANNEL_NAME:
                return ch["id"], False
        cursor = resp.get("response_metadata", {}).get("next_cursor")
        if not cursor:
            break

    # Create it
    resp = client.conversations_create(name=CHANNEL_NAME, is_private=False)
    return resp["channel"]["id"], True


def _get_user_id_by_email(client: WebClient, email: str) -> str | None:
    try:
        resp = client.users_lookupByEmail(email=email)
        return resp["user"]["id"]
    except SlackApiError as e:
        if e.response["error"] == "users_not_found":
            return None
        raise


def _invite_user(client: WebClient, channel_id: str, user_id: str):
    try:
        client.conversations_invite(channel=channel_id, users=user_id)
        return True
    except SlackApiError as e:
        if e.response["error"] in ("already_in_channel", "cant_invite_self"):
            return True  # Already there, fine
        raise


def _join_channel(client: WebClient, channel_id: str):
    try:
        client.conversations_join(channel=channel_id)
    except SlackApiError as e:
        if e.response["error"] not in ("already_in_channel", "method_not_supported_for_channel_type"):
            raise


def _write_channel_to_env(channel_id: str):
    """Write SLACK_CHANNEL_STANDUP and update SLACK_CHANNEL_GENERAL in .env."""
    # Write standup channel
    set_key(ENV_FILE, "SLACK_CHANNEL_STANDUP", channel_id)
    # Also set as the general channel if not already configured
    current_general = os.getenv("SLACK_CHANNEL_GENERAL", "")
    if not current_general or current_general == "C0123456789":
        set_key(ENV_FILE, "SLACK_CHANNEL_GENERAL", channel_id)
    console.print(f"[dim]Updated .env: SLACK_CHANNEL_STANDUP={channel_id}[/dim]")


def _post_welcome(client: WebClient, channel_id: str, founder_name: str):
    """Post a structured welcome message + today's agenda from the CEO bot."""
    today = time.strftime("%A, %B %d %Y")
    blocks = [
        {
            "type": "header",
            "text": {
                "type": "plain_text",
                "text": f"👋  Welcome to #ai-team-standup",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    f"*Today is {today}*\n\n"
                    f"Hey {founder_name}! I'm your AI startup team. "
                    f"This channel is our daily home base — every weekday at *9:00 AM* "
                    f"each team member will post their standup update here."
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "*Your AI team is:*",
            },
        },
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": "\n".join(f"• 🤖  {name}" for name in AGENT_NAMES),
            },
        },
        {"type": "divider"},
        {
            "type": "section",
            "text": {
                "type": "mrkdwn",
                "text": (
                    "*How to interact with your team:*\n"
                    "• *@mention the bot* in any channel → routed to the right agent\n"
                    "• *DM the bot* → private conversation with the right agent\n"
                    "• */team <message>* → quick slash command routing\n"
                    "• Run `python main.py --standup` → trigger an immediate standup\n"
                    "• Run `python main.py --kickoff` → full idea-to-plan workflow"
                ),
            },
        },
        {"type": "divider"},
        {
            "type": "context",
            "elements": [
                {
                    "type": "mrkdwn",
                    "text": "Powered by CrewAI + Gemini  |  Managed by Claude",
                }
            ],
        },
    ]

    client.chat_postMessage(
        channel=channel_id,
        text=f"Welcome to #ai-team-standup, {founder_name}!",
        blocks=blocks,
    )


def _post_first_standup(client: WebClient, channel_id: str):
    """Post a sample standup from each agent."""
    today = time.strftime("%A, %B %d")
    client.chat_postMessage(
        channel=channel_id,
        text=f"*Daily Standup — {today}*",
        blocks=[
            {
                "type": "header",
                "text": {"type": "plain_text", "text": f"📋  Daily Standup — {today}"},
            },
            {
                "type": "section",
                "text": {
                    "type": "mrkdwn",
                    "text": (
                        "*[CEO]*\n"
                        "• *Yesterday:* Set team up on Slack + Notion, aligned on startup vision\n"
                        "• *Today:* Kick off first sprint planning, review PRD with Product\n"
                        "• *Blockers:* None — waiting on founder's idea for kickoff\n\n"
                        "*[CFO]*\n"
                        "• *Yesterday:* Bootstrapped financial model template in Notion\n"
                        "• *Today:* Draft initial burn rate + 6-month runway projection\n"
                        "• *Blockers:* Need team size and monthly spend assumptions\n\n"
                        "*[VP of Product]*\n"
                        "• *Yesterday:* Drafted product vision framework\n"
                        "• *Today:* Start backlog setup in Notion, write first user stories\n"
                        "• *Blockers:* None\n\n"
                        "*[Lead Developer]*\n"
                        "• *Yesterday:* Set up project skeleton and reviewed architecture options\n"
                        "• *Today:* Finalize tech stack, scaffold repo structure\n"
                        "• *Blockers:* Need PRD finalized before committing to data model\n\n"
                        "*[QA Lead]*\n"
                        "• *Yesterday:* Created bug tracker database in Notion\n"
                        "• *Today:* Write test strategy doc, define acceptance criteria format\n"
                        "• *Blockers:* None\n\n"
                        "*[Head of Marketing]*\n"
                        "• *Yesterday:* Researched competitor positioning\n"
                        "• *Today:* Draft ICP definition + initial content calendar in Notion\n"
                        "• *Blockers:* Need startup name and tagline from CEO\n\n"
                        "*[Head of Sales]*\n"
                        "• *Yesterday:* Set up leads database in Notion\n"
                        "• *Today:* Build first outreach sequence + cold email template\n"
                        "• *Blockers:* None"
                    ),
                },
            },
        ],
    )


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    console.print()
    console.print(Panel.fit(
        "[bold cyan]Slack Channel Setup[/bold cyan]\n"
        "[dim]Creates #ai-team-standup and adds your whole team[/dim]",
        border_style="cyan",
    ))
    console.print()

    _check_token()
    client = WebClient(token=BOT_TOKEN)

    # ── 1. Verify the bot token works ────────────────────────────────────────
    console.print("[bold]Step 1/5[/bold] Verifying bot token...")
    try:
        auth = client.auth_test()
        bot_name = auth.get("user", "bot")
        workspace = auth.get("team", "your workspace")
        console.print(f"  Connected as [green]{bot_name}[/green] in workspace [cyan]{workspace}[/cyan]")
    except SlackApiError as e:
        console.print(f"[bold red]Token error:[/] {e.response['error']}")
        console.print("Check your SLACK_BOT_TOKEN in .env")
        sys.exit(1)

    # ── 2. Create (or find) the channel ─────────────────────────────────────
    console.print("[bold]Step 2/5[/bold] Creating #ai-team-standup channel...")
    try:
        channel_id, was_created = _find_or_create_channel(client)
        if was_created:
            console.print(f"  [green]Created[/green] #ai-team-standup  (ID: {channel_id})")
        else:
            console.print(f"  [yellow]Already exists[/yellow] #ai-team-standup  (ID: {channel_id})")
    except SlackApiError as e:
        console.print(f"[bold red]Could not create channel:[/] {e.response['error']}")
        console.print("Make sure your bot has the 'channels:manage' or 'groups:write' scope.")
        sys.exit(1)

    # ── 3. Add the bot to the channel ────────────────────────────────────────
    console.print("[bold]Step 3/5[/bold] Adding bot to channel...")
    _join_channel(client, channel_id)
    console.print(f"  [green]Bot joined[/green] #ai-team-standup")

    # ── 4. Add the founder ───────────────────────────────────────────────────
    console.print("[bold]Step 4/5[/bold] Adding you to the channel...")
    founder_email = os.getenv("FOUNDER_EMAIL", "")
    if not founder_email:
        founder_email = Prompt.ask(
            "  Enter your Slack email address",
            default="",
        ).strip()

    founder_name = "Founder"
    if founder_email:
        user_id = _get_user_id_by_email(client, founder_email)
        if user_id:
            # Extract first name for personal greeting
            try:
                uinfo = client.users_info(user=user_id)
                founder_name = uinfo["user"]["profile"].get("first_name") or \
                               uinfo["user"]["real_name"] or "Founder"
            except Exception:
                pass
            _invite_user(client, channel_id, user_id)
            console.print(f"  [green]Added[/green] {founder_name} ({founder_email}) to channel")
        else:
            console.print(f"  [yellow]Could not find[/yellow] user with email {founder_email}")
            console.print("  You can join #ai-team-standup manually in Slack.")
    else:
        console.print("  [yellow]Skipped[/yellow] — join #ai-team-standup manually in Slack.")

    # ── 5. Save channel ID to .env ───────────────────────────────────────────
    console.print("[bold]Step 5/5[/bold] Saving channel ID to .env...")
    _write_channel_to_env(channel_id)
    console.print(f"  [green]Saved[/green] SLACK_CHANNEL_STANDUP={channel_id}")

    # ── Post welcome + first standup ─────────────────────────────────────────
    console.print("\n[bold]Posting welcome message...[/bold]")
    _post_welcome(client, channel_id, founder_name)
    console.print("\n[bold]Posting first standup preview...[/bold]")
    _post_first_standup(client, channel_id)

    # ── Summary ──────────────────────────────────────────────────────────────
    console.print()
    console.print(Panel(
        f"[bold green]Setup complete![/bold green]\n\n"
        f"Channel: [cyan]#ai-team-standup[/cyan]  (ID: {channel_id})\n"
        f"Members: All 7 AI agents + you\n\n"
        f"[bold]Next steps:[/bold]\n"
        f"  • Open Slack — check [cyan]#ai-team-standup[/cyan]\n"
        f"  • Run `python main.py` to start the full system\n"
        f"  • Run `python main.py --standup` to trigger a real standup now\n"
        f"  • The bot will post daily at [bold]9:00 AM[/bold] automatically",
        border_style="green",
    ))


if __name__ == "__main__":
    main()
