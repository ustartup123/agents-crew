#!/usr/bin/env python3
"""
main.py — Entry point for the AI Startup Team.

Modes:
  python main.py                     → Start full system (Slack bot + scheduler)
  python main.py --kickoff           → Run the idea-to-plan kickoff workflow
  python main.py --standup           → Run a one-off daily standup
  python main.py --review            → Run a one-off weekly review
  python main.py --slack-only        → Only start the Slack bot (no scheduler)
  python main.py --schedule-only     → Only start the scheduler (no Slack bot)
"""

import argparse
import logging
import signal
import sys
import time
import uuid

from rich.console import Console
from rich.panel import Panel
from rich.table import Table

from config.settings import gemini_cfg, slack_cfg, notion_cfg, e2b_cfg, github_cfg, app_cfg
from agents.definitions import AGENT_PERSONAS
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
from graph.standup_graph import run_daily_standup, run_weekly_review
from workflows.slack_bot import SlackBot
from workflows.scheduler import TeamScheduler

console = Console()

# ── Logging ──────────────────────────────────────────────────────────────────

logging.basicConfig(
    level=getattr(logging, app_cfg.log_level, logging.INFO),
    format="%(asctime)s | %(name)-25s | %(levelname)-7s | %(message)s",
    datefmt="%H:%M:%S",
)
logger = logging.getLogger("main")


# ── Validation ───────────────────────────────────────────────────────────────

def _validate_env():
    """Check all required environment variables are set."""
    errors = []
    if not gemini_cfg.api_key:
        errors.append("GEMINI_API_KEY")
    if not slack_cfg.bot_token:
        errors.append("SLACK_BOT_TOKEN")
    if not slack_cfg.app_token:
        errors.append("SLACK_APP_TOKEN")
    if not notion_cfg.api_key:
        errors.append("NOTION_API_KEY")
    if not notion_cfg.root_page_id:
        errors.append("NOTION_ROOT_PAGE_ID")
    if not e2b_cfg.api_key:
        errors.append("E2B_API_KEY")
    if not github_cfg.token:
        errors.append("GITHUB_TOKEN")
    if errors:
        console.print(f"[bold red]Missing environment variables:[/] {', '.join(errors)}")
        console.print("Copy .env.example to .env and fill in all values.")
        sys.exit(1)


def _print_banner():
    """Print a nice startup banner."""
    console.print()
    console.print(Panel.fit(
        f"[bold cyan]{app_cfg.startup_name}[/bold cyan] -- AI Startup Team\n"
        f"[dim]Powered by Gemini ({gemini_cfg.model}) + LangGraph[/dim]",
        border_style="cyan",
    ))

    table = Table(title="Team Roster", show_header=True, header_style="bold magenta")
    table.add_column("Role", style="cyan", width=30)
    table.add_column("Status", justify="center", width=12)

    for key, persona in AGENT_PERSONAS.items():
        table.add_row(persona["role"], "[green]Ready[/green]")

    console.print(table)
    console.print()


# ── Main ─────────────────────────────────────────────────────────────────────

def main():
    parser = argparse.ArgumentParser(
        description="AI Startup Team -- Your full AI-powered startup crew",
        formatter_class=argparse.RawDescriptionHelpFormatter,
    )
    parser.add_argument("--kickoff", action="store_true", help="Run idea-to-plan kickoff")
    parser.add_argument("--idea", type=str, default="", help="Custom startup idea (with --kickoff)")
    parser.add_argument("--standup", action="store_true", help="Run daily standup now")
    parser.add_argument("--review", action="store_true", help="Run weekly review now")
    parser.add_argument("--slack-only", action="store_true", help="Start Slack bot only")
    parser.add_argument("--schedule-only", action="store_true", help="Start scheduler only")
    args = parser.parse_args()

    _validate_env()
    _print_banner()

    # ── One-off workflows ──────────────────────────────────────────────────
    if args.kickoff:
        idea = args.idea or app_cfg.startup_idea
        if not idea:
            console.print("[bold red]No idea provided.[/] Use --idea or set STARTUP_IDEA in .env")
            sys.exit(1)

        project_id = f"project-{uuid.uuid4().hex[:8]}"
        console.print(f"[bold green]Running Idea Kickoff:[/] {idea[:80]}...")
        console.print(f"[dim]Project ID: {project_id}[/dim]")

        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)
        config = {"configurable": {"thread_id": project_id}}
        initial_state = {
            "project_id": project_id,
            "project_name": "",
            "idea": idea,
            "phase": "routing",
            "agents_needed": [],
            "messages": [],
            "slack_channel": slack_cfg.channel_general,
            "slack_thread_ts": "",
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

        result = graph.invoke(initial_state, config=config)
        console.print("[bold green]Kickoff complete![/]")
        console.print(f"Phase: {result.get('phase', 'unknown')}")
        console.print(f"GitHub: {result.get('github_repo_url', 'N/A')}")
        return

    if args.standup:
        console.print("[bold green]Running Daily Standup...[/]")
        run_daily_standup()
        console.print("[bold green]Standup complete![/]")
        return

    if args.review:
        console.print("[bold green]Running Weekly Review...[/]")
        run_weekly_review()
        console.print("[bold green]Review complete![/]")
        return

    # ── Persistent mode: Slack bot + scheduler ─────────────────────────────
    slack_bot = None
    scheduler = None

    def shutdown(sig, frame):
        console.print("\n[bold yellow]Shutting down gracefully...[/]")
        if slack_bot:
            slack_bot.stop()
        if scheduler:
            scheduler.stop()
        sys.exit(0)

    signal.signal(signal.SIGINT, shutdown)
    signal.signal(signal.SIGTERM, shutdown)

    if not args.schedule_only:
        console.print("[bold]Starting Slack bot...[/]")
        slack_bot = SlackBot()
        slack_bot.start_async()
        console.print("[green]Slack bot is live![/green]")

    if not args.slack_only:
        console.print("[bold]Starting scheduler...[/]")
        scheduler = TeamScheduler()
        scheduler.start()

        # Show scheduled jobs
        jobs_table = Table(title="Scheduled Jobs", show_header=True, header_style="bold blue")
        jobs_table.add_column("Job", width=30)
        jobs_table.add_column("Next Run", width=25)
        jobs_table.add_column("Schedule", width=30)

        for job in scheduler.list_jobs():
            jobs_table.add_row(job["name"], job["next_run"], job["trigger"])

        console.print(jobs_table)

    console.print("\n[bold green]System is running![/] Press Ctrl+C to stop.\n")

    # Keep main thread alive
    try:
        while True:
            time.sleep(1)
    except KeyboardInterrupt:
        shutdown(None, None)


if __name__ == "__main__":
    main()
