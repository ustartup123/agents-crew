"""
workflows/scheduler.py — APScheduler-based task scheduler for recurring team rituals.

Schedules:
  - Daily standup (weekdays 9:00 AM)
  - Weekly strategy review (Fridays 4:00 PM)
"""

from __future__ import annotations

import logging

from apscheduler.schedulers.background import BackgroundScheduler
from apscheduler.triggers.cron import CronTrigger

from graph.standup_graph import run_daily_standup, run_weekly_review

logger = logging.getLogger(__name__)


class TeamScheduler:
    """Manages all scheduled workflows for the AI startup team."""

    def __init__(self):
        self.scheduler = BackgroundScheduler()
        self._setup_jobs()

    def _setup_jobs(self):
        """Register all recurring jobs."""

        # Daily standup — weekdays at 9:00 AM
        self.scheduler.add_job(
            func=run_daily_standup,
            trigger=CronTrigger(day_of_week="mon-fri", hour=9, minute=0),
            id="daily_standup",
            name="Daily Standup",
            replace_existing=True,
        )
        logger.info("Scheduled: Daily Standup (Mon-Fri 9:00 AM)")

        # Weekly review — Fridays at 4:00 PM
        self.scheduler.add_job(
            func=run_weekly_review,
            trigger=CronTrigger(day_of_week="fri", hour=16, minute=0),
            id="weekly_review",
            name="Weekly Strategy Review",
            replace_existing=True,
        )
        logger.info("Scheduled: Weekly Review (Fri 4:00 PM)")

    def start(self):
        self.scheduler.start()
        logger.info("TeamScheduler started — all jobs active")

    def stop(self):
        self.scheduler.shutdown(wait=False)
        logger.info("TeamScheduler stopped")

    def list_jobs(self) -> list[dict]:
        """Return a human-readable list of scheduled jobs."""
        jobs = []
        for job in self.scheduler.get_jobs():
            jobs.append({
                "id": job.id,
                "name": job.name,
                "next_run": str(job.next_run_time),
                "trigger": str(job.trigger),
            })
        return jobs

    def run_now(self, job_id: str):
        """Manually trigger a scheduled job immediately."""
        job_map = {
            "daily_standup": run_daily_standup,
            "weekly_review": run_weekly_review,
        }
        fn = job_map.get(job_id)
        if fn:
            logger.info(f"Manual trigger: {job_id}")
            fn()
        else:
            logger.warning(f"Unknown job ID: {job_id}")
