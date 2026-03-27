"""
graph/project_registry.py — In-memory registry of all projects.

Tracks project metadata so the founder can query status, pause/resume,
list all products, and the Slack bot can look up projects by thread or name.

On restart, call rebuild_from_checkpointer() to recover projects.
"""

from __future__ import annotations

import logging
import threading
from datetime import datetime, timezone
from typing import Optional

logger = logging.getLogger(__name__)

_lock = threading.Lock()
_active_projects: dict[str, dict] = {}  # keyed by thread_ts (= project_id)
_completed_projects: dict[str, dict] = {}  # completed projects, keyed by thread_ts


def register_project(
    thread_ts: str,
    project_name: str = "",
    slack_channel: str = "",
    idea: str = "",
) -> dict:
    """Register a new project when a graph is kicked off."""
    entry = {
        "project_id": thread_ts,
        "project_name": project_name,
        "slack_channel": slack_channel,
        "slack_thread_ts": thread_ts,
        "idea": idea,
        "current_node": "ceo_router",
        "status_summary": "Project started, CEO evaluating idea.",
        "paused": False,
        "phase": "routing",
        "github_repo_url": "",
        "notion_prd_url": "",
        "notion_gtm_url": "",
        "started_at": datetime.now(timezone.utc).isoformat(),
        "completed_at": "",
    }
    with _lock:
        _active_projects[thread_ts] = entry
    logger.info(f"Registered project {thread_ts}")
    return entry


def update_project(thread_ts: str, **kwargs) -> None:
    """Update one or more fields on a tracked project."""
    with _lock:
        if thread_ts in _active_projects:
            _active_projects[thread_ts].update(kwargs)
            # Auto-move to completed when done
            if kwargs.get("phase") == "done" or kwargs.get("current_node") == "done":
                proj = _active_projects.pop(thread_ts)
                proj["completed_at"] = datetime.now(timezone.utc).isoformat()
                _completed_projects[thread_ts] = proj


def get_project(thread_ts: str) -> Optional[dict]:
    """Get a project by its thread_ts / project_id (active or completed)."""
    with _lock:
        proj = _active_projects.get(thread_ts) or _completed_projects.get(thread_ts)
        return proj.copy() if proj else None


def get_all_projects() -> list[dict]:
    """Return all active (non-completed) projects."""
    with _lock:
        return list(_active_projects.values())


def get_all_products() -> list[dict]:
    """Return all projects — active and completed — for the products list."""
    with _lock:
        all_projects = list(_active_projects.values()) + list(_completed_projects.values())
    return all_projects


def get_project_by_name(name: str) -> Optional[dict]:
    """Fuzzy-match a project by name (case-insensitive substring). Checks active first, then completed."""
    name_lower = name.lower().strip()
    with _lock:
        for proj in _active_projects.values():
            if name_lower in proj.get("project_name", "").lower():
                return proj.copy()
        for proj in _completed_projects.values():
            if name_lower in proj.get("project_name", "").lower():
                return proj.copy()
    return None


def remove_project(thread_ts: str) -> None:
    """Remove a project from the active registry."""
    with _lock:
        _active_projects.pop(thread_ts, None)


def rebuild_from_checkpointer(graph, checkpointer) -> int:
    """Scan the checkpointer on startup to recover all projects.

    Returns the number of projects recovered.
    """
    count = 0
    try:
        conn = checkpointer.conn
        cursor = conn.execute(
            "SELECT DISTINCT thread_id FROM checkpoints"
        )
        for (thread_id,) in cursor.fetchall():
            config = {"configurable": {"thread_id": thread_id}}
            try:
                snapshot = graph.get_state(config)
                if not snapshot or not snapshot.values:
                    continue
                vals = snapshot.values
                register_project(
                    thread_ts=thread_id,
                    project_name=vals.get("project_name", ""),
                    slack_channel=vals.get("slack_channel", ""),
                    idea=vals.get("idea", ""),
                )
                update_project(
                    thread_id,
                    current_node=vals.get("current_node", ""),
                    status_summary=vals.get("status_summary", ""),
                    paused=vals.get("paused", False),
                    phase=vals.get("phase", ""),
                    github_repo_url=vals.get("github_repo_url", ""),
                    notion_prd_url=vals.get("notion_prd_url", ""),
                    notion_gtm_url=vals.get("notion_gtm_url", ""),
                )
                count += 1
            except Exception as e:
                logger.warning(f"Could not recover project {thread_id}: {e}")
    except Exception as e:
        logger.warning(f"Could not scan checkpointer: {e}")
    logger.info(f"Recovered {count} projects from checkpointer")
    return count
