"""
tools/notion_tools.py — LangChain-compatible tools for Notion workspace management.

Agents use these to create pages, update tasks, query databases,
and organise their work inside Notion.
"""

from __future__ import annotations

import json
import logging
from typing import Optional

from langchain_core.tools import tool
from notion_client import Client as NotionClient

from config.settings import notion_cfg
from tools.retry import retry

logger = logging.getLogger(__name__)

# ── Shared Notion client ─────────────────────────────────────────────────────

_client: Optional[NotionClient] = None


def _get_client() -> NotionClient:
    global _client
    if _client is None:
        _client = NotionClient(auth=notion_cfg.api_key)
    return _client


@retry(max_retries=3, base_delay=1.0)
def _create_page(**kwargs):
    return _get_client().pages.create(**kwargs)


@retry(max_retries=3, base_delay=1.0)
def _append_blocks(**kwargs):
    return _get_client().blocks.children.append(**kwargs)


@retry(max_retries=3, base_delay=1.0)
def _create_database(**kwargs):
    return _get_client().databases.create(**kwargs)


@retry(max_retries=3, base_delay=1.0)
def _query_database(**kwargs):
    return _get_client().databases.query(**kwargs)


# ── Helpers ──────────────────────────────────────────────────────────────────

def _rich_text(text: str) -> list:
    """Build a Notion rich_text block."""
    return [{"type": "text", "text": {"content": text}}]


def _title_prop(text: str) -> dict:
    return {"title": _rich_text(text)}


# Pre-defined schemas for common startup databases
_DB_SCHEMAS = {
    "tasks": {
        "Name": {"title": {}},
        "Status": {
            "select": {
                "options": [
                    {"name": "To Do", "color": "gray"},
                    {"name": "In Progress", "color": "blue"},
                    {"name": "In Review", "color": "yellow"},
                    {"name": "Done", "color": "green"},
                    {"name": "Blocked", "color": "red"},
                ]
            }
        },
        "Priority": {
            "select": {
                "options": [
                    {"name": "Critical", "color": "red"},
                    {"name": "High", "color": "orange"},
                    {"name": "Medium", "color": "yellow"},
                    {"name": "Low", "color": "gray"},
                ]
            }
        },
        "Assignee": {
            "select": {
                "options": [
                    {"name": "CEO", "color": "purple"},
                    {"name": "CFO", "color": "green"},
                    {"name": "Product", "color": "blue"},
                    {"name": "Dev", "color": "orange"},
                    {"name": "QA", "color": "yellow"},
                    {"name": "Marketing", "color": "pink"},
                    {"name": "Sales", "color": "red"},
                ]
            }
        },
        "Due Date": {"date": {}},
        "Description": {"rich_text": {}},
    },
    "sprints": {
        "Sprint": {"title": {}},
        "Status": {
            "select": {
                "options": [
                    {"name": "Planning", "color": "gray"},
                    {"name": "Active", "color": "blue"},
                    {"name": "Completed", "color": "green"},
                ]
            }
        },
        "Start Date": {"date": {}},
        "End Date": {"date": {}},
        "Goals": {"rich_text": {}},
    },
    "leads": {
        "Company": {"title": {}},
        "Contact": {"rich_text": {}},
        "Stage": {
            "select": {
                "options": [
                    {"name": "Prospect", "color": "gray"},
                    {"name": "Contacted", "color": "blue"},
                    {"name": "Meeting Scheduled", "color": "yellow"},
                    {"name": "Proposal Sent", "color": "orange"},
                    {"name": "Negotiation", "color": "purple"},
                    {"name": "Closed Won", "color": "green"},
                    {"name": "Closed Lost", "color": "red"},
                ]
            }
        },
        "Deal Size": {"number": {"format": "dollar"}},
        "Notes": {"rich_text": {}},
    },
    "bugs": {
        "Bug": {"title": {}},
        "Severity": {
            "select": {
                "options": [
                    {"name": "Critical", "color": "red"},
                    {"name": "Major", "color": "orange"},
                    {"name": "Minor", "color": "yellow"},
                    {"name": "Cosmetic", "color": "gray"},
                ]
            }
        },
        "Status": {
            "select": {
                "options": [
                    {"name": "Open", "color": "red"},
                    {"name": "Investigating", "color": "blue"},
                    {"name": "Fix In Progress", "color": "yellow"},
                    {"name": "Fixed", "color": "green"},
                ]
            }
        },
        "Reported By": {"rich_text": {}},
        "Steps to Reproduce": {"rich_text": {}},
    },
}


# ── Tool: Create a page under the root workspace ─────────────────────────────

@tool
def notion_create_page(title: str, content: str = "", parent_page_id: str = "") -> str:
    """Create a new Notion page. Use this to document decisions, write reports, store meeting notes, or create any structured document.
    Args: title (page title), content (markdown-like body text), parent_page_id (optional, defaults to root workspace page).
    Returns: JSON with page_id and url."""
    try:
        parent_id = parent_page_id or notion_cfg.root_page_id
        children = []
        if content:
            for paragraph in content.split("\n\n"):
                paragraph = paragraph.strip()
                if not paragraph:
                    continue
                if paragraph.startswith("# "):
                    children.append({
                        "object": "block",
                        "type": "heading_1",
                        "heading_1": {"rich_text": _rich_text(paragraph[2:])},
                    })
                elif paragraph.startswith("## "):
                    children.append({
                        "object": "block",
                        "type": "heading_2",
                        "heading_2": {"rich_text": _rich_text(paragraph[3:])},
                    })
                elif paragraph.startswith("### "):
                    children.append({
                        "object": "block",
                        "type": "heading_3",
                        "heading_3": {"rich_text": _rich_text(paragraph[4:])},
                    })
                elif paragraph.startswith("- ") or paragraph.startswith("* "):
                    for line in paragraph.split("\n"):
                        line = line.lstrip("- *").strip()
                        if line:
                            children.append({
                                "object": "block",
                                "type": "bulleted_list_item",
                                "bulleted_list_item": {"rich_text": _rich_text(line)},
                            })
                else:
                    for i in range(0, len(paragraph), 1900):
                        children.append({
                            "object": "block",
                            "type": "paragraph",
                            "paragraph": {"rich_text": _rich_text(paragraph[i:i + 1900])},
                        })

        page = _create_page(
            parent={"page_id": parent_id},
            properties=_title_prop(title),
            children=children or None,
        )
        url = page.get("url", "")
        logger.info(f"Notion page created: {title} -> {url}")
        return json.dumps({"ok": True, "page_id": page["id"], "url": url})
    except Exception as e:
        logger.error(f"Notion create page error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


# ── Tool: Update a page (append content) ─────────────────────────────────────

@tool
def notion_update_page(page_id: str, content: str) -> str:
    """Append content to an existing Notion page. Use this to add updates, progress notes, or new sections.
    Args: page_id (ID of the Notion page), content (text to append)."""
    try:
        blocks = []
        for paragraph in content.split("\n\n"):
            paragraph = paragraph.strip()
            if paragraph:
                blocks.append({
                    "object": "block",
                    "type": "paragraph",
                    "paragraph": {"rich_text": _rich_text(paragraph[:1900])},
                })
        _append_blocks(block_id=page_id, children=blocks)
        logger.info(f"Notion page updated: {page_id}")
        return json.dumps({"ok": True, "page_id": page_id})
    except Exception as e:
        logger.error(f"Notion update error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


# ── Tool: Create a database (for tasks, sprints, etc.) ───────────────────────

@tool
def notion_create_database(title: str, db_type: str = "tasks", parent_page_id: str = "") -> str:
    """Create a structured Notion database for tracking tasks, sprints, leads, or bugs.
    Args: title (database title), db_type (one of 'tasks', 'sprints', 'leads', 'bugs'), parent_page_id (optional, defaults to root)."""
    try:
        parent_id = parent_page_id or notion_cfg.root_page_id
        schema = _DB_SCHEMAS.get(db_type, _DB_SCHEMAS["tasks"])

        db = _create_database(
            parent={"page_id": parent_id},
            title=_rich_text(title),
            properties=schema,
        )
        logger.info(f"Notion database created: {title} (type={db_type})")
        return json.dumps({
            "ok": True,
            "database_id": db["id"],
            "url": db.get("url", ""),
            "type": db_type,
        })
    except Exception as e:
        logger.error(f"Notion create database error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


# ── Tool: Add a task/row to a database ────────────────────────────────────────

@tool
def notion_add_task(database_id: str, name: str, status: str = "To Do",
                    priority: str = "Medium", assignee: str = "",
                    description: str = "", due_date: str = "") -> str:
    """Add a new task (row) to a Notion database. Great for creating action items, tracking bugs, or adding leads.
    Args: database_id (Notion database ID), name (task name), status (default 'To Do'), priority (default 'Medium'), assignee (role name), description, due_date (YYYY-MM-DD)."""
    try:
        properties: dict = {
            "Name": {"title": _rich_text(name)},
            "Status": {"select": {"name": status}},
        }
        if priority:
            properties["Priority"] = {"select": {"name": priority}}
        if assignee:
            properties["Assignee"] = {"select": {"name": assignee}}
        if description:
            properties["Description"] = {"rich_text": _rich_text(description)}
        if due_date:
            properties["Due Date"] = {"date": {"start": due_date}}

        page = _create_page(
            parent={"database_id": database_id},
            properties=properties,
        )
        logger.info(f"Notion task added: {name}")
        return json.dumps({"ok": True, "page_id": page["id"]})
    except Exception as e:
        logger.error(f"Notion add task error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


# ── Tool: Query a database ───────────────────────────────────────────────────

@tool
def notion_query_database(database_id: str, filter_status: str = "",
                          filter_assignee: str = "", limit: int = 20) -> str:
    """Query a Notion database with optional filters. Use this to check task status, find assigned work, or review pipeline.
    Args: database_id (Notion database ID), filter_status (e.g. 'In Progress'), filter_assignee (e.g. 'Dev'), limit (max results, default 20)."""
    try:
        filters = []
        if filter_status:
            filters.append({
                "property": "Status",
                "select": {"equals": filter_status},
            })
        if filter_assignee:
            filters.append({
                "property": "Assignee",
                "select": {"equals": filter_assignee},
            })

        query_params = {"database_id": database_id, "page_size": min(limit, 100)}
        if len(filters) == 1:
            query_params["filter"] = filters[0]
        elif len(filters) > 1:
            query_params["filter"] = {"and": filters}

        resp = _query_database(**query_params)
        results = []
        for page in resp.get("results", []):
            props = page.get("properties", {})
            item = {"id": page["id"]}
            for key, val in props.items():
                if val["type"] == "title":
                    texts = val.get("title", [])
                    item[key] = texts[0]["plain_text"] if texts else ""
                elif val["type"] == "select":
                    sel = val.get("select")
                    item[key] = sel["name"] if sel else ""
                elif val["type"] == "rich_text":
                    texts = val.get("rich_text", [])
                    item[key] = texts[0]["plain_text"] if texts else ""
                elif val["type"] == "date":
                    d = val.get("date")
                    item[key] = d["start"] if d else ""
                elif val["type"] == "number":
                    item[key] = val.get("number", 0)
            results.append(item)

        return json.dumps({"ok": True, "count": len(results), "results": results})
    except Exception as e:
        logger.error(f"Notion query error: {e}")
        return json.dumps({"ok": False, "error": str(e)})


# ── Convenience list for imports ─────────────────────────────────────────────

NOTION_TOOLS = [notion_create_page, notion_update_page, notion_create_database,
                notion_add_task, notion_query_database]
