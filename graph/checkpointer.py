"""
graph/checkpointer.py — SQLite-based state persistence for all projects.

Every project is stored by thread_id. To resume a past project:
  graph.invoke({"idea": "add feature X"}, config={"configurable": {"thread_id": "project-saas-abc"}})
"""

import os
import sqlite3
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "projects.db")


def get_checkpointer() -> SqliteSaver:
    """Return a SQLite checkpointer. Creates the data/ directory and DB if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    conn = sqlite3.connect(DB_PATH, check_same_thread=False)
    return SqliteSaver(conn)
