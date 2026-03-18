# Implementation Plan: CrewAI → LangGraph Rewrite

## Goal

Transform this project from a one-shot linear CrewAI pipeline into a persistent, feedback-loop-capable multi-agent system where:
- The founder sends any idea (via Slack or CLI) → CEO routes it → the right agents collaborate to completion
- Agents can ask each other questions mid-task (Dev ↔ Product, QA ↔ Dev)
- The Dev agent writes and executes real code in an E2B sandbox, commits to GitHub
- Every project is persisted by ID — the team can return to any past project and add features or fix bugs
- All inter-agent communication is visible in Slack in real time
- Agents that are not needed for a given project simply skip (CEO decides)
- Daily standup posts a status update per agent to Slack and saves a summary to Notion

---

## Decision: Refactor, Not Rewrite

**Keep as-is (no changes needed):**
- `tools/slack_tools.py` — already works; just change base class (see Step 3)
- `tools/notion_tools.py` — already works; just change base class (see Step 4)
- `workflows/scheduler.py` — APScheduler logic is fine; just rewire imports (see Step 14)
- `setup.sh`, `setup_slack_channel.py`, `.env.example`

**Modify:**
- `config/settings.py` — add E2B and GitHub config blocks
- `tools/slack_tools.py` — swap CrewAI `BaseTool` → LangChain `@tool` decorator
- `tools/notion_tools.py` — same swap
- `workflows/slack_bot.py` — replace CrewAI Crew/Task calls with LangGraph graph invocations
- `workflows/scheduler.py` — update imports, replace `run_daily_standup` etc. with new graph functions
- `main.py` — remove AgentRoster building, initialize LangGraph checkpointer and graphs
- `requirements.txt` — remove crewai, add langgraph, e2b, pygithub, langchain-core

**Delete:**
- `agents/factory.py` — replaced by `agents/definitions.py` + `graph/nodes.py`
- `workflows/scheduled_crews.py` — replaced by `graph/project_graph.py` + `graph/standup_graph.py`

**Create (new files):**
- `agents/definitions.py` — agent personas (role, goal, backstory) extracted from factory.py
- `tools/github_tools.py` — GitHub repo/file/commit/PR tools via PyGitHub
- `tools/code_exec_tools.py` — E2B sandbox tools (run code, run tests, write/read files, install packages)
- `graph/__init__.py`
- `graph/state.py` — `ProjectState` TypedDict (the shared state passed between nodes)
- `graph/checkpointer.py` — SQLite checkpointer setup (persists every project by thread_id)
- `graph/nodes.py` — all 7 agent node functions + CEO router + QA approval check
- `graph/project_graph.py` — the main `StateGraph` with all edges and conditional routing
- `graph/standup_graph.py` — simplified daily standup (posts to Slack, saves to Notion)

---

## New File Structure

```
crewai/
├── agents/
│   └── definitions.py          # NEW: agent personas only (no CrewAI dependency)
├── config/
│   └── settings.py             # MODIFIED: add E2B + GitHub config
├── graph/
│   ├── __init__.py             # NEW
│   ├── state.py                # NEW: ProjectState TypedDict
│   ├── checkpointer.py         # NEW: SQLite checkpointer
│   ├── nodes.py                # NEW: all 7 agent node functions
│   ├── project_graph.py        # NEW: main StateGraph (idea → completion)
│   └── standup_graph.py        # NEW: daily standup graph
├── tools/
│   ├── slack_tools.py          # MODIFIED: BaseTool → @tool
│   ├── notion_tools.py         # MODIFIED: BaseTool → @tool
│   ├── github_tools.py         # NEW
│   └── code_exec_tools.py      # NEW: E2B sandbox tools
├── workflows/
│   ├── slack_bot.py            # MODIFIED: LangGraph instead of CrewAI
│   └── scheduler.py            # MODIFIED: updated imports
├── main.py                     # MODIFIED
├── requirements.txt            # MODIFIED
└── IMPLEMENTATION_PLAN.md      # this file
```

---

## Step-by-Step Implementation

Implement steps in order — each step builds on the previous.

---

### Step 1: Update `requirements.txt`

Replace the entire file with:

```
# ── LangGraph (replaces CrewAI) ─────────────────────────────────────────────
langgraph>=0.2.0,<1.0.0
langgraph-checkpoint-sqlite>=1.0.0,<2.0.0
langchain>=0.3.0,<1.0.0
langchain-core>=0.3.0,<1.0.0
langchain-google-genai>=2.0.0,<3.0.0

# ── Code execution (E2B sandbox) ────────────────────────────────────────────
e2b-code-interpreter>=1.0.0,<2.0.0

# ── GitHub integration ──────────────────────────────────────────────────────
PyGitHub>=2.3.0,<3.0.0

# ── Slack ────────────────────────────────────────────────────────────────────
slack-bolt>=1.20.0
slack-sdk>=3.33.0

# ── Notion ───────────────────────────────────────────────────────────────────
notion-client>=2.2.0

# ── Scheduling ───────────────────────────────────────────────────────────────
apscheduler>=3.10.0,<4.0.0

# ── Utilities ────────────────────────────────────────────────────────────────
python-dotenv>=1.0.0
pydantic>=2.0.0
rich>=13.0.0
```

Run: `.venv/bin/pip install -r requirements.txt`

---

### Step 2: Update `config/settings.py`

Add two new config dataclasses after `NotionConfig`. Keep everything else unchanged.

```python
# ── E2B ───────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class E2BConfig:
    api_key: str = field(default_factory=lambda: _env("E2B_API_KEY"))


# ── GitHub ────────────────────────────────────────────────────────────────────

@dataclass(frozen=True)
class GitHubConfig:
    token: str = field(default_factory=lambda: _env("GITHUB_TOKEN"))
    username: str = field(default_factory=lambda: _env("GITHUB_USERNAME"))
    default_private: bool = True  # all repos private by default


# Add these singleton instances at the bottom:
e2b_cfg = E2BConfig()
github_cfg = GitHubConfig()
```

Also add `e2b_cfg` and `github_cfg` to the `_validate_env()` check in `main.py` (Step 15).

---

### Step 3: Update `tools/slack_tools.py`

The existing tool logic is correct. Only change the base class pattern.

**Change from (CrewAI pattern):**
```python
from crewai.tools import BaseTool
from pydantic import BaseModel, Field

class _SlackSendInput(BaseModel):
    channel: str = Field(...)
    message: str = Field(...)

class SlackSendMessageTool(BaseTool):
    name: str = "slack_send_message"
    description: str = "..."
    args_schema: Type[BaseModel] = _SlackSendInput

    def _run(self, channel: str, message: str, agent_name: str = "") -> str:
        ...
```

**Change to (LangChain @tool pattern):**
```python
from langchain_core.tools import tool

@tool
def slack_send_message(channel: str, message: str, agent_name: str = "") -> str:
    """Send a message to a Slack channel. Args: channel (channel ID), message (text to send), agent_name (optional sender label)."""
    client = _get_client()
    formatted = f"*[{agent_name}]*\n{message}" if agent_name else message
    result = client.chat_postMessage(channel=channel, text=formatted, mrkdwn=True)
    return json.dumps({"ok": result["ok"], "ts": result["ts"], "channel": result["channel"]})

@tool
def slack_read_channel(channel: str, limit: int = 50) -> str:
    """Read recent messages from a Slack channel. Args: channel (channel ID), limit (max messages, default 50)."""
    ...

@tool
def slack_reply_thread(channel: str, thread_ts: str, message: str, agent_name: str = "") -> str:
    """Reply to a Slack thread. Args: channel (channel ID), thread_ts (thread timestamp), message (reply text), agent_name (optional sender label)."""
    ...
```

Export all three as a list for convenience:
```python
SLACK_TOOLS = [slack_send_message, slack_read_channel, slack_reply_thread]
```

Keep the `_get_client()` singleton exactly as-is.

---

### Step 4: Update `tools/notion_tools.py`

Same pattern change as Step 3 — convert all 5 tools from `BaseTool` classes to `@tool` functions.

```python
from langchain_core.tools import tool

@tool
def notion_create_page(title: str, content: str, parent_id: str = "") -> str:
    """Create a new Notion page. Args: title, content (markdown-like text), parent_id (optional, defaults to root)."""
    ...

@tool
def notion_update_page(page_id: str, content: str) -> str:
    """Append content to an existing Notion page. Args: page_id, content."""
    ...

@tool
def notion_create_database(title: str, db_type: str) -> str:
    """Create a Notion database. Args: title, db_type (one of: 'tasks', 'sprints', 'leads', 'bugs')."""
    ...

@tool
def notion_add_task(database_id: str, name: str, status: str = "To Do",
                    priority: str = "Medium", assignee: str = "",
                    description: str = "", due_date: str = "") -> str:
    """Add a row to a Notion database. Args: database_id, name, status, priority, assignee, description, due_date."""
    ...

@tool
def notion_query_database(database_id: str, filter_status: str = "",
                          filter_assignee: str = "") -> str:
    """Query a Notion database with optional filters. Args: database_id, filter_status, filter_assignee."""
    ...

NOTION_TOOLS = [notion_create_page, notion_update_page, notion_create_database,
                notion_add_task, notion_query_database]
```

Keep all internal implementation logic identical. Only change the interface layer.

---

### Step 5: Create `tools/github_tools.py`

New file. All operations use `PyGitHub`.

```python
"""
tools/github_tools.py — GitHub repo management tools for the Dev agent.
"""

import base64
from langchain_core.tools import tool
from github import Github, GithubException
from config.settings import github_cfg

def _get_client() -> Github:
    return Github(github_cfg.token)

@tool
def github_create_repo(name: str, description: str = "") -> str:
    """Create a new private GitHub repository.
    Args: name (repo name, use kebab-case), description (optional).
    Returns: JSON with repo_url, clone_url, repo_name."""
    g = _get_client()
    user = g.get_user()
    repo = user.create_repo(
        name=name,
        description=description,
        private=github_cfg.default_private,
        auto_init=True,  # creates main branch with README
    )
    return json.dumps({
        "repo_url": repo.html_url,
        "clone_url": repo.clone_url,
        "repo_name": repo.full_name,
    })

@tool
def github_create_file(repo_name: str, file_path: str, content: str,
                        commit_message: str = "Add file") -> str:
    """Create or update a file in a GitHub repo.
    Args: repo_name (owner/repo format), file_path (e.g. 'src/main.py'),
    content (file content as string), commit_message."""
    g = _get_client()
    repo = g.get_repo(repo_name)
    try:
        # File exists — update it
        existing = repo.get_contents(file_path)
        repo.update_file(file_path, commit_message, content, existing.sha)
        action = "updated"
    except GithubException:
        # File doesn't exist — create it
        repo.create_file(file_path, commit_message, content)
        action = "created"
    return json.dumps({"action": action, "file": file_path, "repo": repo_name})

@tool
def github_read_file(repo_name: str, file_path: str) -> str:
    """Read a file from a GitHub repo.
    Args: repo_name (owner/repo format), file_path."""
    g = _get_client()
    repo = g.get_repo(repo_name)
    content = repo.get_contents(file_path)
    return base64.b64decode(content.content).decode("utf-8")

@tool
def github_create_pull_request(repo_name: str, title: str, body: str,
                                 head_branch: str, base_branch: str = "main") -> str:
    """Create a pull request in a GitHub repo.
    Args: repo_name, title, body (PR description), head_branch (source), base_branch (target, default 'main')."""
    g = _get_client()
    repo = g.get_repo(repo_name)
    pr = repo.create_pull(title=title, body=body, head=head_branch, base=base_branch)
    return json.dumps({"pr_url": pr.html_url, "pr_number": pr.number})

@tool
def github_list_files(repo_name: str, path: str = "") -> str:
    """List files in a GitHub repo directory.
    Args: repo_name (owner/repo), path (directory path, empty for root)."""
    g = _get_client()
    repo = g.get_repo(repo_name)
    contents = repo.get_contents(path or "")
    files = [{"name": f.name, "path": f.path, "type": f.type} for f in contents]
    return json.dumps(files)

GITHUB_TOOLS = [github_create_repo, github_create_file, github_read_file,
                github_create_pull_request, github_list_files]
```

---

### Step 6: Create `tools/code_exec_tools.py`

New file. E2B sandbox tools. The sandbox is kept alive per invocation using a context manager. For iterative dev loops, the Dev agent should use the same sandbox session by passing a `sandbox_id`.

```python
"""
tools/code_exec_tools.py — E2B sandbox tools for the Dev and QA agents.

The sandbox is a secure cloud microVM where code can be written, executed,
and tested. Results (stdout, stderr, errors) are returned to the agent.
"""

import json
import os
from langchain_core.tools import tool
from e2b_code_interpreter import Sandbox
from config.settings import e2b_cfg

# Module-level sandbox instance — reused within a single agent node execution.
# Initialized on first use, destroyed when the node completes.
_active_sandbox: Sandbox | None = None

def get_sandbox() -> Sandbox:
    """Get or create the active E2B sandbox for this session."""
    global _active_sandbox
    if _active_sandbox is None:
        _active_sandbox = Sandbox(api_key=e2b_cfg.api_key, timeout=300)
    return _active_sandbox

def close_sandbox():
    """Close and destroy the active sandbox."""
    global _active_sandbox
    if _active_sandbox is not None:
        _active_sandbox.kill()
        _active_sandbox = None

@tool
def run_python_code(code: str) -> str:
    """Execute Python code in a secure cloud sandbox and return the output.
    Args: code (Python code to execute).
    Returns: JSON with stdout, stderr, error (if any)."""
    sandbox = get_sandbox()
    execution = sandbox.run_code(code)
    return json.dumps({
        "stdout": execution.logs.stdout,
        "stderr": execution.logs.stderr,
        "error": str(execution.error) if execution.error else None,
    })

@tool
def run_shell_command(command: str) -> str:
    """Run a shell command in the sandbox (e.g., install packages, run tests, git ops).
    Args: command (shell command string, e.g. 'pip install requests' or 'pytest tests/').
    Returns: JSON with stdout, stderr, exit_code."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(command, timeout=120)
    return json.dumps({
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    })

@tool
def write_sandbox_file(file_path: str, content: str) -> str:
    """Write a file to the sandbox filesystem.
    Args: file_path (e.g. 'src/main.py'), content (file content as string).
    Returns: confirmation message."""
    sandbox = get_sandbox()
    sandbox.files.write(file_path, content)
    return f"File written: {file_path}"

@tool
def read_sandbox_file(file_path: str) -> str:
    """Read a file from the sandbox filesystem.
    Args: file_path.
    Returns: file content as string."""
    sandbox = get_sandbox()
    return sandbox.files.read(file_path)

@tool
def install_package(package_name: str) -> str:
    """Install a Python package in the sandbox.
    Args: package_name (e.g. 'requests' or 'fastapi uvicorn').
    Returns: installation result."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(f"pip install {package_name} -q", timeout=60)
    return json.dumps({"stdout": result.stdout, "stderr": result.stderr, "exit_code": result.exit_code})

@tool
def run_tests(test_command: str = "pytest") -> str:
    """Run the test suite in the sandbox.
    Args: test_command (default 'pytest', can be 'pytest tests/ -v' or 'python -m pytest').
    Returns: JSON with test output, pass/fail status."""
    sandbox = get_sandbox()
    result = sandbox.commands.run(test_command, timeout=120)
    passed = result.exit_code == 0
    return json.dumps({
        "passed": passed,
        "stdout": result.stdout,
        "stderr": result.stderr,
        "exit_code": result.exit_code,
    })

CODE_EXEC_TOOLS = [run_python_code, run_shell_command, write_sandbox_file,
                   read_sandbox_file, install_package, run_tests]
```

---

### Step 7: Create `agents/definitions.py`

Extract agent personas from the old `factory.py`. No CrewAI dependency here — just plain Python dicts.

```python
"""
agents/definitions.py — Agent persona definitions (role, goal, backstory).

These are used by graph/nodes.py to build LangGraph ReAct agents.
Extracted from the old CrewAI factory — no framework dependency here.
"""

from config.settings import slack_cfg

AGENT_PERSONAS = {
    "ceo": {
        "role": "CEO / Chief Executive Officer",
        "goal": (
            "Lead the startup with a clear vision. Make high-level strategic decisions, "
            "coordinate all team members, set priorities, resolve conflicts, and ensure "
            "the startup moves from idea to production. Communicate decisions in Slack "
            "and document strategy in Notion."
        ),
        "backstory": (
            "You are a serial entrepreneur who has founded two successful startups "
            "(one acquired, one IPO'd). You are decisive yet collaborative, data-driven "
            "but people-first. You run tight weekly sprints and hold everyone accountable. "
            "You believe in radical transparency and post all major decisions to Slack. "
            "You document strategy, OKRs, and roadmaps in Notion."
        ),
    },
    "cfo": {
        "role": "CFO / Chief Financial Officer",
        "goal": (
            "Manage the startup's finances: budgets, burn rate, runway, fundraising "
            "strategy, pricing models, and financial projections. Provide weekly "
            "financial updates and flag risks early."
        ),
        "backstory": (
            "You are a finance leader with experience at both venture-backed startups "
            "and investment banks. You can build financial models from scratch, negotiate "
            "term sheets, and translate complex numbers into clear founder-friendly language. "
            "You are conservative on spend, aggressive on finding revenue, and meticulous "
            "about tracking every dollar."
        ),
    },
    "product": {
        "role": "VP of Product",
        "goal": (
            "Define product vision, write PRDs, prioritize the backlog, and ensure "
            "the team builds what users actually need. Bridge between business goals "
            "and engineering execution."
        ),
        "backstory": (
            "You are a product leader who shipped 3 B2B SaaS products from 0 to 1. "
            "You are obsessed with user research and data-driven prioritization. "
            "You write PRDs that engineers love — clear, concise, with edge cases covered. "
            "You use the RICE framework for prioritization."
        ),
    },
    "dev": {
        "role": "Lead Software Developer",
        "goal": (
            "Design system architecture, write production-quality code, set up CI/CD, "
            "choose the tech stack, and lead engineering execution. Write real, runnable "
            "code. Use the code execution sandbox to run and test code. Commit all code "
            "to the project GitHub repo."
        ),
        "backstory": (
            "You are a staff-level full-stack engineer with 12+ years of experience "
            "across Python, TypeScript, React, cloud infra (AWS/GCP), databases, and APIs. "
            "You believe in clean code, comprehensive testing, and pragmatic architecture. "
            "You can prototype fast but also build for scale. IMPORTANT: You write actual, "
            "executable code — not pseudocode or descriptions. You always test your code "
            "in the sandbox before committing. If you have a question about requirements, "
            "you flag it explicitly so Product can clarify."
        ),
    },
    "qa": {
        "role": "QA Lead / Quality Assurance",
        "goal": (
            "Ensure product quality through test planning, writing automated tests, "
            "running them in the sandbox, and tracking bugs. Gate code from going to "
            "production until tests pass."
        ),
        "backstory": (
            "You are a QA engineer with 8 years of experience in both manual and "
            "automated testing. You've built test frameworks from scratch. You believe "
            "QA should be involved from the PRD stage. You write pytest tests, run them "
            "in the sandbox, and report results. If you find critical bugs, you clearly "
            "describe them so Dev can fix them. You never approve code with failing tests."
        ),
    },
    "marketing": {
        "role": "Head of Marketing",
        "goal": (
            "Build brand awareness, create go-to-market strategies, produce content, "
            "manage launch campaigns, and drive user acquisition."
        ),
        "backstory": (
            "You have driven growth from 0 to 100k users at two B2B SaaS startups. "
            "You are expert at positioning, content marketing, SEO, Product Hunt launches, "
            "social media strategy, and building a brand that resonates. You think in "
            "funnels — TOFU, MOFU, BOFU — and measure everything."
        ),
    },
    "sales": {
        "role": "Head of Sales",
        "goal": (
            "Build the sales pipeline, create outreach strategies, write pitch decks, "
            "handle objections, and close deals. Maintain a CRM-style leads database."
        ),
        "backstory": (
            "You have closed $5M+ in enterprise SaaS deals and built outbound pipelines "
            "from scratch. You understand buyer psychology, can navigate complex buying "
            "committees, and know how to tailor pitches for technical vs. executive "
            "audiences. You never oversell — you solve customer problems."
        ),
    },
}
```

---

### Step 8: Create `graph/state.py`

```python
"""
graph/state.py — Shared state passed between all LangGraph nodes.

Every project has a unique thread_id (= project_id) stored in the SQLite
checkpointer. Loading state: graph.invoke(input, config={"configurable": {"thread_id": "project-xyz"}})
"""

from __future__ import annotations
from typing import Annotated, Optional
from typing_extensions import TypedDict
import operator


class Message(TypedDict):
    """A message between two agents, also posted to Slack."""
    from_agent: str       # e.g. "dev"
    to_agent: str         # e.g. "product"
    content: str
    message_type: str     # "question", "answer", "update", "bug_report", "approval"
    timestamp: str


class ProjectState(TypedDict):
    # ── Identity ─────────────────────────────────────────────────────────────
    project_id: str               # = LangGraph thread_id
    project_name: str             # short slug, also GitHub repo name
    idea: str                     # the original idea from the founder

    # ── Routing ──────────────────────────────────────────────────────────────
    phase: str                    # "routing" | "planning" | "development" | "qa" | "business" | "summary" | "done"
    agents_needed: list[str]      # CEO decides, e.g. ["product", "dev", "qa"] or all 7

    # ── Communication (visible in Slack) ─────────────────────────────────────
    # Use Annotated with operator.add so messages are appended, not overwritten
    messages: Annotated[list[Message], operator.add]
    slack_channel: str            # channel where this project is discussed
    slack_thread_ts: str          # thread timestamp — all messages go in one thread

    # ── Pending clarification (feedback loops) ────────────────────────────────
    pending_clarification: Optional[dict]  # {"from": "dev", "to": "product", "question": "...", "answered": False}

    # ── Documents produced (Notion pages) ────────────────────────────────────
    notion_vision_url: str
    notion_prd_url: str
    notion_arch_url: str
    notion_test_strategy_url: str
    notion_financial_url: str
    notion_gtm_url: str
    notion_sales_url: str
    notion_task_db_id: str        # ID of the master task board database

    # ── Document content (passed as context between agents) ───────────────────
    vision_content: str
    prd_content: str
    arch_content: str

    # ── Development ───────────────────────────────────────────────────────────
    github_repo_url: str
    github_repo_name: str         # owner/repo format
    code_iterations: int          # how many dev→qa→dev cycles completed
    max_code_iterations: int      # default 3; CEO can escalate if exceeded
    qa_approved: bool
    qa_feedback: str              # bug report from QA to Dev (empty if approved)

    # ── Error tracking ────────────────────────────────────────────────────────
    errors: Annotated[list[str], operator.add]
```

---

### Step 9: Create `graph/checkpointer.py`

```python
"""
graph/checkpointer.py — SQLite-based state persistence for all projects.

Every project is stored by thread_id. To resume a past project:
  graph.invoke({"idea": "add feature X"}, config={"configurable": {"thread_id": "project-saas-abc"}})
"""

import os
from langgraph.checkpoint.sqlite import SqliteSaver

DB_PATH = os.path.join(os.path.dirname(__file__), "..", "data", "projects.db")

def get_checkpointer() -> SqliteSaver:
    """Return a SQLite checkpointer. Creates the data/ directory and DB if needed."""
    os.makedirs(os.path.dirname(DB_PATH), exist_ok=True)
    return SqliteSaver.from_conn_string(DB_PATH)
```

This creates `data/projects.db` in the project root. Add `data/` to `.gitignore`.

---

### Step 10: Create `graph/nodes.py`

This is the largest file. Each agent is a LangGraph node — an async function that receives `ProjectState`, invokes a Gemini ReAct agent with appropriate tools, posts to Slack, and returns updated state.

**Pattern for every node:**

```python
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain.agents import create_react_agent, AgentExecutor
from langchain_core.prompts import PromptTemplate
from config.settings import gemini_cfg, slack_cfg
from agents.definitions import AGENT_PERSONAS
from tools.slack_tools import SLACK_TOOLS, slack_reply_thread, slack_send_message
from tools.notion_tools import NOTION_TOOLS
from tools.github_tools import GITHUB_TOOLS
from tools.code_exec_tools import CODE_EXEC_TOOLS, close_sandbox
from graph.state import ProjectState, Message
from datetime import datetime, timezone
import uuid, json, logging

logger = logging.getLogger(__name__)

def _build_llm():
    return ChatGoogleGenerativeAI(
        model=gemini_cfg.model,
        google_api_key=gemini_cfg.api_key,
        temperature=gemini_cfg.temperature,
    )

def _post_to_slack(state: ProjectState, agent_key: str, message: str) -> None:
    """Post a message to the project's Slack thread."""
    persona = AGENT_PERSONAS[agent_key]
    agent_name = persona["role"]
    channel = state.get("slack_channel") or slack_cfg.channel_general
    thread_ts = state.get("slack_thread_ts", "")

    if thread_ts:
        slack_reply_thread.invoke({
            "channel": channel,
            "thread_ts": thread_ts,
            "message": message,
            "agent_name": agent_name,
        })
    else:
        slack_send_message.invoke({
            "channel": channel,
            "message": message,
            "agent_name": agent_name,
        })

def _make_message(from_agent: str, to_agent: str, content: str, msg_type: str) -> Message:
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        message_type=msg_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )

def _run_agent(agent_key: str, tools: list, prompt_text: str) -> str:
    """Run a ReAct agent with the given tools and prompt. Returns the final output."""
    llm = _build_llm()
    persona = AGENT_PERSONAS[agent_key]

    system_prompt = (
        f"You are the {persona['role']} of a startup.\n"
        f"Your goal: {persona['goal']}\n"
        f"Your background: {persona['backstory']}\n\n"
        f"Task: {prompt_text}\n\n"
        f"Use your tools as needed. Be decisive and action-oriented."
    )

    # Use create_react_agent with Gemini (which supports tool calling natively)
    agent = create_react_agent(llm, tools)
    executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=15)
    result = executor.invoke({"input": system_prompt})
    return result.get("output", "")
```

**CEO Router Node:**

```python
def ceo_router_node(state: ProjectState) -> dict:
    """
    CEO analyzes the idea and decides:
    1. Which agents are needed (e.g., skip CFO/Marketing/Sales for internal tools)
    2. Sets project_name (slug for GitHub repo)
    3. Creates GitHub repo
    4. Posts kickoff message to Slack
    5. Always includes: product, dev, qa
    """
    idea = state["idea"]
    project_id = state["project_id"]

    prompt = f"""
You are the CEO. A startup idea has arrived:

"{idea}"

Your tasks:
1. Create a short project name (kebab-case, max 30 chars) for the GitHub repo and project tracking.
2. Decide which team members are needed. ALWAYS include: product, dev, qa.
   Add cfo if the idea needs financial modeling. Add marketing and sales if it's a customer-facing product.
3. Create a GitHub repo using github_create_repo with the project name.
4. Post a kickoff message to Slack channel {state.get('slack_channel', slack_cfg.channel_general)}
   announcing the project, the team assembled, and the first steps.
5. Create a tasks database in Notion titled "{project_id} — Tasks".

Respond with a JSON block (inside ```json ... ```) containing:
{{
  "project_name": "my-saas-app",
  "agents_needed": ["product", "dev", "qa"],  // or include cfo, marketing, sales
  "github_repo_url": "https://github.com/...",
  "github_repo_name": "owner/repo-name",
  "notion_task_db_id": "abc123",
  "kickoff_summary": "One paragraph summary of the plan."
}}
"""

    tools = SLACK_TOOLS + NOTION_TOOLS + GITHUB_TOOLS
    output = _run_agent("ceo", tools, prompt)

    # Parse JSON from output
    import re
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    parsed = json.loads(json_match.group(1)) if json_match else {}

    _post_to_slack(state, "ceo", parsed.get("kickoff_summary", output))

    return {
        "phase": "planning",
        "project_name": parsed.get("project_name", project_id),
        "agents_needed": parsed.get("agents_needed", ["product", "dev", "qa"]),
        "github_repo_url": parsed.get("github_repo_url", ""),
        "github_repo_name": parsed.get("github_repo_name", ""),
        "notion_task_db_id": parsed.get("notion_task_db_id", ""),
        "messages": [_make_message("ceo", "team", parsed.get("kickoff_summary", ""), "update")],
    }
```

**Product Node:**

```python
def product_node(state: ProjectState) -> dict:
    """VP Product writes the PRD and saves it to Notion."""

    context = ""
    if state.get("pending_clarification") and state["pending_clarification"].get("to") == "product":
        # Dev asked Product a question — answer it instead of writing PRD
        question = state["pending_clarification"]["question"]
        context = f"\n\nIMPORTANT: The Dev team has a clarification question:\n{question}\nPlease answer it clearly."

    prompt = f"""
Project idea: "{state['idea']}"
CEO vision summary: {state.get('vision_content', 'See Slack for CEO kickoff message.')}

{context}

Your tasks:
1. Write a detailed PRD (Product Requirements Document) covering:
   - Problem statement
   - Target personas (2-3)
   - MVP feature set (5-7 features with user stories)
   - Success metrics
   - Out of scope items
   - Technical requirements / constraints for Dev
2. Save the PRD to Notion as a page titled "PRD — {state.get('project_name', 'Project')}"
3. Post a summary to Slack thread.

If answering a Dev clarification: post your answer to Slack thread, then output:
```json
{{"clarification_answer": "Your detailed answer here"}}
```

Otherwise output:
```json
{{"prd_content": "Full PRD text here", "notion_prd_url": "https://notion.so/..."}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS
    output = _run_agent("product", tools, prompt)

    import re, json
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    parsed = json.loads(json_match.group(1)) if json_match else {}

    _post_to_slack(state, "product", f"PRD complete. See Notion: {parsed.get('notion_prd_url', '')}")

    new_state = {
        "messages": [_make_message("product", "team", "PRD complete.", "update")],
        "notion_prd_url": parsed.get("notion_prd_url", ""),
        "prd_content": parsed.get("prd_content", output),
    }

    # If answering a clarification, clear the pending question
    if parsed.get("clarification_answer"):
        new_state["pending_clarification"] = {
            **state["pending_clarification"],
            "answer": parsed["clarification_answer"],
            "answered": True,
        }
        _post_to_slack(state, "product", f"Answering Dev's question: {parsed['clarification_answer']}")

    return new_state
```

**Dev Node:**

```python
def dev_node(state: ProjectState) -> dict:
    """
    Lead Dev writes real code, runs it in E2B sandbox, commits to GitHub.
    Can ask Product for clarification (sets pending_clarification).
    Iterates until tests pass or max_iterations reached.
    """
    qa_feedback = state.get("qa_feedback", "")
    pending = state.get("pending_clarification", {})

    # If Product answered a clarification, include the answer
    clarification_context = ""
    if pending and pending.get("answered"):
        clarification_context = f"\nProduct clarified: {pending.get('answer', '')}"

    prompt = f"""
Project: {state.get('project_name')}
GitHub repo: {state.get('github_repo_name')}
PRD: {state.get('prd_content', 'See Notion for PRD.')}
Architecture: {state.get('arch_content', '')}
{clarification_context}

{"QA found these issues that need fixing:" + qa_feedback if qa_feedback else ""}

Your tasks:
1. If you have a question about requirements that blocks you, post it to Slack and output:
   ```json
   {{"needs_clarification": true, "question": "Your specific question for Product"}}
   ```
   STOP after this — do not write code until clarified.

2. Otherwise, implement the MVP:
   a. Use write_sandbox_file to write all source files
   b. Use run_shell_command to install dependencies
   c. Use run_python_code or run_shell_command to run the code and verify it works
   d. Write tests (pytest) using write_sandbox_file
   e. Use run_tests to verify all tests pass
   f. Use github_create_file to commit each source file to {state.get('github_repo_name')}
   g. Post a summary of what was built to Slack

Output:
```json
{{
  "needs_clarification": false,
  "files_committed": ["src/main.py", "tests/test_main.py"],
  "summary": "What was built and what was tested"
}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS + GITHUB_TOOLS + CODE_EXEC_TOOLS
    output = _run_agent("dev", tools, prompt)
    close_sandbox()  # clean up E2B sandbox after dev is done

    import re, json
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    parsed = json.loads(json_match.group(1)) if json_match else {}

    new_state = {
        "messages": [_make_message("dev", "team", parsed.get("summary", "Dev work complete."), "update")],
        "arch_content": parsed.get("summary", state.get("arch_content", "")),
        "qa_feedback": "",  # clear previous QA feedback
    }

    if parsed.get("needs_clarification"):
        question = parsed.get("question", "")
        _post_to_slack(state, "dev", f"Need clarification from Product before coding:\n{question}")
        new_state["pending_clarification"] = {
            "from": "dev",
            "to": "product",
            "question": question,
            "answered": False,
        }
    else:
        _post_to_slack(state, "dev", f"Code committed to GitHub. Summary: {parsed.get('summary', '')}")
        new_state["pending_clarification"] = None

    return new_state
```

**QA Node:**

```python
def qa_node(state: ProjectState) -> dict:
    """
    QA pulls code from GitHub, writes tests, runs them in sandbox.
    Approves or sends bugs back to Dev.
    """

    prompt = f"""
Project: {state.get('project_name')}
GitHub repo: {state.get('github_repo_name')}
PRD: {state.get('prd_content', '')}
Iteration: {state.get('code_iterations', 0) + 1} of {state.get('max_code_iterations', 3)}

Your tasks:
1. Use github_list_files and github_read_file to review the code in the repo.
2. Write comprehensive pytest tests covering all PRD features using write_sandbox_file.
3. Use run_shell_command to install dependencies.
4. Use run_tests to run the test suite.
5. Post results to Slack.

If ALL tests pass:
```json
{{"qa_approved": true, "test_summary": "X/Y tests passed. All PRD features verified."}}
```

If tests fail or bugs found:
```json
{{
  "qa_approved": false,
  "qa_feedback": "Detailed bug report: list each bug with steps to reproduce and expected vs actual behavior",
  "test_summary": "X/Y tests passed. Y failed."
}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS + GITHUB_TOOLS + CODE_EXEC_TOOLS
    output = _run_agent("qa", tools, prompt)
    close_sandbox()

    import re, json
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    parsed = json.loads(json_match.group(1)) if json_match else {}

    approved = parsed.get("qa_approved", False)
    feedback = parsed.get("qa_feedback", "")
    summary = parsed.get("test_summary", "")

    if approved:
        _post_to_slack(state, "qa", f"QA APPROVED. {summary}")
    else:
        _post_to_slack(state, "qa", f"QA FAILED — sending back to Dev.\n{feedback}")

    return {
        "qa_approved": approved,
        "qa_feedback": feedback,
        "code_iterations": state.get("code_iterations", 0) + 1,
        "messages": [_make_message("qa", "dev" if not approved else "team",
                                   feedback if not approved else summary,
                                   "bug_report" if not approved else "approval")],
    }
```

**CFO, Marketing, Sales Nodes:** Same pattern as Product node. Each queries the idea + PRD + vision context, produces a document (Financial Plan / GTM Strategy / Sales Playbook), saves to Notion, posts summary to Slack. Can be shorter since no code execution or feedback loops needed.

**CEO Summary Node:**

```python
def ceo_summary_node(state: ProjectState) -> dict:
    """
    CEO synthesizes all outputs into an executive summary, posts to all Slack channels,
    and creates the master 2-week task board in Notion.
    """
    prompt = f"""
Project "{state.get('project_name')}" is complete.

Summary of what was built:
- GitHub repo: {state.get('github_repo_url')}
- PRD: {state.get('notion_prd_url')}
- Architecture: {state.get('notion_arch_url', 'N/A')}
- Financial Plan: {state.get('notion_financial_url', 'N/A')}
- GTM Strategy: {state.get('notion_gtm_url', 'N/A')}
- Sales Playbook: {state.get('notion_sales_url', 'N/A')}

Team communications: {len(state.get('messages', []))} messages exchanged.
QA iterations: {state.get('code_iterations', 0)}

Your tasks:
1. Write an executive launch summary in Notion covering: what was built, key decisions made, risks, next steps.
2. Create the first 2-week task board: add 5-10 tasks to the Notion task database {state.get('notion_task_db_id', '')}
   assigned to each relevant team member.
3. Post a launch announcement to Slack general channel {slack_cfg.channel_general}.

Output:
```json
{{"launch_summary": "One paragraph", "notion_launch_url": "https://notion.so/..."}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS
    output = _run_agent("ceo", tools, prompt)

    _post_to_slack(state, "ceo", "Project complete! See Notion for full launch plan.")

    return {
        "phase": "done",
        "messages": [_make_message("ceo", "founder", "Project complete.", "update")],
    }
```

---

### Step 11: Create `graph/project_graph.py`

```python
"""
graph/project_graph.py — Main LangGraph StateGraph for the idea-to-completion workflow.

Graph topology:
  START → ceo_router
  ceo_router → product (always)
  product → dev  (if no pending clarification, or clarification answered)
  product → product (if dev had a clarification, product answers and routes back to dev)
  dev → product  (if dev has a clarification question)
  dev → qa       (if no clarification needed and code is written)
  qa → dev       (if tests fail and iterations < max)
  qa → ceo_escalation (if iterations >= max)
  qa → [business agents or ceo_summary] (if approved)
  cfo / marketing / sales → ceo_summary (if in agents_needed)
  ceo_summary → END
"""

from langgraph.graph import StateGraph, START, END
from graph.state import ProjectState
from graph.nodes import (
    ceo_router_node, product_node, dev_node, qa_node,
    cfo_node, marketing_node, sales_node, ceo_summary_node,
)


def _route_after_ceo(state: ProjectState) -> str:
    """CEO always goes to Product first."""
    return "product"


def _route_after_product(state: ProjectState) -> str:
    """After Product runs, go to Dev — unless Dev asked a question and Product just answered."""
    pending = state.get("pending_clarification") or {}
    if pending.get("answered"):
        return "dev"  # Product answered Dev's question, back to Dev
    return "dev"  # Normal PRD flow


def _route_after_dev(state: ProjectState) -> str:
    """After Dev runs, either ask Product for clarification or go to QA."""
    pending = state.get("pending_clarification") or {}
    if pending and not pending.get("answered"):
        return "product"  # Dev has a question — route to Product
    return "qa"


def _route_after_qa(state: ProjectState) -> str:
    """After QA, approve or send back to Dev. If too many iterations, escalate to CEO."""
    if state.get("qa_approved"):
        return "business_check"
    iterations = state.get("code_iterations", 0)
    max_iter = state.get("max_code_iterations", 3)
    if iterations >= max_iter:
        return "ceo_summary"  # CEO decides what to do (flags as at-risk in summary)
    return "dev"  # Send back to Dev with QA feedback


def _route_business(state: ProjectState) -> list[str]:
    """Fan out to whichever business agents are needed, or go straight to CEO summary."""
    needed = state.get("agents_needed", [])
    next_nodes = []
    if "cfo" in needed:
        next_nodes.append("cfo")
    if "marketing" in needed:
        next_nodes.append("marketing")
    if "sales" in needed:
        next_nodes.append("sales")
    return next_nodes if next_nodes else ["ceo_summary"]


def build_project_graph(checkpointer):
    """Build and compile the project graph with checkpointing."""
    g = StateGraph(ProjectState)

    # Add nodes
    g.add_node("ceo_router", ceo_router_node)
    g.add_node("product", product_node)
    g.add_node("dev", dev_node)
    g.add_node("qa", qa_node)
    g.add_node("cfo", cfo_node)
    g.add_node("marketing", marketing_node)
    g.add_node("sales", sales_node)
    g.add_node("ceo_summary", ceo_summary_node)

    # Edges
    g.add_edge(START, "ceo_router")
    g.add_conditional_edges("ceo_router", _route_after_ceo, {"product": "product"})
    g.add_conditional_edges("product", _route_after_product, {"dev": "dev"})
    g.add_conditional_edges("dev", _route_after_dev, {"product": "product", "qa": "qa"})
    g.add_conditional_edges("qa", _route_after_qa, {
        "dev": "dev",
        "business_check": "ceo_summary",  # simplified: CEO decides on business agents
        "ceo_summary": "ceo_summary",
    })
    g.add_edge("cfo", "ceo_summary")
    g.add_edge("marketing", "ceo_summary")
    g.add_edge("sales", "ceo_summary")
    g.add_edge("ceo_summary", END)

    return g.compile(checkpointer=checkpointer)
```

**Note on business agent routing:** For simplicity, route QA approval → ceo_summary, and have the CEO summary node invoke business agents internally if needed (or make them separate graph runs). Full parallel fan-out via `Send` API is possible but adds complexity; implement sequentially first.

---

### Step 12: Create `graph/standup_graph.py`

Replaces `run_daily_standup` from the old `scheduled_crews.py`. Much simpler — no LangGraph needed, just sequential calls.

```python
"""
graph/standup_graph.py — Daily standup: each agent posts status to Slack, CEO summarizes.
Saves summary to Notion. No full LangGraph graph needed — sequential function calls.
"""

import logging
from datetime import datetime
from langchain.agents import create_react_agent, AgentExecutor
from langchain_google_genai import ChatGoogleGenerativeAI
from config.settings import gemini_cfg, slack_cfg, notion_cfg
from agents.definitions import AGENT_PERSONAS
from tools.slack_tools import SLACK_TOOLS
from tools.notion_tools import NOTION_TOOLS

logger = logging.getLogger(__name__)

STANDUP_AGENTS = ["product", "dev", "qa", "marketing", "sales", "cfo"]


def run_daily_standup():
    """Run the daily standup: each agent queries Notion and posts to Slack. CEO summarizes."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    channel = slack_cfg.channel_standup or slack_cfg.channel_general

    logger.info(f"Running daily standup for {today}")

    llm = ChatGoogleGenerativeAI(
        model=gemini_cfg.model,
        google_api_key=gemini_cfg.api_key,
        temperature=0.5,
    )

    tools = SLACK_TOOLS + NOTION_TOOLS

    for agent_key in STANDUP_AGENTS:
        persona = AGENT_PERSONAS[agent_key]
        prompt = (
            f"It's the daily standup ({today}). As the {persona['role']}:\n"
            f"1. Query the Notion tasks database (root page: {notion_cfg.root_page_id}) "
            f"   for tasks assigned to you (filter by assignee = '{persona['role']}')\n"
            f"2. Post a standup update to Slack channel {channel} with format:\n"
            f"   *Yesterday:* what you completed\n"
            f"   *Today:* what you're working on\n"
            f"   *Blockers:* any issues (or 'None')\n"
            f"Keep it to 3-5 bullet points total."
        )
        try:
            agent = create_react_agent(llm, tools)
            executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=5)
            executor.invoke({"input": f"You are the {persona['role']}. {persona['backstory']}\n\n{prompt}"})
        except Exception as e:
            logger.error(f"Standup error for {agent_key}: {e}")

    # CEO synthesizes
    ceo_prompt = (
        f"Read the standup updates just posted to Slack channel {channel}. "
        f"Post a brief CEO synthesis: top team priorities today, any blockers needing your attention. "
        f"Then save a standup summary to Notion (create a page titled 'Standup {today}' under root page {notion_cfg.root_page_id})."
    )
    try:
        ceo_persona = AGENT_PERSONAS["ceo"]
        agent = create_react_agent(llm, tools)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=False, max_iterations=8)
        executor.invoke({"input": f"You are the {ceo_persona['role']}. {ceo_persona['backstory']}\n\n{ceo_prompt}"})
    except Exception as e:
        logger.error(f"CEO standup synthesis error: {e}")

    logger.info("Daily standup complete.")


def run_weekly_review():
    """Simplified weekly review: each agent posts status to Slack + saves to Notion."""
    # Same pattern as standup but with a longer prompt asking for weekly metrics.
    # Implementation follows the same structure as run_daily_standup().
    # Left as an exercise — same pattern, different prompt strings.
    pass
```

---

### Step 13: Update `workflows/slack_bot.py`

Replace the `_handle_agent_task` function and `SlackBot` class to use LangGraph instead of CrewAI. Keep the routing logic (`_route_message` and `_ROUTE_KEYWORDS`) — it's still useful.

**Key changes:**
1. Import `build_project_graph` and `get_checkpointer` instead of CrewAI
2. `_handle_agent_task` now checks: is this a new project idea? Or a message for an existing project?
3. New project ideas → kick off `project_graph` with a new `thread_id`
4. Messages to specific agents → run a single-agent LangGraph node
5. The `SlackBot.__init__` signature changes from `roster: AgentRoster` to just no roster needed

**New `_handle_agent_task`:**

```python
import uuid, re
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
from config.settings import slack_cfg

PROJECT_TRIGGER_KEYWORDS = ["build", "create", "make", "develop", "idea", "startup", "launch", "new project"]

def _is_new_project(text: str) -> bool:
    return any(kw in text.lower() for kw in PROJECT_TRIGGER_KEYWORDS)

def _handle_agent_task(agent_key: str, user_msg: str, channel: str, thread_ts: str):
    if agent_key == "ceo" and _is_new_project(user_msg):
        # Kick off a full project
        project_id = f"project-{uuid.uuid4().hex[:8]}"
        checkpointer = get_checkpointer()
        graph = build_project_graph(checkpointer)

        config = {"configurable": {"thread_id": project_id}}
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
            # Notion URLs (populated by nodes)
            "notion_vision_url": "", "notion_prd_url": "", "notion_arch_url": "",
            "notion_test_strategy_url": "", "notion_financial_url": "",
            "notion_gtm_url": "", "notion_sales_url": "", "notion_task_db_id": "",
            # Content
            "vision_content": "", "prd_content": "", "arch_content": "",
        }

        try:
            graph.invoke(initial_state, config=config)
        except Exception as e:
            logger.error(f"Project graph error: {e}")
    else:
        # One-off agent response (not a full project)
        # Run a simple single-agent executor for the message
        from langchain.agents import create_react_agent, AgentExecutor
        from langchain_google_genai import ChatGoogleGenerativeAI
        from agents.definitions import AGENT_PERSONAS
        from tools.slack_tools import SLACK_TOOLS, slack_reply_thread
        from tools.notion_tools import NOTION_TOOLS

        persona = AGENT_PERSONAS.get(agent_key, AGENT_PERSONAS["ceo"])
        llm = ChatGoogleGenerativeAI(model=gemini_cfg.model, google_api_key=gemini_cfg.api_key, temperature=0.7)
        tools = SLACK_TOOLS + NOTION_TOOLS

        prompt = (
            f"You received this message from the founder: \"{user_msg}\"\n"
            f"Respond as the {persona['role']}. Take action if needed. "
            f"Post your response to Slack channel {channel} in thread {thread_ts}."
        )

        agent = create_react_agent(llm, tools)
        executor = AgentExecutor(agent=agent, tools=tools, verbose=True, max_iterations=10)
        try:
            executor.invoke({"input": f"You are the {persona['role']}. {persona['backstory']}\n\n{prompt}"})
        except Exception as e:
            logger.error(f"Agent task error ({agent_key}): {e}")
            slack_reply_thread.invoke({"channel": channel, "thread_ts": thread_ts,
                                       "message": f"Error: {str(e)[:200]}", "agent_name": persona["role"]})
```

Also update the `SlackBot.__init__` to remove the `AgentRoster` dependency:
```python
def __init__(self):  # no longer needs roster
    self.app = App(token=slack_cfg.bot_token, signing_secret=slack_cfg.signing_secret)
    self._register_handlers()
```

---

### Step 14: Update `workflows/scheduler.py`

Replace CrewAI imports with the new standup graph functions.

```python
# Remove:
from agents.factory import AgentRoster
from workflows.scheduled_crews import run_daily_standup, run_weekly_review, run_sprint_planning

# Add:
from graph.standup_graph import run_daily_standup, run_weekly_review

# Update _setup_jobs — remove sprint planning job (not needed), keep standup + weekly review
# Update _biweekly_sprint — remove entirely
# Update TeamScheduler.__init__ to not take roster argument:
def __init__(self):
    self.scheduler = BackgroundScheduler()
    self._setup_jobs()
```

---

### Step 15: Update `main.py`

```python
# Remove: build_all_agents, AgentRoster, run_idea_kickoff, run_standup, etc.
# Remove: from agents.factory import build_all_agents
# Remove: --kickoff, --standup, --review, --sprint args (or keep --kickoff for CLI use)

# Add:
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
from graph.standup_graph import run_daily_standup

# Update _validate_env to check E2B_API_KEY and GITHUB_TOKEN
# Update SlackBot() initialization (no roster arg)
# Update TeamScheduler() initialization (no roster arg)
# Add --kickoff --idea support via graph invocation:
if args.kickoff:
    idea = args.idea or app_cfg.startup_idea
    project_id = f"project-{uuid.uuid4().hex[:8]}"
    checkpointer = get_checkpointer()
    graph = build_project_graph(checkpointer)
    config = {"configurable": {"thread_id": project_id}}
    initial_state = { ... }  # same as in slack_bot.py
    result = graph.invoke(initial_state, config=config)
```

---

### Step 16: Update `.env` and `.env.example`

Add to both files:
```
E2B_API_KEY=e2b_xxxx
GITHUB_TOKEN=ghp_xxxx
GITHUB_USERNAME=your-github-username
```

Add `data/` to `.gitignore` (SQLite DB lives there).

---

### Step 17: Delete Old Files

After all steps are complete and verified:
```bash
rm agents/factory.py
rm workflows/scheduled_crews.py
```

---

## Implementation Order (for the implementing agent)

Execute strictly in this order to avoid broken imports:

1. `requirements.txt` (install new packages immediately after)
2. `config/settings.py`
3. `tools/slack_tools.py`
4. `tools/notion_tools.py`
5. `tools/github_tools.py` (new)
6. `tools/code_exec_tools.py` (new)
7. `agents/definitions.py` (new)
8. `graph/__init__.py` (empty file)
9. `graph/state.py` (new)
10. `graph/checkpointer.py` (new)
11. `graph/nodes.py` (new — depends on steps 3-9)
12. `graph/project_graph.py` (new — depends on step 11)
13. `graph/standup_graph.py` (new — depends on steps 3-7)
14. `workflows/slack_bot.py` (modify — depends on steps 11-12)
15. `workflows/scheduler.py` (modify — depends on step 13)
16. `main.py` (modify — depends on steps 12-15)
17. Delete `agents/factory.py` and `workflows/scheduled_crews.py`
18. Add `data/` to `.gitignore`
19. Run `.venv/bin/python main.py --kickoff --idea "test idea"` to verify

---

## Testing the Implementation

### Smoke test (no Slack/Notion/E2B needed):
```bash
.venv/bin/python -c "from graph.project_graph import build_project_graph; from graph.checkpointer import get_checkpointer; print('Graph imports OK')"
```

### Full kickoff test:
```bash
.venv/bin/python main.py --kickoff --idea "A SaaS tool that helps freelancers track time and invoice clients automatically"
```

Expected output:
- GitHub repo created under your account
- Slack thread started in #ai-team-general
- CEO posts kickoff message
- Product posts PRD
- Dev writes and tests code, commits to GitHub
- QA runs tests
- CEO posts launch summary
- All in one Slack thread, all persisted in SQLite

### Return to a past project:
```bash
# Use the project_id printed at start of kickoff (e.g., "project-a1b2c3d4")
.venv/bin/python -c "
from graph.project_graph import build_project_graph
from graph.checkpointer import get_checkpointer
checkpointer = get_checkpointer()
graph = build_project_graph(checkpointer)
config = {'configurable': {'thread_id': 'project-a1b2c3d4'}}
graph.invoke({'idea': 'add a dashboard feature'}, config=config)
"
```

---

## Key Constraints for the Implementing Agent

1. **Do not use CrewAI** — remove all `from crewai import ...` imports
2. **Do not use `BaseTool`** — use `@tool` decorator from `langchain_core.tools`
3. **All tool functions must have complete docstrings** — LangGraph passes the docstring as the tool description to the LLM
4. **E2B sandbox is per-node** — call `close_sandbox()` at the end of every Dev and QA node
5. **All Slack posting uses the project thread** — use `slack_reply_thread` with `state["slack_thread_ts"]` for all in-project messages
6. **GitHub repo name** must be unique — prefix with the project_id or use a timestamp suffix if collision occurs
7. **JSON output from agents** — all nodes parse a ```json ... ``` block from the agent's output to extract structured data. If parsing fails, fall back gracefully (use the raw output string where a string is needed)
8. **State is append-only for lists** — `messages` and `errors` use `Annotated[list, operator.add]`, so return only the NEW items in those fields, not the full list
9. **Agents that are not needed simply don't run** — the conditional routing in `project_graph.py` ensures only nodes in `agents_needed` are invoked
10. **SQLite DB path** — `data/projects.db` relative to project root; create `data/` directory if it doesn't exist
