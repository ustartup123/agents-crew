"""
graph/nodes.py — All agent node functions for the LangGraph project graph.

Each agent is a LangGraph node — a function that receives ProjectState,
invokes a Gemini-powered ReAct agent with appropriate tools, posts to Slack,
and returns updated state.
"""

from __future__ import annotations

import json
import logging
import re
from datetime import datetime, timezone

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config.settings import gemini_cfg, slack_cfg
from agents.definitions import AGENT_PERSONAS
from tools.slack_tools import SLACK_TOOLS, slack_reply_thread, slack_send_message
from tools.notion_tools import NOTION_TOOLS
from tools.github_tools import GITHUB_TOOLS
from tools.code_exec_tools import CODE_EXEC_TOOLS, close_sandbox
from graph.state import ProjectState, Message

logger = logging.getLogger(__name__)


# ── Shared helpers ───────────────────────────────────────────────────────────

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

    try:
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
    except Exception as e:
        logger.error(f"Slack post error ({agent_key}): {e}")


def _make_message(from_agent: str, to_agent: str, content: str, msg_type: str) -> Message:
    return Message(
        from_agent=from_agent,
        to_agent=to_agent,
        content=content,
        message_type=msg_type,
        timestamp=datetime.now(timezone.utc).isoformat(),
    )


def _parse_json_from_output(output: str) -> dict:
    """Extract a JSON block from ```json ... ``` in agent output."""
    json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
    if json_match:
        try:
            return json.loads(json_match.group(1))
        except json.JSONDecodeError:
            logger.warning("Failed to parse JSON from agent output")
    return {}


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

    agent = create_react_agent(llm, tools, state_modifier=system_prompt)
    try:
        result = agent.invoke({"messages": [("user", prompt_text)]})
        # Extract the last AI message content
        ai_messages = [m for m in result.get("messages", []) if hasattr(m, "content") and m.type == "ai"]
        if ai_messages:
            return ai_messages[-1].content
        return str(result)
    except Exception as e:
        logger.error(f"Agent execution error ({agent_key}): {e}")
        return f"Error: {str(e)}"


# ── CEO Router Node ─────────────────────────────────────────────────────────

def ceo_router_node(state: ProjectState) -> dict:
    """
    CEO analyzes the idea and decides:
    1. Which agents are needed
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
  "agents_needed": ["product", "dev", "qa"],
  "github_repo_url": "https://github.com/...",
  "github_repo_name": "owner/repo-name",
  "notion_task_db_id": "abc123",
  "kickoff_summary": "One paragraph summary of the plan."
}}
"""

    tools = SLACK_TOOLS + NOTION_TOOLS + GITHUB_TOOLS
    output = _run_agent("ceo", tools, prompt)
    parsed = _parse_json_from_output(output)

    _post_to_slack(state, "ceo", parsed.get("kickoff_summary", output[:500]))

    return {
        "phase": "planning",
        "project_name": parsed.get("project_name", project_id),
        "agents_needed": parsed.get("agents_needed", ["product", "dev", "qa"]),
        "github_repo_url": parsed.get("github_repo_url", ""),
        "github_repo_name": parsed.get("github_repo_name", ""),
        "notion_task_db_id": parsed.get("notion_task_db_id", ""),
        "messages": [_make_message("ceo", "team", parsed.get("kickoff_summary", ""), "update")],
    }


# ── Product Node ─────────────────────────────────────────────────────────────

def product_node(state: ProjectState) -> dict:
    """VP Product writes the PRD and saves it to Notion."""
    context = ""
    if state.get("pending_clarification") and state["pending_clarification"].get("to") == "product":
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
    parsed = _parse_json_from_output(output)

    _post_to_slack(state, "product", f"PRD complete. See Notion: {parsed.get('notion_prd_url', '')}")

    new_state: dict = {
        "messages": [_make_message("product", "team", "PRD complete.", "update")],
        "notion_prd_url": parsed.get("notion_prd_url", ""),
        "prd_content": parsed.get("prd_content", output),
    }

    if parsed.get("clarification_answer"):
        new_state["pending_clarification"] = {
            **state["pending_clarification"],
            "answer": parsed["clarification_answer"],
            "answered": True,
        }
        _post_to_slack(state, "product", f"Answering Dev's question: {parsed['clarification_answer']}")

    return new_state


# ── Dev Node ─────────────────────────────────────────────────────────────────

def dev_node(state: ProjectState) -> dict:
    """
    Lead Dev writes real code, runs it in E2B sandbox, commits to GitHub.
    Can ask Product for clarification (sets pending_clarification).
    """
    qa_feedback = state.get("qa_feedback", "")
    pending = state.get("pending_clarification") or {}

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
    close_sandbox()

    parsed = _parse_json_from_output(output)

    new_state: dict = {
        "messages": [_make_message("dev", "team", parsed.get("summary", "Dev work complete."), "update")],
        "arch_content": parsed.get("summary", state.get("arch_content", "")),
        "qa_feedback": "",
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


# ── QA Node ──────────────────────────────────────────────────────────────────

def qa_node(state: ProjectState) -> dict:
    """QA pulls code from GitHub, writes tests, runs them in sandbox. Approves or sends bugs back to Dev."""
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

    parsed = _parse_json_from_output(output)

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


# ── CFO Node ─────────────────────────────────────────────────────────────────

def cfo_node(state: ProjectState) -> dict:
    """CFO creates financial plan and saves to Notion."""
    prompt = f"""
Project: "{state.get('project_name')}"
Idea: "{state['idea']}"
PRD summary: {state.get('prd_content', 'N/A')[:2000]}

Your tasks:
1. Create a financial plan covering:
   - Estimated development costs
   - Monthly burn rate projection (6 months)
   - Revenue model and pricing strategy
   - Break-even analysis
   - Fundraising recommendation (bootstrap vs. seed round)
2. Save to Notion as "Financial Plan — {state.get('project_name', 'Project')}"
3. Post highlights to Slack.

Output:
```json
{{"notion_financial_url": "https://notion.so/...", "summary": "Key financial highlights"}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS
    output = _run_agent("cfo", tools, prompt)
    parsed = _parse_json_from_output(output)

    _post_to_slack(state, "cfo", f"Financial plan complete. {parsed.get('summary', '')}")

    return {
        "notion_financial_url": parsed.get("notion_financial_url", ""),
        "messages": [_make_message("cfo", "team", "Financial plan complete.", "update")],
    }


# ── Marketing Node ───────────────────────────────────────────────────────────

def marketing_node(state: ProjectState) -> dict:
    """Marketing creates GTM strategy and saves to Notion."""
    prompt = f"""
Project: "{state.get('project_name')}"
Idea: "{state['idea']}"
PRD summary: {state.get('prd_content', 'N/A')[:2000]}

Your tasks:
1. Create a go-to-market strategy covering:
   - Brand positioning and messaging
   - Target audience and ICP
   - Launch plan (pre-launch, launch day, post-launch)
   - Content strategy (blog, social, email)
   - Channel priorities and KPIs
2. Save to Notion as "GTM Strategy — {state.get('project_name', 'Project')}"
3. Post summary to Slack.

Output:
```json
{{"notion_gtm_url": "https://notion.so/...", "summary": "Key GTM highlights"}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS
    output = _run_agent("marketing", tools, prompt)
    parsed = _parse_json_from_output(output)

    _post_to_slack(state, "marketing", f"GTM strategy complete. {parsed.get('summary', '')}")

    return {
        "notion_gtm_url": parsed.get("notion_gtm_url", ""),
        "messages": [_make_message("marketing", "team", "GTM strategy complete.", "update")],
    }


# ── Sales Node ───────────────────────────────────────────────────────────────

def sales_node(state: ProjectState) -> dict:
    """Sales creates playbook and saves to Notion."""
    prompt = f"""
Project: "{state.get('project_name')}"
Idea: "{state['idea']}"
PRD summary: {state.get('prd_content', 'N/A')[:2000]}

Your tasks:
1. Create an initial sales strategy covering:
   - Ideal customer profile
   - Outreach templates (cold email + LinkedIn)
   - Objection handling playbook
   - Pricing and packaging recommendation
   - 90-day sales target
2. Create a leads database in Notion using notion_create_database with db_type='leads'.
3. Save playbook to Notion as "Sales Playbook — {state.get('project_name', 'Project')}"
4. Post summary to Slack.

Output:
```json
{{"notion_sales_url": "https://notion.so/...", "summary": "Key sales highlights"}}
```
"""

    tools = SLACK_TOOLS + NOTION_TOOLS
    output = _run_agent("sales", tools, prompt)
    parsed = _parse_json_from_output(output)

    _post_to_slack(state, "sales", f"Sales playbook complete. {parsed.get('summary', '')}")

    return {
        "notion_sales_url": parsed.get("notion_sales_url", ""),
        "messages": [_make_message("sales", "team", "Sales playbook complete.", "update")],
    }


# ── CEO Summary Node ─────────────────────────────────────────────────────────

def ceo_summary_node(state: ProjectState) -> dict:
    """CEO synthesizes all outputs into an executive summary."""
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
