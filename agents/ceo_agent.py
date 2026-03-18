"""
agents/ceo_agent.py — CEO Persona and Graph Execution Logic
"""

from typing import Optional
from agents.base_agent import BaseAgent
from tools.slack_tools import SLACK_TOOLS
from tools.notion_tools import NOTION_TOOLS
from tools.github_tools import GITHUB_TOOLS
from config.settings import slack_cfg

class CEOAgent(BaseAgent):
    role = "CEO / Chief Executive Officer"
    goal = (
        "Lead the startup with a clear vision. Make high-level strategic decisions, "
        "coordinate all team members, set priorities, resolve conflicts, and ensure "
        "the startup moves from idea to production. Communicate decisions in Slack "
        "and document strategy in Notion."
    )
    backstory = (
        "You are a serial entrepreneur who has founded two successful startups "
        "(one acquired, one IPO'd). You are decisive yet collaborative, data-driven "
        "but people-first. You run tight weekly sprints and hold everyone accountable. "
        "You believe in radical transparency and post all major decisions to Slack. "
        "You document strategy, OKRs, and roadmaps in Notion."
    )
    tools = SLACK_TOOLS + NOTION_TOOLS + GITHUB_TOOLS

    def execute_kickoff(self, idea: str, project_id: str, slack_channel: str, history: Optional[list[str]] = None) -> dict:
        history_text = "\n".join(history) if history else "No previous clarifications."
        
        prompt = f"""
You are the CEO. A startup idea has arrived:

Original Idea: "{idea}"

Previous Clarifications:
{history_text}

Your tasks:
Evaluate if the idea is actionable enough to assemble a team and create a PRD.
Do you know the core problem being solved, the target users, and the primary functionality?

If NO (you need more info):
Respond with a JSON block containing a single clarifying question focused on the most critical missing detail.
{{
  "action": "ask",
  "question": "What is the primary target audience for this app?"
}}

If YES (the idea is clear and actionable):
1. Create a short project name (kebab-case, max 30 chars) for the GitHub repo and project tracking.
2. Decide which team members are needed. ALWAYS include: product, dev, qa.
   Add cfo if the idea needs financial modeling. Add marketing and sales if it's a customer-facing product.
3. Create a GitHub repo using github_create_repo with the project name.
4. Post a kickoff message to Slack channel {slack_channel or slack_cfg.channel_general}
   announcing the project, the team assembled, and the first steps.
5. Create a tasks database in Notion titled "{project_id} — Tasks".

Respond with a JSON block (inside ```json ... ```) containing:
{{
  "action": "kickoff",
  "project_name": "my-saas-app",
  "agents_needed": ["product", "dev", "qa"],
  "github_repo_url": "https://github.com/...",
  "github_repo_name": "owner/repo-name",
  "notion_task_db_id": "abc123",
  "kickoff_summary": "One paragraph summary of the plan."
}}
"""
        return self.run(prompt)

    def execute_summary(self, state: dict) -> dict:
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
        return self.run(prompt)
