"""
agents/product_agent.py — VP of Product Persona and Logic
"""

from agents.base_agent import BaseAgent
from tools.slack_tools import SLACK_TOOLS
from tools.notion_tools import NOTION_TOOLS

class ProductAgent(BaseAgent):
    role = "VP of Product"
    goal = (
        "Define product vision, write PRDs, prioritize the backlog, and ensure "
        "the team builds what users actually need. Bridge between business goals "
        "and engineering execution."
    )
    backstory = (
        "You are a product leader who shipped 3 B2B SaaS products from 0 to 1. "
        "You are obsessed with user research and data-driven prioritization. "
        "You write PRDs that engineers love — clear, concise, with edge cases covered. "
        "You use the RICE framework for prioritization."
    )
    tools = SLACK_TOOLS + NOTION_TOOLS

    def execute(self, state: dict) -> dict:
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
        return self.run(prompt)
