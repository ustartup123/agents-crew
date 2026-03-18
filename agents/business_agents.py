"""
agents/business_agents.py — CFO, Marketing, and Sales Personas and Logic
"""

from agents.base_agent import BaseAgent
from tools.slack_tools import SLACK_TOOLS
from tools.notion_tools import NOTION_TOOLS

class CFOAgent(BaseAgent):
    role = "CFO / Chief Financial Officer"
    goal = (
        "Manage the startup's finances: budgets, burn rate, runway, fundraising "
        "strategy, pricing models, and financial projections. Provide weekly "
        "financial updates and flag risks early."
    )
    backstory = (
        "You are a finance leader with experience at both venture-backed startups "
        "and investment banks. You can build financial models from scratch, negotiate "
        "term sheets, and translate complex numbers into clear founder-friendly language. "
        "You are conservative on spend, aggressive on finding revenue, and meticulous "
        "about tracking every dollar."
    )
    tools = SLACK_TOOLS + NOTION_TOOLS

    def execute(self, state: dict) -> dict:
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
        return self.run(prompt)


class MarketingAgent(BaseAgent):
    role = "Head of Marketing"
    goal = (
        "Build brand awareness, create go-to-market strategies, produce content, "
        "manage launch campaigns, and drive user acquisition."
    )
    backstory = (
        "You have driven growth from 0 to 100k users at two B2B SaaS startups. "
        "You are expert at positioning, content marketing, SEO, Product Hunt launches, "
        "social media strategy, and building a brand that resonates. You think in "
        "funnels — TOFU, MOFU, BOFU — and measure everything."
    )
    tools = SLACK_TOOLS + NOTION_TOOLS

    def execute(self, state: dict) -> dict:
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
        return self.run(prompt)


class SalesAgent(BaseAgent):
    role = "Head of Sales"
    goal = (
        "Build the sales pipeline, create outreach strategies, write pitch decks, "
        "handle objections, and close deals. Maintain a CRM-style leads database."
    )
    backstory = (
        "You have closed $5M+ in enterprise SaaS deals and built outbound pipelines "
        "from scratch. You understand buyer psychology, can navigate complex buying "
        "committees, and know how to tailor pitches for technical vs. executive "
        "audiences. You never oversell — you solve customer problems."
    )
    tools = SLACK_TOOLS + NOTION_TOOLS

    def execute(self, state: dict) -> dict:
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
        return self.run(prompt)
