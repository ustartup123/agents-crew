"""
graph/standup_graph.py — Daily standup: each agent posts status to Slack, CEO summarizes.
Saves summary to Notion. No full LangGraph graph needed — sequential function calls.
"""

import logging
from datetime import datetime

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent

from config.settings import gemini_cfg, slack_cfg, notion_cfg
from agents import AGENT_PERSONAS
from tools.slack_tools import SLACK_TOOLS
from tools.notion_tools import NOTION_TOOLS

logger = logging.getLogger(__name__)

STANDUP_AGENTS = ["product", "dev", "qa", "marketing", "sales", "cfo"]


def _build_llm():
    return ChatGoogleGenerativeAI(
        model=gemini_cfg.model,
        google_api_key=gemini_cfg.api_key,
        temperature=0.5,
    )


def run_daily_standup():
    """Run the daily standup: each agent queries Notion and posts to Slack. CEO summarizes."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    channel = slack_cfg.channel_standup or slack_cfg.channel_general

    logger.info(f"Running daily standup for {today}")

    llm = _build_llm()
    tools = SLACK_TOOLS + NOTION_TOOLS

    for agent_key in STANDUP_AGENTS:
        persona = AGENT_PERSONAS[agent_key]
        system_prompt = f"You are the {persona['role']}. {persona['backstory']}"
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
            agent = create_react_agent(llm, tools, state_modifier=system_prompt)
            agent.invoke({"messages": [("user", prompt)]})
        except Exception as e:
            logger.error(f"Standup error for {agent_key}: {e}")

    # CEO synthesizes
    ceo_persona = AGENT_PERSONAS["ceo"]
    ceo_system = f"You are the {ceo_persona['role']}. {ceo_persona['backstory']}"
    ceo_prompt = (
        f"Read the standup updates just posted to Slack channel {channel}. "
        f"Post a brief CEO synthesis: top team priorities today, any blockers needing your attention. "
        f"Then save a standup summary to Notion (create a page titled 'Standup {today}' "
        f"under root page {notion_cfg.root_page_id})."
    )
    try:
        agent = create_react_agent(llm, tools, state_modifier=ceo_system)
        agent.invoke({"messages": [("user", ceo_prompt)]})
    except Exception as e:
        logger.error(f"CEO standup synthesis error: {e}")

    logger.info("Daily standup complete.")


def run_weekly_review():
    """Weekly review: each agent posts weekly status to Slack + saves to Notion."""
    today = datetime.now().strftime("%A, %B %d, %Y")
    channel = slack_cfg.channel_general

    logger.info(f"Running weekly review for {today}")

    llm = _build_llm()
    tools = SLACK_TOOLS + NOTION_TOOLS

    review_agents = {
        "product": "weekly product review: features shipped, backlog changes, user feedback",
        "dev": "weekly engineering report: code shipped, tech debt, architecture decisions, deployments",
        "qa": "weekly QA report: tests written/passed/failed, bugs found/fixed, quality assessment",
        "cfo": "weekly financial snapshot: spend, runway, revenue milestones, fundraising status",
        "marketing": "weekly marketing report: content published, metrics, campaigns, upcoming launches",
        "sales": "weekly sales pipeline report: new leads, deals advanced/lost, pipeline value",
    }

    for agent_key, focus in review_agents.items():
        persona = AGENT_PERSONAS[agent_key]
        system_prompt = f"You are the {persona['role']}. {persona['backstory']}"
        prompt = (
            f"Prepare a {focus}. "
            f"Query Notion for relevant data. Post to Slack channel {channel}. "
            f"Save a detailed version to Notion as '{persona['role']} Weekly Review — {today}'."
        )
        try:
            agent = create_react_agent(llm, tools, state_modifier=system_prompt)
            agent.invoke({"messages": [("user", prompt)]})
        except Exception as e:
            logger.error(f"Weekly review error for {agent_key}: {e}")

    # CEO synthesis
    ceo_persona = AGENT_PERSONAS["ceo"]
    ceo_system = f"You are the {ceo_persona['role']}. {ceo_persona['backstory']}"
    ceo_prompt = (
        f"Read the weekly reports from all team members in Slack channel {channel}. "
        f"Synthesize into a 'Weekly CEO Report — {today}' Notion page. "
        f"Include: executive summary, key wins, risks/blockers, decisions needed, next week priorities. "
        f"Post a concise summary to Slack."
    )
    try:
        agent = create_react_agent(llm, tools, state_modifier=ceo_system)
        agent.invoke({"messages": [("user", ceo_prompt)]})
    except Exception as e:
        logger.error(f"CEO weekly synthesis error: {e}")

    logger.info("Weekly review complete.")
