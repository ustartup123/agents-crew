"""
agents/base_agent.py — Base class for startup agents.
"""

import json
import logging
import re
from typing import Any

from langchain_google_genai import ChatGoogleGenerativeAI
from langgraph.prebuilt import create_react_agent
from config.settings import gemini_cfg

logger = logging.getLogger(__name__)

def _build_llm():
    return ChatGoogleGenerativeAI(
        model=gemini_cfg.model,
        google_api_key=gemini_cfg.api_key,
        temperature=gemini_cfg.temperature,
    )

class BaseAgent:
    role: str = ""
    goal: str = ""
    backstory: str = ""
    tools: list = []
    
    def __init__(self):
        self.llm = _build_llm()

    def _parse_json(self, output: str) -> dict:
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse JSON from agent output")
        return {}

    def run(self, prompt_text: str) -> dict:
        system_prompt = (
            f"You are the {self.role} of a startup.\n"
            f"Your goal: {self.goal}\n"
            f"Your background: {self.backstory}\n\n"
            f"Task: {prompt_text}\n\n"
            f"Use your tools as needed. Be decisive and action-oriented."
        )

        agent = create_react_agent(self.llm, self.tools, state_modifier=system_prompt)
        try:
            result = agent.invoke({"messages": [("user", prompt_text)]})
            ai_messages = [m for m in result.get("messages", []) if getattr(m, "type", "") == "ai"]
            output = getattr(ai_messages[-1], "content", str(result)) if ai_messages else str(result)
            return self._parse_json(output)
        except Exception as e:
            logger.error(f"Agent execution error ({self.role}): {e}")
            return {}
