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
        include_thoughts=True,
    )

class BaseAgent:
    role: str = ""
    goal: str = ""
    backstory: str = ""
    tools: list = []
    
    def __init__(self):
        self.llm = _build_llm()

    def _parse_json(self, output: str) -> dict:
        """Extract a JSON dict from the agent's text output.

        Tries multiple strategies in order:
        1. Direct parse (agent returned pure JSON)
        2. Fenced ```json ... ``` block
        3. Fenced ``` ... ``` block (no language tag)
        4. First bare {...} object in the text (greedy, supports nesting)
        Falls back to {"raw_output": output} so callers never get an
        empty dict that silently hides all agent work.
        """
        # Strategy 0: Direct parse — the whole output is valid JSON
        stripped = output.strip()
        if stripped.startswith("{"):
            try:
                return json.loads(stripped)
            except json.JSONDecodeError:
                pass

        # Strategy 1: ```json ... ```
        json_match = re.search(r'```json\s*(.*?)\s*```', output, re.DOTALL)
        if json_match:
            try:
                return json.loads(json_match.group(1))
            except json.JSONDecodeError:
                logger.warning("Failed to parse ```json block, trying fallbacks")

        # Strategy 2: ``` ... ``` (any fenced block)
        fence_match = re.search(r'```\s*(.*?)\s*```', output, re.DOTALL)
        if fence_match:
            try:
                return json.loads(fence_match.group(1))
            except json.JSONDecodeError:
                pass

        # Strategy 3: Greedy nested JSON object extraction
        # Find the first { and match to its balanced closing }
        start = output.find("{")
        if start != -1:
            depth = 0
            for i in range(start, len(output)):
                if output[i] == "{":
                    depth += 1
                elif output[i] == "}":
                    depth -= 1
                    if depth == 0:
                        try:
                            return json.loads(output[start:i + 1])
                        except json.JSONDecodeError:
                            break

        logger.warning("Could not parse JSON from agent output, returning raw output")
        return {"raw_output": output[:2000]}

    def run(self, prompt_text: str) -> dict:
        system_prompt = (
            f"You are the {self.role} of a startup.\n"
            f"Your goal: {self.goal}\n"
            f"Your background: {self.backstory}\n\n"
            f"Task: {prompt_text}\n\n"
            f"Use your tools as needed. Be decisive and action-oriented.\n"
            f"IMPORTANT: Your final response MUST contain a JSON object "
            f"wrapped in ```json ... ``` fences. Do not include any text "
            f"after the closing ``` fence."
        )

        agent = create_react_agent(self.llm, self.tools, prompt=system_prompt)
        try:
            result = agent.invoke({"messages": [("user", prompt_text)]})
            ai_messages = [m for m in result.get("messages", []) if getattr(m, "type", "") == "ai"]
            output = getattr(ai_messages[-1], "content", str(result)) if ai_messages else str(result)
            return self._parse_json(output)
        except Exception as e:
            logger.error(f"Agent execution error ({self.role}): {e}")
            return {}
