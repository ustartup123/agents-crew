from agents.ceo_agent import CEOAgent
from agents.product_agent import ProductAgent
from agents.dev_agent import DevAgent
from agents.qa_agent import QAAgent
from agents.business_agents import CFOAgent, MarketingAgent, SalesAgent

AGENT_PERSONAS = {
    "ceo": {"role": CEOAgent.role, "goal": CEOAgent.goal, "backstory": CEOAgent.backstory},
    "product": {"role": ProductAgent.role, "goal": ProductAgent.goal, "backstory": ProductAgent.backstory},
    "dev": {"role": DevAgent.role, "goal": DevAgent.goal, "backstory": DevAgent.backstory},
    "qa": {"role": QAAgent.role, "goal": QAAgent.goal, "backstory": QAAgent.backstory},
    "cfo": {"role": CFOAgent.role, "goal": CFOAgent.goal, "backstory": CFOAgent.backstory},
    "marketing": {"role": MarketingAgent.role, "goal": MarketingAgent.goal, "backstory": MarketingAgent.backstory},
    "sales": {"role": SalesAgent.role, "goal": SalesAgent.goal, "backstory": SalesAgent.backstory},
}

__all__ = ["AGENT_PERSONAS"]
