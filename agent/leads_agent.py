import asyncio
import logging
import os
from typing import Dict, Any
from pathlib import Path

from dotenv import load_dotenv
from langchain_google_genai import ChatGoogleGenerativeAI
from langchain_core.messages import HumanMessage, SystemMessage
from langgraph.graph import StateGraph, END
import yaml

from agent.states import LeadState, LeadResponse, _initialize_state

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── Env & Config ────────────────────────────────────────────────────────────

load_dotenv()

_PROMPTS_PATH = Path(__file__).parent / "prompts.yaml"
with open(_PROMPTS_PATH, "r") as f:
    _prompts = yaml.safe_load(f)

LEAD_QUALIFIER_PROMPT: str = _prompts["LEAD_QUALIFIER_PROMPT"]


# ================= Lead Qualifier Agent ================


class LeadAgent:
    """
    Single-node LangGraph agent that qualifies sales leads.
    Uses Gemini as the LLM with structured output (LeadResponse).
    Factory pattern for clean async resource initialisation.
    """

    def __init__(self, llm, compiled_graph):
        self.llm = llm
        self.lead_agent = compiled_graph

    # ── Factory ──────────────────────────────────────────────────────────────

    @classmethod
    async def build(cls) -> "LeadAgent":
        llm = ChatGoogleGenerativeAI(
            model="gemini-3.6-flash",
            google_api_key=os.getenv("GEMINI_API_KEY"),
        )

        async def qualify_lead_node(state: LeadState):
            """The single qualifying node."""
            payload_text = str(state.get("payload", {}))

            messages = [
                SystemMessage(content=LEAD_QUALIFIER_PROMPT),
                HumanMessage(content=payload_text),
            ]

            response: LeadResponse = await llm.with_structured_output(
                LeadResponse
            ).ainvoke(messages)

            return {"output": response.model_dump()}

        # ── Build graph ───────────────────────────────────────────────────────

        graph_builder = StateGraph(LeadState)
        graph_builder.add_node("qualify_lead", qualify_lead_node)
        graph_builder.set_entry_point("qualify_lead")
        graph_builder.add_edge("qualify_lead", END)

        compiled_graph = graph_builder.compile()

        logger.info("LeadAgent graph compiled successfully.")
        return cls(llm, compiled_graph)


# ================= Runner ================


class RunLeadAgent:
    """Thin wrapper around the compiled graph for easy invocation."""

    def __init__(self, agent: LeadAgent):
        self.agent = agent.lead_agent

    async def run(self, payload: Dict[str, Any]) -> Dict[str, Any]:
        state = _initialize_state(payload)
        result = await self.agent.ainvoke(state)
        return result.get("output", {})
