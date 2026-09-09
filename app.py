import logging
from fastapi import FastAPI, HTTPException
from pydantic import BaseModel
from typing import Any, Dict

from agent.leads_agent import LeadAgent, RunLeadAgent

# ─── Logging ────────────────────────────────────────────────────────────────

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# ─── App ─────────────────────────────────────────────────────────────────────

app = FastAPI(
    title="Leads RPA — Sales Lead Qualifier",
    description="Single-node LangGraph agent that scores and qualifies inbound sales leads using Gemini.",
    version="1.0.0",
)

# Agent instance (initialised on first request — keeps it simple, no lifespan)
_lead_agent: LeadAgent | None = None


async def _get_agent() -> LeadAgent:
    """Lazily initialise the agent on first call."""
    global _lead_agent
    if _lead_agent is None:
        logger.info("Initialising LeadAgent…")
        _lead_agent = await LeadAgent.build()
        logger.info("LeadAgent ready.")
    return _lead_agent


# ─── Request / Response schemas ───────────────────────────────────────────────


class LeadPayload(BaseModel):
    """
    Pass any lead fields the robot has collected.
    All fields are optional — include whatever is available.
    """

    payload: Dict[str, Any]


class QualificationResult(BaseModel):
    result: Dict[str, Any]


# ─── Routes ──────────────────────────────────────────────────────────────────


@app.post("/qualify", response_model=QualificationResult)
async def qualify_lead(request: LeadPayload):
    """
    Qualify a sales lead.

    Send a JSON body with a `payload` key containing the lead data dict.
    Returns the full LeadResponse structured output.
    """
    try:
        agent = await _get_agent()
        runner = RunLeadAgent(agent)
        result = await runner.run(request.payload)
        return {"result": result}
    except Exception as e:
        logger.error(f"Error qualifying lead: {e}")
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/health")
async def health():
    return {"status": "✅ Leads RPA is running"}


# ─── Entry point ─────────────────────────────────────────────────────────────

if __name__ == "__main__":
    import uvicorn

    uvicorn.run("app:app", host="0.0.0.0", port=8000, reload=True, log_level="info")
