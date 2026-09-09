from typing import Dict, Any, TypedDict
from pydantic import BaseModel


# ================= Structured Output Schema ================


class LeadResponse(BaseModel):
    LeadID: str
    FullName: str
    Score: float
    Qualification: str  # Hot | Warm | Cold | Disqualified
    Reason: str
    Email: str
    Phone: str
    ProductsOfInterest: str
    Existing: bool
    CatchPhrase: str


# =================== Graph State ===============


class LeadState(TypedDict):
    payload: Dict[str, Any]
    output: Dict[str, Any]


# ================== State Initializer ================


def _initialize_state(payload: Dict[str, Any]) -> LeadState:
    return {
        "payload": payload,
        "output": {},
    }
