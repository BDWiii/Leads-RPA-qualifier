"""
main.py — Quick smoke-test for the LeadAgent.
Run from the project root:  python main.py
"""

import asyncio
from agent.leads_agent import LeadAgent, RunLeadAgent

# ── Sample lead payload ───────────────────────────────────────────────────────

SAMPLE_LEAD = {
    "LeadID": "LEAD-001",
    "FullName": "Sara Mitchell",
    "Email": "sara.mitchell@techcorp.io",
    "Phone": "+1-555-234-5678",
    "JobTitle": "VP of Operations",
    "Company": "TechCorp Inc.",
    "Message": (
        "Hi, we are looking for a solution to automate our sales pipeline. "
        "We have a budget of around $8,000/year and want to get started within the next month. "
        "We came across your platform through a referral from a partner."
    ),
    "ProductsOfInterest": "Sales Automation, CRM Integration",
    "Existing": False,
    "WebsiteVisits": 6,
    "RequestedDemo": True,
}


# ── Test runner ───────────────────────────────────────────────────────────────


async def main():
    print("\nBuilding LeadAgent…")
    agent = await LeadAgent.build()

    runner = RunLeadAgent(agent)

    print("Sending sample lead for qualification…\n")
    result = await runner.run(SAMPLE_LEAD)

    print("Qualification Result:")
    for key, value in result.items():
        print(f"   {key}: {value}")


if __name__ == "__main__":
    asyncio.run(main())
