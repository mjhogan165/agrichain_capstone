from typing import TypedDict


class AgriChainState(TypedDict, total=False):

    complaint_text: str
    predicted_category: str
    severity: str
    severity_reasoning: str
    retrieved_documents: list[dict]
    resolution_plan: dict
    customer_response: str
    error_message: str
    escalate: bool
    escalation_reason: str
