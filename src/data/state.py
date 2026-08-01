from typing import TypedDict


class AgriChainState(TypedDict, total=False):
    """
    Defines the information that moves through the LangGraph workflow.
    """

    complaint_text: str
    severity: str
    severity_reasoning: str  # new field
    retrieved_documents: list[dict]
    resolution_plan: dict
    customer_response: str
    error_message: str
