from src.data.state import AgriChainState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class CustomerResponse(BaseModel):
    """Output schema for the Communicator Agent"""

    response: str = Field(description="The drafted customer response")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(CustomerResponse)


def draft_customer_response(state: AgriChainState) -> AgriChainState:
    """Communicator Agent, drafts a customer response"""

    severity = state.get("severity", "")
    resolution_plan = state.get("resolution_plan", {})
    complaint_text = state.get("complaint_text", "")

    if not resolution_plan:
        state["error_message"] = "No resolution plan available to draft a response."
        state["customer_response"] = (
            "Thank you for reaching out. We've received your complaint and it is "
            "currently under review. We'll follow up with next steps shortly."
        )
        return state

    prompt = f"""You are tasked with drafting a customer response for a complaint in an agricultural supply chain company.
Complaint text: "{complaint_text}"
Complaint severity: {severity}
Resolution plan: {resolution_plan}

Write a short, professional reply to the customer. Acknowledge their issue, explain what will be done to fix it, and keep the tone polite."""

    result = structured_llm.invoke(prompt)
    state["customer_response"] = result.response  # type: ignore
    return state
