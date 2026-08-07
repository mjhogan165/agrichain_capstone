from src.data.state import AgriChainState
from langchain_openai import ChatOpenAI
from pydantic import BaseModel, Field


class CustomerResponse(BaseModel):
    """Structured output schema for the Communicator agent's LLM call."""

    response: str = Field(description="The drafted customer response")


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(CustomerResponse)


def draft_customer_response(state: AgriChainState) -> AgriChainState:
    """Communicator Agent, drafts a customer response from the complaint, severity, and resolution plan."""

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

    # Build a prompt combining the complaint text, severity, and resolution plan
    prompt = f"""You are tasked with drafting a customer response for a complaint in an agricultural supply chain company.
Complaint text: "{complaint_text}"
Complaint severity: {severity}
Resolution plan: {resolution_plan}

Draft a professional and empathetic response to the customer, addressing their concerns and outlining the steps that will be taken to resolve the issue. Ensure the response is clear, concise, and provides the customer with a sense of being heard and valued.
    """

    result = structured_llm.invoke(prompt)
    state["customer_response"] = result.response  # type: ignore
    return state
