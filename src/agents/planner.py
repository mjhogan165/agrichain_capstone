from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.data.state import AgriChainState


class ResolutionPlan(BaseModel):
    """Schema for the Planner agent's LLM call."""

    plan_summary: str = Field(description="A summary of the resolution plan")
    steps: list[str] = Field(
        description="A list of actionable steps to resolve the complaint"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ResolutionPlan)


def plan_resolution(state: AgriChainState) -> AgriChainState:
    """Planner agent, generates a resolution plan from the complaint, severity, and retrieved documents."""

    severity = state.get("severity", "")
    retrieved_documents = state.get("retrieved_documents", [])
    complaint_text = state.get("complaint_text", "")

    #  Build a prompt combining the complaint text, severity, and retrieved documents
    prompt = f"""You are tasked with creating a resolution plan for a customer complaint in an agricultural supply chain company.
Complaint text: "{complaint_text}"
Complaint severity: {severity}
Retrieved documents: {retrieved_documents}

Given the severity of the complaint and the information from the retrieved documents, create a detailed resolution plan that addresses the customer's concerns, outlines steps to resolve the issue, and includes measures to prevent similar issues in the future. Provide your response in a structured format.
    """
    result = structured_llm.invoke(prompt)
    assert isinstance(
        result, ResolutionPlan
    )  # we know this, because we passed a Pydantic schema in

    # Write the result into state
    result_dict = {
        "plan_summary": result.plan_summary,
        "steps": result.steps,
    }
    state["resolution_plan"] = result_dict

    return state
