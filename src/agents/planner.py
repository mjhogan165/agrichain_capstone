from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.data.state import AgriChainState


class ResolutionPlan(BaseModel):
    """Schema for the Planner Agent"""

    plan_summary: str = Field(description="A summary of the resolution plan")
    steps: list[str] = Field(
        description="A list of actionable steps to resolve the complaint"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(ResolutionPlan)


def plan_resolution(state: AgriChainState) -> AgriChainState:
    """Planner agent, generates a resolution plan"""

    severity = state.get("severity", "")
    retrieved_documents = state.get("retrieved_documents", [])
    complaint_text = state.get("complaint_text", "")

    prompt = f"""You are to create a resolution plan for a customer complaint in an agricultural supply chain company.
Complaint text: "{complaint_text}"
Complaint severity: {severity}
Retrieved documents: {retrieved_documents}

Based on the severity and the retrieved documents, write a resolution plan for this complaint. Include a short summary and a list of steps to fix the issue."""
    result = structured_llm.invoke(prompt)
    assert isinstance(result, ResolutionPlan)

    result_dict = {
        "plan_summary": result.plan_summary,
        "steps": result.steps,
    }
    state["resolution_plan"] = result_dict

    return state
