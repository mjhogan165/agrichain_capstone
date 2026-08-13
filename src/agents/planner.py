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


def format_documents_for_prompt(documents: list[dict]) -> str:
    if not documents:
        return "No relevant documents were retrieved."

    lines = []
    for doc in documents:
        score = doc.get("score", 0)
        if score < 1.1:
            confidence = "strong match"
        elif score < 1.3:
            confidence = "moderate match"
        else:
            confidence = "weak match, likely not directly relevant"

        lines.append(
            f"[{doc.get('doc_type', 'unknown')}] "
            f"(relevance score: {score:.3f} - {confidence})\n"
            f"{doc.get('content', '')}"
        )
    return "\n\n".join(lines)


def plan_resolution(state: AgriChainState) -> AgriChainState:
    """Planner agent, generates a resolution plan"""

    severity = state.get("severity", "")
    retrieved_documents = state.get("retrieved_documents", [])
    complaint_text = state.get("complaint_text", "")

    formatted_documents = format_documents_for_prompt(retrieved_documents)

    prompt = f"""You are to create a resolution plan for a customer complaint in an agricultural supply chain company.
Complaint text: "{complaint_text}"
Complaint severity: {severity}

Retrieved documents:
{formatted_documents}

Each document is labeled with a match confidence. Treat "strong match" documents
as reliable. Treat "moderate match" documents as useful context, but don't
present them as a direct answer. Treat "weak match" documents as background
only, and do not build resolution steps around them. If no document is a
strong or moderate match, say so in your plan rather than inventing steps
that aren't supported by the retrieved documents or general standard practice.

Based on the severity and the retrieved documents, write a resolution plan for this complaint. Include a short summary and a list of steps to fix the issue."""
    result = structured_llm.invoke(prompt)
    assert isinstance(result, ResolutionPlan)

    result_dict = {
        "plan_summary": result.plan_summary,
        "steps": result.steps,
    }
    state["resolution_plan"] = result_dict

    return state
