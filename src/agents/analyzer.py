from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.data.state import AgriChainState
from src.models.classify import predict_category


class SeverityAssessment(BaseModel):
    """Output schema for the Analyzer Agent"""

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="The assessed severity of the complaint"
    )
    reasoning: str = Field(
        description="Brief explanation of why this severity was assigned"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(SeverityAssessment)


def analyze_severity(state: AgriChainState) -> AgriChainState:
    """Analyzer agent. Predicts category, assess severity."""

    complaint_text = state.get("complaint_text", "")

    category = predict_category(complaint_text)
    prompt = f"""You are assessing the severity of a customer complaint for an agricultural supply chain company.

Complaint category: {category}
Complaint text: "{complaint_text}"

Rate the severity as low, medium, high, or critical. Consider whether perishable goods might spoil before this gets resolved, how much money is at stake, any safety or contamination risk, and whether this could damage the customer relationship.

Give your rating and a short reason why."""

    result = structured_llm.invoke(prompt)
    assert isinstance(result, SeverityAssessment)

    state["severity"] = result.severity
    state["severity_reasoning"] = result.reasoning
    state["predicted_category"] = category

    return state
