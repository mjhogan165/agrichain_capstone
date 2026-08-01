# src/agents/analyzer.py
from typing import Literal

from pydantic import BaseModel, Field
from langchain_openai import ChatOpenAI

from src.data.state import AgriChainState
from src.models.classify import predict_category


class SeverityAssessment(BaseModel):
    """Structured output schema for the Analyzer agent's LLM call."""

    severity: Literal["low", "medium", "high", "critical"] = Field(
        description="The assessed severity of the complaint"
    )
    reasoning: str = Field(
        description="Brief explanation of why this severity was assigned"
    )


llm = ChatOpenAI(model="gpt-4o-mini", temperature=0)
structured_llm = llm.with_structured_output(SeverityAssessment)


def analyze_severity(state: AgriChainState) -> AgriChainState:
    """Analyzer agent: predicts category, then uses it + the complaint text to assess severity."""

    complaint_text = state.get("complaint_text", "")

    # Step 1: use the trained classifier for category context
    category = predict_category(complaint_text)

    # Step 2: build a prompt combining the complaint text and the predicted category
    prompt = f"""You are assessing the severity of a customer complaint for an agricultural supply chain company.

Complaint category: {category}
Complaint text: "{complaint_text}"

Assess the severity as low, medium, high, or critical, based on factors like:
- Whether perishable goods are at risk of spoiling before resolution
- The scale of financial impact
- Any safety or contamination concerns
- How much the customer relationship is at risk

Provide your severity rating and a brief reasoning."""

    # Step 3: call the LLM, get back a validated SeverityAssessment object
    result = structured_llm.invoke(prompt)
    assert isinstance(
        result, SeverityAssessment
    )  # we know this, because we passed a Pydantic schema in
    # Step 4: write the result into state
    state["severity"] = result.severity
    state["severity_reasoning"] = result.reasoning
    return state
