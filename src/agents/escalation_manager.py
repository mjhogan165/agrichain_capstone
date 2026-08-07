from src.data.state import AgriChainState


def identify_escalation(state: AgriChainState) -> AgriChainState:
    """Escalation Manager, decides if a case needs human review."""

    severity = state.get("severity", "")
    error_message = state.get("error_message", "")

    if error_message:
        state["escalate"] = True
        state["escalation_reason"] = f"Unresolved: {error_message}"
    elif severity == "critical":  # <- threshold depends on your answer above
        state["escalate"] = True
        state["escalation_reason"] = "Critical severity complaint"
    else:
        state["escalate"] = False
        state["escalation_reason"] = ""

    return state
