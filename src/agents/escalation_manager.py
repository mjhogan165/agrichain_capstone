from src.data.state import AgriChainState


def identify_escalation(state: AgriChainState) -> AgriChainState:
    """Escalation Manager, flags if it needs human review."""

    severity = state.get("severity", "")
    error_message = state.get("error_message", "")

    if error_message:
        state["escalate"] = True
        state["escalation_reason"] = error_message
    elif severity == "critical":
        state["escalate"] = True
        state["escalation_reason"] = "Critical severity"
    else:
        state["escalate"] = False
        state["escalation_reason"] = ""

    return state
