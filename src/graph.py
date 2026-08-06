from langgraph.graph import StateGraph, END

from src.data.state import AgriChainState
from src.agents.analyzer import analyze_severity
from src.agents.investigator import investigate
from src.agents.planner import plan_resolution
from src.agents.communicator import draft_customer_response
from src.agents.escalation_manager import identify_escalation

# Build the graph
graph = StateGraph(AgriChainState)

# Register each function as a node
graph.add_node("analyzer", analyze_severity)
graph.add_node("investigator", investigate)
graph.add_node("planner", plan_resolution)
graph.add_node("communicator", draft_customer_response)
graph.add_node("escalation_manager", identify_escalation)

# Wire them in sequence
graph.set_entry_point("analyzer")
graph.add_edge("analyzer", "investigator")
graph.add_edge("investigator", "planner")
graph.add_edge("planner", "communicator")
graph.add_edge("communicator", "escalation_manager")
graph.add_edge("escalation_manager", END)

# Compile it
app = graph.compile()
