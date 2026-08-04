from src.data.state import AgriChainState
from src.rag.retrieve import retrieve_relevant_documents


def investigate(state: AgriChainState) -> AgriChainState:
    """Investigator agent: retrieves relevant knowledge base documents for the complaint."""
    return retrieve_relevant_documents(state)
