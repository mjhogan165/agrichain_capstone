from src.data.state import AgriChainState
from src.rag.retrieve import retrieve_relevant_documents


def investigate(state: AgriChainState) -> AgriChainState:
    """Investigator agent, gets the relevant knowledge base documents"""
    return retrieve_relevant_documents(state)
