import streamlit as st
import sys
from pathlib import Path

# Add project root to sys.path so `from src...` imports work,
# regardless of where Streamlit's working directory happens to be
project_root = Path(__file__).resolve().parent.parent
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

from src.rag.retrieve import retrieve_relevant_documents
from src.data.state import AgriChainState

st.title("AgriChain RAG Knowledge Assistant")
st.write(
    "Search the knowledge base for relevant supplier info, SOPs, and resolution guides."
)

query = st.text_area("Enter a query or complaint")

if st.button("Search"):
    if query.strip():
        state: AgriChainState = {"complaint_text": query}
        state = retrieve_relevant_documents(state)
        for doc in state.get("retrieved_documents", []):
            st.write(f"**{doc['doc_type']}** (score: {doc['score']:.3f})")
            st.write(doc["content"])
            st.divider()
    else:
        st.warning("Please enter a query first.")
