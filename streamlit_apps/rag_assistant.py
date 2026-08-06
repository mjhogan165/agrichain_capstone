import streamlit as st

from pathlib import Path

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
