from src.data.state import AgriChainState
from langchain_community.vectorstores import FAISS
from pathlib import Path
import time

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAISS_INDEX_PATH = str(PROJECT_ROOT / "models" / "faiss_knowledge_base")

from src.models.embeddings import embedding_model

faiss_index = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,  # LangChain requires this
)


def retrieve_relevant_documents(state: AgriChainState) -> AgriChainState:
    """Retrieves relevant knowledge base documents for the complaint using FAISS vector search."""

    start = time.time()
    complaint_text = state.get("complaint_text", "")
    results = faiss_index.similarity_search_with_score(complaint_text, k=5)
    retrieved_documents = [
        {"content": doc.page_content, "score": score, **doc.metadata}
        for doc, score in results
    ]
    state["retrieved_documents"] = retrieved_documents
    print(f"Retrieval took {time.time() - start:.2f} seconds")
    return state
