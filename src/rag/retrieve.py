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
    """Retrieves relevant knowledge base documents for the complaint using FAISS vector search.

    Over-fetches, then picks the best-matching doc per doc_type, so the Investigator
    always gets one resolution_guide, one sop, and one supplier_info doc — not
    whatever 5 happen to be closest overall, which skews toward supplier_info
    since it's 73% of the knowledge base.
    """
    start = time.time()
    complaint_text = state.get("complaint_text", "")
    results = faiss_index.similarity_search_with_score(complaint_text, k=274)

    all_docs = [
        {"content": doc.page_content, "score": score, **doc.metadata}
        for doc, score in results
    ]

    # Lower score = more similar
    best_by_type = {}
    for doc in all_docs:
        doc_type = doc.get("doc_type", "unknown")
        if (
            doc_type not in best_by_type
            or doc["score"] < best_by_type[doc_type]["score"]
        ):
            best_by_type[doc_type] = doc

    retrieved_documents = list(best_by_type.values())
    state["retrieved_documents"] = retrieved_documents
    print(f"Retrieval took {time.time() - start:.2f} seconds")
    return state
