from src.data.state import AgriChainState
from langchain_community.vectorstores import FAISS
from langchain_community.embeddings import SentenceTransformerEmbeddings
from pathlib import Path
import time

# __file__ = the path to this retrieve.py file itself
# .parent three times: rag/ -> src/ -> project root
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
FAISS_INDEX_PATH = str(PROJECT_ROOT / "models" / "faiss_knowledge_base")

embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
faiss_index = FAISS.load_local(
    FAISS_INDEX_PATH,
    embeddings=embedding_model,
    allow_dangerous_deserialization=True,  # LangChain requires this explicit opt-in to unpickle a local index
)


def retrieve_relevant_documents(state: AgriChainState) -> AgriChainState:
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
