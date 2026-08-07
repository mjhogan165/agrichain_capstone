from langchain_community.embeddings import SentenceTransformerEmbeddings

# Loaded ONCE, shared by both classify.py and retrieve.py,
# so we don't initialize the same torch model twice
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
