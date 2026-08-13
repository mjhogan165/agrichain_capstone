from langchain_community.embeddings import SentenceTransformerEmbeddings

# Loaded once
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
