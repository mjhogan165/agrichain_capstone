import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras
from langchain_community.embeddings import SentenceTransformerEmbeddings

# Compute paths relative to this file's own location, not wherever the code happens to run from
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "complaint_classifier.keras"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

# Load everything ONCE, when this file is first imported — not inside the function
embedding_model = SentenceTransformerEmbeddings(model_name="all-MiniLM-L6-v2")
classifier_model = keras.models.load_model(str(CLASSIFIER_PATH))


with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def predict_category(complaint_text: str) -> str:
    """Predicts the complaint category using the trained ANN classifier."""

    # 1. Turn the raw text into a 384-number vector
    embedding = embedding_model.embed_query(complaint_text)

    # 2. Keras expects a batch — wrap the single embedding so shape is (1, 384), not (384,)
    embedding_batch = np.array([embedding])

    # 3. Run the trained network — output is (1, 8): one row, 8 category probabilities
    probabilities = classifier_model.predict(embedding_batch)

    # 4. Find which of the 8 positions has the highest probability
    predicted_index = np.argmax(probabilities[0])

    # 5. Turn that index back into an actual category name
    category = label_encoder.inverse_transform([predicted_index])[0]

    return category
