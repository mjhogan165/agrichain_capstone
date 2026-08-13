import pickle
from pathlib import Path

import numpy as np
from tensorflow import keras

PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent
CLASSIFIER_PATH = PROJECT_ROOT / "models" / "complaint_classifier.keras"
LABEL_ENCODER_PATH = PROJECT_ROOT / "models" / "label_encoder.pkl"

from src.models.embeddings import embedding_model

# Loaded once at import time
classifier_model = keras.models.load_model(str(CLASSIFIER_PATH))

with open(LABEL_ENCODER_PATH, "rb") as f:
    label_encoder = pickle.load(f)


def predict_category(complaint_text: str) -> str:
    """Predicts the category using the already trained ANN classifier."""

    embedding = embedding_model.embed_query(complaint_text)
    embedding_batch = np.array([embedding])
    probabilities = classifier_model.predict(embedding_batch)
    predicted_index = np.argmax(probabilities[0])

    category = label_encoder.inverse_transform([predicted_index])[0]

    return category
