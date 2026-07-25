"""
preprocess_text.py
-------------------
Task 1: Text preprocessing for complaint_text, as required by the
project instructions - normalization and stopword removal.
"""

import string
import nltk
from nltk.corpus import stopwords

# NLTK's stopword list needs to be downloaded once per machine - this
# checks if it's already there first, so it doesn't re-download every
# time this file gets imported.
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def normalize_text(text: str) -> str:
    """
    Basic text normalization: lowercase + strip punctuation + collapse
    extra whitespace.
    """
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())  # collapses multiple spaces into one
    return text


def remove_stopwords(text: str) -> str:
    """
    Remove common English stopwords (e.g. 'the', 'is', 'for').
    """
    words = text.split()
    filtered = [w for w in words if w not in STOPWORDS]
    return " ".join(filtered)


if __name__ == "__main__":
    sample = "The delivery was VERY late, and I am extremely unhappy about it!!!"
    normalized = normalize_text(sample)
    no_stopwords = remove_stopwords(normalized)

    print("Original:  ", sample)
    print("Normalized:", normalized)
    print("No stop:   ", no_stopwords)
