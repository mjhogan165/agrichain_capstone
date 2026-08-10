import string
import nltk
from nltk.corpus import stopwords, words

# Checks if the NLTK stopword list is already downloaded, and if not, downloads it.
try:
    nltk.data.find("corpora/stopwords")
except LookupError:
    nltk.download("stopwords")

STOPWORDS = set(stopwords.words("english"))


def normalize_text(text: str) -> str:
    # lowercase, strip punctuation, and removes extra whitespace.

    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    text = " ".join(text.split())  # collapses multiple spaces into one
    return text


def remove_stopwords(text: str) -> str:
    words = text.split()

    filtered = []
    for w in words:
        if w not in STOPWORDS:
            filtered.append(w)
    return " ".join(filtered)
