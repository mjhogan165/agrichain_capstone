# Base image: a minimal Debian Linux with Python 3.12 pre-installed.
# "slim" = stripped down (no GUI libs, docs, etc.) - keeps the image smaller.
FROM python:3.12-slim

# All following commands run from /app inside the container's filesystem.
# This is an isolated filesystem - it does NOT touch your Windows folders.
WORKDIR /app

# faiss-cpu and torch need this system-level library (OpenMP) to run.
# --no-install-recommends keeps out extra packages we don't need.
# The rm -rf at the end deletes apt's package cache to keep the image smaller.
RUN apt-get update && apt-get install -y --no-install-recommends \
    libgomp1 \
    && rm -rf /var/lib/apt/lists/*

# --- Dependency layer (changes rarely -> cached most of the time) ---
COPY requirements-docker.txt requirements.txt
RUN pip install --no-cache-dir -r requirements.txt

# --- Package registration layer ---
# Mirrors your local `pip install -e .` setup, so `from src.agents...` imports work.
COPY pyproject.toml .
COPY src/ ./src/
RUN pip install -e . --no-deps

# --- Application code + assets (changes most often -> copied last) ---
COPY streamlit_apps/ ./streamlit_apps/
COPY models/ ./models/
COPY data/ ./data/

# Streamlit config via env vars: don't try to open a browser inside the
# container (there isn't one), and listen on all network interfaces so
# traffic from outside the container can actually reach it.
ENV STREAMLIT_SERVER_HEADLESS=true
ENV STREAMLIT_SERVER_ADDRESS=0.0.0.0
ENV KMP_DUPLICATE_LIB_OK=TRUE
ENV OMP_NUM_THREADS=1
# Documents which ports this image is meant to serve on (informational -
# actual publishing to your machine happens in docker-compose, next step).
EXPOSE 8501 8502 8503 8504