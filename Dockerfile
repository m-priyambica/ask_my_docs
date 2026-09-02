# Hugging Face Spaces, SDK = docker.
#
# Two things about Spaces drive most of this file:
#   1. Spaces routes traffic to port 7860, not 8000.
#   2. The container runs as a non-root user, and anything that writes at
#      runtime -- the model cache above all -- must live somewhere that user
#      owns. Skipping this is the classic "works locally, PermissionError on
#      Spaces" failure.

FROM python:3.11-slim

# git is needed by huggingface_hub for some model pulls; the rest is build
# tooling for wheels that have no prebuilt binary for slim images.
RUN apt-get update && apt-get install -y --no-install-recommends \
        git build-essential \
    && rm -rf /var/lib/apt/lists/*

# Spaces gives the running process UID 1000. Create it and own everything.
RUN useradd -m -u 1000 appuser
USER appuser
ENV HOME=/home/appuser \
    PATH=/home/appuser/.local/bin:$PATH \
    PYTHONUNBUFFERED=1 \
    HF_HOME=/home/appuser/.cache/huggingface

WORKDIR $HOME/app

# Dependencies first, so edits to the app don't reinstall torch every build.
COPY --chown=appuser requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

COPY --chown=appuser . .

# Bake the two BGE models into the image rather than downloading them on first
# request. They are ~1.2 GB and would otherwise make the first user wait for a
# download while the page sits on the "Load reranker" stage.
RUN python -c "\
from sentence_transformers import SentenceTransformer, CrossEncoder; \
SentenceTransformer('BAAI/bge-small-en-v1.5'); \
CrossEncoder('BAAI/bge-reranker-base')"

EXPOSE 7860

# One worker on purpose: retrievers are held in this process's memory
# (see RETRIEVERS in api.py), so a second worker would not see a document
# indexed by the first.
CMD ["uvicorn", "api:app", "--host", "0.0.0.0", "--port", "7860", "--workers", "1"]
