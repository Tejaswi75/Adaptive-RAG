# Single container (e.g. Hugging Face Spaces Docker SDK, Render, Railway):
# FastAPI backend on 127.0.0.1:8000, Streamlit UI on the public port 7860.
FROM python:3.11-slim

RUN useradd -m -u 1000 user
USER user
ENV HOME=/home/user \
    PATH=/home/user/.local/bin:$PATH \
    HF_HOME=/home/user/.cache/huggingface \
    PYTHONUNBUFFERED=1
WORKDIR /home/user/app

COPY --chown=user requirements.txt .
RUN pip install --no-cache-dir --user -r requirements.txt

# Download the embedding model at build time so the first request is fast
RUN python -c "from fastembed import TextEmbedding; TextEmbedding('sentence-transformers/all-MiniLM-L6-v2')"

COPY --chown=user . .

EXPOSE 7860
CMD ["sh", "start.sh"]
