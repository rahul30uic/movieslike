# Movieslike API — Hugging Face Space (Docker SDK, CPU)
# Build context: repo root  (docker build -f deploy/Dockerfile .)
FROM python:3.12-slim

WORKDIR /app

# CPU-only torch keeps the image lean
RUN pip install --no-cache-dir torch --index-url https://download.pytorch.org/whl/cpu && \
    pip install --no-cache-dir \
    fastapi uvicorn[standard] python-multipart pillow numpy sentence-transformers

COPY api/ /app/api/
COPY deploy/data/ /app/data/

# Pre-download models into the image so cold starts don't fetch from HF hub
RUN python -c "from sentence_transformers import SentenceTransformer as S; \
    S('BAAI/bge-base-en-v1.5'); S('google/siglip-base-patch16-224')"

ENV ALLOWED_ORIGINS="http://localhost:3000"
EXPOSE 7860
CMD ["uvicorn", "main:app", "--host", "0.0.0.0", "--port", "7860", "--app-dir", "api"]
