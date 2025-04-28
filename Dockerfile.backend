# syntax=docker/dockerfile:1.3
FROM python:3.10-slim

# set directory to /app
WORKDIR /app

# system dependencies
RUN apt-get update && apt-get install -y \
    git \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Accept build arguments
ARG HUGGINGFACE_TOKEN
ARG DJANGO_SECRET_KEY

# Debug: Print token info without exposing the entire token
RUN echo "HF_API_KEY exists: $(if [ -n "$HUGGINGFACE_TOKEN" ]; then echo "yes"; else echo "no"; fi)" && \
    echo "HF_API_KEY length: ${#HUGGINGFACE_TOKEN}" && \
    if [ -n "$HUGGINGFACE_TOKEN" ]; then \
        echo "HF_API_KEY first 4 chars: $(echo $HUGGINGFACE_TOKEN | cut -c1-4)" && \
        echo "HF_API_KEY first 10 chars: $(echo $HUGGINGFACE_TOKEN | cut -c1-10)" && \
        echo "HF_API_KEY last 4 chars: $(echo $HUGGINGFACE_TOKEN | rev | cut -c1-4 | rev)"; \
    fi

# Test direct API connectivity
RUN if [ -n "$HUGGINGFACE_TOKEN" ]; then \
    echo "Testing API connection directly..." && \
    curl -v https://huggingface.co/api/whoami-v2 -H "Authorization: Bearer $HUGGINGFACE_TOKEN" 2>&1 | grep -v "< Authorization:"; \
    fi

# Set environment variables - preserve original token exactly as received
ENV HUGGINGFACE_API_KEY="${HUGGINGFACE_TOKEN}" \
    HUGGINGFACE_HUB_TOKEN="${HUGGINGFACE_TOKEN}" \
    HF_HUB_TOKEN="${HUGGINGFACE_TOKEN}" \
    HF_HOME=/app/huggingface_cache \
    TRANSFORMERS_CACHE=/app/huggingface_cache \
    HF_HUB_CACHE=/app/huggingface_cache \
    SENTENCE_TRANSFORMERS_HOME=/app/huggingface_cache \
    DJANGO_SECRET_KEY="${DJANGO_SECRET_KEY}"

ENV DJANGO_SETTINGS_MODULE=brewing.settings \
    PORT=8080 \
    PYTHONTRACEMALLOC=10 \
    PYTHONIOENCODING=utf-8 \
    PYTHONUNBUFFERED=1

# Model cache directory
RUN mkdir -p /app/huggingface_cache && \
    chmod -R 777 /app/huggingface_cache

# Copy and install requirements first for better caching
COPY requirements.txt .
RUN pip install --upgrade pip && \
    pip install -r requirements.txt && \
    pip install --no-cache-dir sentence-transformers requests

# copy project files
COPY . .

# run the download_model.py script to download the model files
RUN python download_model.py

# copy the faiss index and build_faiss.py script
COPY build_faiss.py .
COPY brewing/data data/   

# run the build_faiss.py script to build the faiss index
RUN python build_faiss.py

# Move the faiss index
RUN mkdir -p /app/brewing/faiss_store && \
    if [ -d "faiss_store" ]; then \
        mv faiss_store/* /app/brewing/faiss_store/ || echo "Could not move files"; \
    else \
        echo "faiss_store directory not found"; \
        touch /app/brewing/faiss_store/dummy_index.faiss; \
    fi

WORKDIR /app/brewing

# expose port 8080 for the server
EXPOSE 8080

# start the server using gunicorn
CMD ["sh", "-c", "exec gunicorn brewing.wsgi:application --bind 0.0.0.0:${PORT:-8080} --workers 1 --timeout 1800 --threads 4 --preload --capture-output --log-level debug --access-logfile - --error-logfile -"]



