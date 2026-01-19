FROM python:3.11.9-slim

WORKDIR /app

# System Dependencies
RUN apt-get update && apt-get install -y \
    build-essential \
    curl \
    && rm -rf /var/lib/apt/lists/*

# Python Dependencies
COPY pyproject.toml ./
RUN pip install --no-cache-dir poetry==2.3.0 && \
    poetry config virtualenvs.create false && \
    poetry lock && \
    poetry install --only main --no-interaction --no-ansi --no-root

# Application Code
COPY . .

# Model Cache Directory
RUN mkdir -p /models
ENV TRANSFORMERS_CACHE=/models

# Expose Port
EXPOSE 8000

# Health Check
HEALTHCHECK --interval=30s --timeout=10s --start-period=60s --retries=3 \
    CMD curl -f http://localhost:8000/health || exit 1

# Start Service
CMD ["python", "main.py"]
